# from data_loader import prepare_data
import random
import sys

from dataclasses import dataclass, field
import torch.nn.functional as F
import numpy as np
import json
import logging
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import copy
import torch
import math
import evaluate
import sacrebleu
import os
import logging
import wandb
import warnings
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    HfArgumentParser, 
    DataCollatorForSeq2Seq
)
import datasets
import nltk
# nltk.download('wordnet')
from utils.evaluation import f1_score, eval_f1, eval_all

# sys.path.append(os.path.dirname(os.path.realpath(__file__)))
os.environ["WANDB_PROJECT"]="InterpretableKGC"
os.environ["WANDB_MODE"] = "offline"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
# os.environ["WANDB_LOG_MODEL"] = "end"


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    model_name: str = field(
        default="mt5-large",
        metadata={"help": "model name"},
    )
    model_path: str = field(
        default="mt5-large",
        metadata={"help": "model path"},
    )
    dataset_name: str = field(
        default="ldc",
        metadata={"help": "dataset name"},
    )
    resume_checkpoint: str =field(
        default=None,
        metadata={"help": "dataset name"},
    )


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    labels_for_bleu = [[label.strip()] for label in labels]
    return preds, labels, labels_for_bleu


def main():
    ############################################################ Set Arguments ############################################################
    parser = HfArgumentParser((Seq2SeqTrainingArguments, DataTrainingArguments))
    training_args, data_args = parser.parse_args_into_dataclasses()


    ############################################################ logger ############################################################
    logger = logging.getLogger(__name__)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # logger.warning(f"Training/evaluation parameters {training_args}")


    ############################################################ model and tokenzier ############################################################
    model_path = data_args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    logger.warning("tokenizer and model initiliazed")


    ############################################################ Preprocess Function ############################################################
    def preprocess_function(examples):
        def selector(inputs, mode="random"):
            np.random.shuffle(inputs)
            return inputs[0]

        dialogue_context = [example[0] for example in examples["dialogue_context"]]
        knowledges = [example[0] for example in examples["knowledges"]]
        response = [example[0] for example in examples["response"]]

        inputs = [dialogs + "\n" + selector(knows) for dialogs, knows in zip(dialogue_context, knowledges)]
        targets = response

        model_inputs = tokenizer(inputs, text_target=targets, max_length=512, truncation=True)

        return model_inputs


    ############################################################ Data ############################################################
    data_dir = data_args.data_path
    train_file = os.path.join(data_dir, "train.json")
    valid_file = os.path.join(data_dir, "valid.json")
    # seen_test_file = os.path.join(data_dir, "test_seen.json")
    # seen_test_file = os.path.join(data_dir, "test_seen.json")
    test_data_key_list = ["test"] if "wizard" not in data_args.dataset_name else ["test_seen", "test_unseen"]
    test_file_dict = {test_data_key:os.path.join(data_dir, f"{test_data_key}.json") for test_data_key in test_data_key_list}

    data_files = {"train":train_file, "valid":valid_file}
    data_files.update(test_file_dict)
    
    data = datasets.load_dataset("json", data_files=data_files, field="data")

    data = data.map(preprocess_function, batched=True)
    data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    logger.warning(f'{len(data["train"])} train samples')
    logger.warning(f'{len(data["valid"])} valid samples')
    for test_data_key in test_data_key_list:
        logger.warning(f'{len(data[test_data_key])} {test_data_key} samples')
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    logger.warning("Data has been preprocessed")


    ############################################################ save generation and complete metrics (ppl) ############################################################
    def log_test_results(test_outputs, test_data_key):
        preds = test_outputs.predictions
        labels = test_outputs.label_ids
        metrics = test_outputs.metrics

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels, decoded_labels_for_bleu = postprocess_text(decoded_preds, decoded_labels)

        with open(f"{test_data_key}_generation.txt", "w") as f:
            for item in decoded_preds:
                f.write(item + "\n")
        
        metrics[f"{test_data_key}_ppl"] = np.exp(metrics[f"{test_data_key}_loss"])
        return metrics

    metric = evaluate.load("/mnt/ai2lab/weishao4/programs/SINMT_LLM/T5/scripts/sacrebleu.py")

    def compute_metrics(eval_preds):

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels, decoded_labels_for_bleu = postprocess_text(decoded_preds, decoded_labels)
        result = eval_all(decoded_preds, decoded_labels)

        bleu_eval = metric.compute(predictions=decoded_preds, references=decoded_labels_for_bleu)
        result["BLEU4"] = bleu_eval["score"]

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        result = {k: round(v, 4) for k, v in result.items()}
        return result


    ############################################################ Set Trainer ############################################################
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # start train
    trainer.train(resume_from_checkpoint=data_args.resume_checkpoint)


    ############################################################ Set Test ############################################################
    for test_data_key in ["test_seen", "test_unseen", "test"]:
        if test_data_key in data:
            pred_outputs = trainer.predict(data[test_data_key], metric_key_prefix=test_data_key)
            pred_result = log_test_results(pred_outputs, test_data_key)
            logger.warning(f"{test_data_key} performance: {pred_result}")
            rank_id = int(os.environ.get('LOCAL_RANK', -1))
            if rank_id == 0 or rank_id == -1:
                os.system(f"mv {test_data_key}_generation.txt {training_args.output_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()