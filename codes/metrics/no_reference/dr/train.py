import os
import torch

# from data_loader import prepare_data
import random
import sys

from dataclasses import dataclass, field
import torch.nn.functional as F
import time
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
import warnings
import transformers
from collections import defaultdict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    HfArgumentParser, 
    DataCollatorForSeq2Seq,
    GenerationConfig,
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
import datasets
# import nltk
# nltk.download('wordnet')
# from utils.evaluation import f1_score, eval_f1, eval_all

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
    output_dir_path: str = field(
        default="ldc",
        metadata={"help": "dataset name"},
    )
    train_s: int = field(
        default=0,
        metadata={"help":"start index for training data"}
    )
    train_e: int = field(
        default=-1,
        metadata={"help":"end index for training data"}
    )


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    labels_for_bleu = [[label.strip()] for label in labels]
    return preds, labels, labels_for_bleu


def main():
    ############################################################ Set Arguments ############################################################
    parser = HfArgumentParser((DataTrainingArguments))
    data_args = parser.parse_args_into_dataclasses()[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ############################################################ logger ############################################################
    logger = logging.getLogger(__name__)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    ############################################################ model and tokenzier ############################################################
    model_path = data_args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.model_max_length = 512
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    logger.warning("tokenizer and model initiliazed")


    ############################################################ Data ############################################################
    data_dir = data_args.data_path
    train_file = os.path.join(data_dir, "train.json")
    valid_file = os.path.join(data_dir, "valid.json")
    test_data_key_list = ["test"] if "wizard" not in data_args.dataset_name else ["test_seen", "test_unseen"]
    test_file_dict = {test_data_key:os.path.join(data_dir, f"{test_data_key}.json") for test_data_key in test_data_key_list}

    data_files = {"train":train_file, "valid":valid_file}
    data_files.update(test_file_dict)
    
    data = datasets.load_dataset("json", data_files=data_files, field="data")

    def preprocess_function(examples):
        dialogue_contexts = examples["dialogue_context"]
        responses = examples['response']

        results = tokenizer([x[0] + "\n" + y[0] for x, y in zip(dialogue_contexts, responses)], truncation=True, max_length=512)
        results["labels"] = [1] * len(results.input_ids)

        for _ in range(2):
            random.shuffle(responses)
            tmp_results = tokenizer([x[0] + "\n" + y[0] for x, y in zip(dialogue_contexts, responses)], truncation=True, max_length=512)
            results["input_ids"] += tmp_results["input_ids"]
            results["attention_mask"] += tmp_results["attention_mask"]
            results["labels"] += [0] * len(tmp_results["input_ids"])

        return results

    tokenized_data = data.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=data["train"].column_names,
    )

    ############################################################ Data collator ############################################################
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=512)

    training_args = TrainingArguments(
        output_dir="/mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/metrics/no_reference/dr/save_models/roberta_base_wow_mlm_model",
        learning_rate=2e-5,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=10000,
        eval_steps=10000,
        load_best_model_at_end=True,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["valid"],
        data_collator=data_collator,
    )

    trainer.train()


    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")



if __name__ == "__main__":
    main()