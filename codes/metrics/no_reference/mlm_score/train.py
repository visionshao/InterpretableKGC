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
import wandb
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
    Trainer
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
    model = AutoModelForMaskedLM.from_pretrained(model_path)
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
        return tokenizer([" ".join(x[0]) + "\n" + y[0] for x, y in zip(examples["dialogue_context"], examples["response"])])

    tokenized_data = data.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=data["train"].column_names,
    )

    block_size = 128


    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_dataset = tokenized_data.map(group_texts, batched=True, num_proc=4)
    

    ############################################################ Data collator ############################################################
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    
    training_args = TrainingArguments(
        output_dir="/mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/metrics/no_reference/mlm_score/save_models/roberta_base_wow_mlm_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["valid"],
        data_collator=data_collator,
    )

    trainer.train()


    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")



if __name__ == "__main__":
    main()