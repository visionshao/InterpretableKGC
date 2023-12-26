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
    GenerationConfig
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
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
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


    ############################################################ Set Generation Config ############################################################

    

    # for data_key in ["test_seen", "test_unseen", "test", "valid", "train"]:
    train_s = data_args.train_s
    train_e = data_args.train_e

    if train_e == -1:
        train_prefxi_name = ""
    else:
        train_prefxi_name = f"{train_s}_to_{train_e}"
    for data_key in ["train"]:
        if data_key in data:
            s = time.time()
            gen_data_dict = {"dialogue_context":[], "knowleges":[], "generation":[], "response":[]}                                      
            count = 0
            if data_key == "train":
                sub_train_data = data[data_key].select(range(train_s, train_e))
            for item in tqdm(sub_train_data):
                dialogue_context = item["dialogue_context"][0]
                knowledges = item["knowledges"][0]
                response = item["response"][0]
                inputs = [dialogue_context + "\n" + k for k in knowledges]
                inputs = tokenizer(inputs, return_tensors="pt", padding=True)
                inputs.to(device)
                outputs = model.generate(**inputs, max_length=128)
                gen = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                count += len(gen)
                gen_data_dict["dialogue_context"].append(dialogue_context)
                gen_data_dict["knowleges"].append(knowledges)
                gen_data_dict["generation"].append(gen)
                gen_data_dict["response"].append(response)
            e = time.time()

            print(f"Generating {count} respones for {len(data[data_key])} sample from {data_key} dataset cost {(e-s)/ 60} mins")
             
            s = time.time()
            with open(os.path.join(data_args.output_dir_path, f"{data_key}.json".replace("train", f"train_{train_prefxi_name}")), "w") as f:
                json.dump(gen_data_dict, f, ensure_ascii=True)
            e = time.time()
            print(f"Save respones for {data_key} dataset cost {(e-s)/ 60} mins")
   
if __name__ == "__main__":
    main()