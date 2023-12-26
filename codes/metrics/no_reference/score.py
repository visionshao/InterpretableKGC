import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
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
    AutoModelForSequenceClassification
)

device = "cuda" if torch.cuda.is_available() else "cpu"
############################################################ model and tokenzier ############################################################
mlm_score_model_path = r'/mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/metrics/no_reference/mlm_score/save_models/roberta_base_wow_mlm_model/checkpoint-113500'
dr_score_model_path = r'/mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/metrics/no_reference/dr/save_models/roberta_base_wow_mlm_model/checkpoint-30000'
tokenizer = AutoTokenizer.from_pretrained(mlm_score_model_path, use_fast=True)
mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_score_model_path)
dr_model = AutoModelForSequenceClassification.from_pretrained(dr_score_model_path)
mlm_model.to(device)
dr_model.to(device)
loss_fct = nn.CrossEntropyLoss(reduction="none")

# for data_key in ["train_20000_to_40000", "valid", "test_seen", "test_unseen"]:
for data_key in ["train_20000_to_40000"]:
    generation_path = r'/mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/gen_response/generation_results/flan_t5_base_wizard_of_wikipedia'
    test_seen_file = os.path.join(generation_path, f"{data_key}.json")
    output_file = test_seen_file.replace(".json", "_scores.json")

    data = json.load(open(test_seen_file, "r"))
    dialogue_context_list = data['dialogue_context']
    knowledge_list = data['knowleges']
    generation_list = data["generation"]
    response_list = data["response"]



    mlm_score_list = []
    dr_score_list = []
    with torch.no_grad():
        for dc, knows, gens, response in tqdm(zip(dialogue_context_list, knowledge_list, generation_list, response_list)):
            # compute mlm scores
            prefix = tokenizer([dc]).input_ids[0]
            prefix_length = len(prefix) - 1
            inputs = tokenizer([dc + "\n" + gen for gen in gens], return_tensors="pt", padding=True, truncation=True, max_length=512)
            labels = inputs.input_ids.clone()
            labels[:, :prefix_length] = -100
            labels[labels == tokenizer.pad_token_id] = -100
            labels = labels.to(device)
            inputs.to(device)

            outputs = mlm_model(**inputs, labels=labels)
            prediction_scores = outputs.logits
            bs, length, vocab_size = prediction_scores.size()
            loss = loss_fct(prediction_scores.view(-1, vocab_size), labels.view(-1))
            loss = loss.view(bs, -1)
            scores = -loss.sum(1)
            mlm_score_list.append(scores.tolist())

            # compute dr scores
            outputs = dr_model(**inputs)
            logits = outputs.logits
            dr_scores = torch.softmax(logits, dim=1)[:, 1]
            dr_score_list.append(dr_scores.tolist())

    print(len(mlm_score_list))
    print(len(dr_score_list))

    data["mlm_scores"] = mlm_score_list
    data["dr_scores"] = dr_score_list

    json.dump(data, open(output_file, "w"), ensure_ascii=True)