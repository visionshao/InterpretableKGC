import random
import sys

import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import copy
import torch
import math
import os
import logging
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


class GPTData(Dataset):
    def __init__(self, data, tokenizer, context_len=256, sent_len=64, max_length=1024, test=False, psg_filter=None,
                 psg_num=1, use_oracle=False, shuffle_id=False, max_id=128, add_aux_loss=False, gpt_style=False,
                 add_label=True, add_response=False, add_label_to_prefix=None, add_hyperlink=False,
                 use_pred_label=None, dialogue_first=True, knowledge_response=False, second_id=False, drop_null=True,
                 max_num_of_know=None):
        super(Dataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.sent_len = sent_len
        self.max_length = max_length
        self.test = test
        self.psg_filter = psg_filter
        self.psg_num = psg_num
        self.use_oracle = use_oracle
        self.shuffle_id = shuffle_id
        self.max_id = max_id
        self.add_aux_loss = add_aux_loss
        self.gpt_style = gpt_style
        self.add_response = add_response
        self.add_label = add_label
        self.response = [example['labels'][0] for example in self.data]
        self.add_label_to_prefix = add_label_to_prefix
        self.add_hyperlink = add_hyperlink
        self.use_pred_label = use_pred_label
        self.dialogue_first = dialogue_first
        self.knowledge_response = knowledge_response
        self.second_id = second_id
        self.drop_null = drop_null
        self.max_num_of_know = max_num_of_know

    def __getitem__(self, index):
        example = self.data[index]
        hf_data = dict()
        # =============================
        # Build knowledge
        # =============================

        knowledge = example['knowledge']
        if self.psg_filter is not None:
            print("self.psg filter is not None")
            positive = [k for k in knowledge if k == example['title']]
            titles = positive + [k for k in knowledge if k != example['title']]
            titles = [titles[pid] for pid in self.psg_filter[index]][:self.psg_num]
            if self.use_oracle and example['title'] != 'no_passages_used' and \
                    example['title'] in knowledge and example['title'] not in titles:
                titles = [example['title']] + titles[:-1]
            new_knowledge = OrderedDict()
            for k in titles:
                new_knowledge[k] = knowledge[k]
            knowledge = new_knowledge
        else:
            # enter here
            # print(f"self.psg filter is {self.psg_filter}")
            titles = [k for k in knowledge][:self.psg_num]
            # requirements: 1-use knowledge, 2-used knowledge in knowledge base, 3-curent topic not in knowledge base, 4-curent topic not in selected topics
            if self.use_oracle and example['title'] != 'no_passages_used' and \
                    example['title'] in knowledge and example['title'] not in titles:
                titles = [example['title']] + titles[:-1]
            
            new_knowledge = OrderedDict()
            for k in titles:
                new_knowledge[k] = knowledge[k]
            knowledge = new_knowledge

            # only get one topic's knowledge

        if self.drop_null and not self.test and example['title'] != 'no_passages_used':
            print("drop_null and train and has a topic") # Not execute
            if example['title'] not in knowledge or example['checked_sentence'] not in knowledge[example['title']]:
                return self[np.random.randint(len(self))]

        id_map = [i for i in range(2, self.max_id)]
        if self.shuffle_id:
            np.random.shuffle(id_map)
        id_map = [0, 1] + id_map # knowledge sentence id, keep getting the first 2 sentences and randomly get the following sentence

        # =============================
        # Passage sequence
        # =============================

        sequence = []
        sent_id = 0
        label = f'<s{id_map[0]}>'
        sentence_to_id = {}
        sequence_str_list = []
        # sequence_str += '\nPassage information.\n'
        sequence += self.tokenizer.encode('\nPassage information.\n',
                                          add_special_tokens=False)

        sentence = 'no_passages_used'
        sent_id += 1
        sequence_str_list += [f'{sentence}']
        sequence += self.tokenizer.encode(f'<s{id_map[sent_id]}>{sentence}<s{id_map[sent_id]}>\n',
                                          add_special_tokens=False)

        sentence_to_id[sentence] = sent_id
        if sentence == example['checked_sentence']:
            label = f'<s{id_map[sent_id]}>'

        second_best = ''
        second_best_score = 0
        for pid, (title, passage) in enumerate(knowledge.items()):
            # sequence_str += f'Passage {pid + 1}, Title: {title}\n'
            sequence += self.tokenizer.encode(f'Passage {pid + 1}, Title: {title}\n', add_special_tokens=False)
            # np.random.shuffle(passage)
            for sentence in passage:
                if len(sequence) > self.max_length:
                    break
                sent_id += 1
                sequence_str_list += [f'{sentence}']
                sequence += self.tokenizer.encode(f'<s{id_map[sent_id]}>{sentence}',
                                                  truncation=True, max_length=self.sent_len, add_special_tokens=False)
                sequence += self.tokenizer.encode(f'<s{id_map[sent_id]}>\n', add_special_tokens=False)
                sentence_to_id[sentence] = sent_id
                if sentence == example['checked_sentence']:
                    label = f'<s{id_map[sent_id]}>'
                elif self.second_id and \
                        f1_score(sentence + example['checked_sentence'], [example['labels'][0]]) > second_best_score:
                    second_best = f'<s{id_map[sent_id]}>'
                    second_best_score = f1_score(sentence + example['checked_sentence'], [example['labels'][0]])
                if self.max_num_of_know is not None and sent_id >= self.max_num_of_know:
                    break
            if self.max_num_of_know is not None and sent_id >= self.max_num_of_know:
                break

        passage_sequence = copy.deepcopy(sequence)

        dialogue_context_str_list = []

        if self.second_id:
            label = label + second_best

        # =============================
        # Dialogue sequence
        # =============================

        role = {'0_Wizard': 'User1: ', '1_Apprentice': 'User2: ', '0_Apprentice': 'User2: ', '1_Wizard': 'User1: ',
                0: 'User1: ', 1: 'User2: ', 'user1': 'User1: ', 'user2': 'User2: '}
        context = ''
        for turn in example['context']:
            speaker = role.get(turn['speaker'], turn['speaker'])
            text = turn['text']
            dialogue_context_str_list += [f'{speaker}{text}']
            kk = ''
            if self.add_hyperlink and 'title' in turn:
                kk = f"[{turn['title']}]"
                if turn['checked_sentence'] in sentence_to_id:
                    kk += f"<s{id_map[sentence_to_id[turn['checked_sentence']]]}>"
                kk += ' '
            context += f'{speaker}{kk}{text}\n'
            

        topic = 'Chosen topic: ' + example['chosen_topic'] + '\n'
        sequence = []
        sequence += self.tokenizer.encode('\nDialogue history.\n',
                                          add_special_tokens=False)
        sequence += self.tokenizer.encode(topic, add_special_tokens=False)
        sequence += self.tokenizer.encode(context, add_special_tokens=False)[-self.context_len:]
        sequence += self.tokenizer.encode('Predict the next knowledge sentence id and response of User1.\n',
                                          add_special_tokens=False)

        sequence_str = ""
        sequence_str += '\nDialogue history.\n'
        sequence_str += topic
        sequence_str += context
        sequence_str += 'Predict the next knowledge sentence id and response of User1.\n'

        if self.add_label_to_prefix:
            if isinstance(self.add_label_to_prefix, list):
                pred_label = self.add_label_to_prefix[index]
                # pred_label = '<s5>'
                sequence += self.tokenizer.encode(f'Selected knowledge = {pred_label}\n', add_special_tokens=False)
            else:
                sequence += self.tokenizer.encode(f'Selected knowledge = {label}\n',
                                                  add_special_tokens=False)
        sequence_str += f'Selected knowledge = {label}\n'
        dialogue_sequence = copy.deepcopy(sequence)

        # print(sequence_str)
        print(dialogue_context_str_list)

        response = example['labels'][0] 
        # =============================
        # Build input output sequence
        # =============================

        sequence = []
        passage_sequence = passage_sequence[:self.max_length - len(dialogue_sequence)]
        if self.dialogue_first:
            sequence += dialogue_sequence
            sequence += passage_sequence
        else:
            sequence += passage_sequence
            sequence += dialogue_sequence
        target = []

        if self.add_label:
            if isinstance(self.use_pred_label, list):
                target.append(self.use_pred_label[index][0])
                # target.append('<s5>')
            else:
                target.append(f'{label}')
        print(target)
        if self.add_response:
            if self.knowledge_response and example['checked_sentence'] != 'no_passages_used' and \
                    np.random.random() < self.knowledge_response:
                target.append(f"{example['checked_sentence']}")
            else:
                target.append(f"{example['labels'][0]}")
        target = ' '.join(target)
        print(target)

        if self.gpt_style:
            sequence += self.tokenizer.encode('</s>', add_special_tokens=False)
            labels = [-100] * len(sequence)
            sequence += self.tokenizer.encode(target, add_special_tokens=False)
            labels += self.tokenizer.encode(target, add_special_tokens=False)
        else:  # bart style
            sequence = sequence
            labels = self.tokenizer.encode(target, truncation=True, max_length=self.context_len,
                                           add_special_tokens=True)

        return torch.tensor(sequence), torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def collate_fn(self, data):
        padding_value = self.tokenizer.pad_token_id
        input_ids, labels = zip(*data)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(padding_value),
            'labels': labels,
        }


def get_hf_datasets(data, output_file):
    psg_num = 1
    hf_dataset = []
    for example in tqdm(data):
        hf_data = dict()
        # =============================
        # Build knowledge
        # =============================

        knowledge = example['knowledge']

        # enter here
        # print(f"self.psg filter is {self.psg_filter}")
        titles = [k for k in knowledge][:psg_num]
        # requirements: 1-use knowledge, 2-used knowledge in knowledge base, 3-curent topic not in knowledge base, 4-curent topic not in selected topics
        if example['title'] != 'no_passages_used' and \
                example['title'] in knowledge and example['title'] not in titles:
            titles = [example['title']] + titles[:-1]
        new_knowledge = OrderedDict()
        for k in titles:
            new_knowledge[k] = knowledge[k]
        knowledge = new_knowledge

        # only get one topic's knowledge
        # if self.drop_null and not self.test and example['title'] != 'no_passages_used':
        #     print("drop_null and train and has a topic") # Not execute
        #     if example['title'] not in knowledge or example['checked_sentence'] not in knowledge[example['title']]:
        #         return self[np.random.randint(len(self))]

        # =============================
        # Passage sequence
        # =============================
        knowledge_str_list = []
        sentence = 'no_passages_used'
        knowledge_str_list += [f'{sentence}']

        for pid, (title, passage) in enumerate(knowledge.items()):
            for sentence in passage:
                knowledge_str_list += [f'{sentence}']

        # =============================
        # Dialogue sequence
        # =============================
        dialogue_context_str_list = []
        role = {'0_Wizard': 'User1: ', '1_Apprentice': 'User2: ', '0_Apprentice': 'User2: ', '1_Wizard': 'User1: ',
                0: 'User1: ', 1: 'User2: ', 'user1': 'User1: ', 'user2': 'User2: '}
        context = ''
        for turn in example['context']:
            speaker = role.get(turn['speaker'], turn['speaker'])
            text = turn['text']
            dialogue_context_str_list.append(f'{speaker}{text}')
        dialog_context = "\n".join(dialogue_context_str_list)

        response = example['labels'][0] 

        hf_data = {"dialogue_context":[], "knowledges":[], "response":[]}
        hf_data["dialogue_context"].append(dialog_context)
        hf_data["knowledges"].append(knowledge_str_list)
        hf_data["response"].append(response)

        hf_dataset.append(hf_data)
    
        # =============================
        # Build input output sequence
        # =============================
        
    with open(output_file, "w") as f:
        json.dump({"data":hf_dataset}, f)

# def prepare_data(tokenizer, data_path):
#     psg_filter = None
#     batch_size = 2
#     epochs = 2
#     data = json.load(open(data_path))
#     dataset = GPTData(data, tokenizer, psg_filter=psg_filter, context_len=128, sent_len=64, max_length=512,
#                         psg_num=1, shuffle_id=False, max_id=128, add_aux_loss=False, gpt_style=False, use_oracle=True,
#                         add_label=True, add_response=False, add_hyperlink=True, add_label_to_prefix=False,
#                         dialogue_first=True, knowledge_response=0.0, second_id=False, drop_null=False)
#     data_loader = torch.utils.data.DataLoader(
#         dataset, collate_fn=dataset.collate_fn, batch_size=batch_size, shuffle=True, num_workers=8)

#     return data_loader


# proprocess wow
data_path = '/mnt/ai2lab/weishao4/programs/InterpretableKGC/data/wizard_of_wikipedia/processed_data'
output_dir = '/mnt/ai2lab/weishao4/programs/InterpretableKGC/data/wizard_of_wikipedia/hf_data'
os.system(f"mkdir -p {output_dir}")
# for train
data = json.load(open(f'{data_path}/train.json'))
get_hf_datasets(data, f"{output_dir}/train.json")

# for dev
data_1 = json.load(open(f'{data_path}/valid_random_split.json'))
data_2 = json.load(open(f'{data_path}/valid_topic_split.json'))

data = data_1 + data_2
get_hf_datasets(data, f"{output_dir}/valid.json")

# for test
data_1 = json.load(open(f'{data_path}/test_random_split.json'))
get_hf_datasets(data_1, f"{output_dir}/test_seen.json")
data_2 = json.load(open(f'{data_path}/test_topic_split.json'))
get_hf_datasets(data_2, f"{output_dir}/test_unseen.json")


