import os
import json
import sys

from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer
from nltk import word_tokenize

data_dir = r'/mnt/ai2lab/weishao4/programs/InterpretableKGC/data/Wizard-of-Wikipedia/prepared_data'

for fname in os.listdir(data_dir):
    if fname.endswith("json"):
        fpath = os.path.join(data_dir, fname)
        data = json.load(open(fpath, "r"))
        print(f"{fname}, {len(data)}")
        print(len(data))
        # for item in data:
        #     context = item["post"]
        #     response = item["response"]
        #     knowledges = item["knowledge"]
        #     labels = item["labels"]
        #     print(f"context:\n {len(context)}")
        #     print(f"response:\n {len(response)}")
        #     print(f"labels:\n {len(labels)}")
        #     print("*"*100)

