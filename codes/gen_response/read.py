import json

file_path = r'/mnt/ai2lab/weishao4/programs/InterpretableKGC/codes/gen_response/generation_results/flan_t5_base_wizard_of_wikipedia/test_seen.json'
data = json.load(open(file_path, "r"))
for k, v in data.items():
    print(f"{k} {len(v)}")