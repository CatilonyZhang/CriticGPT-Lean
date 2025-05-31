from datasets import Dataset
import json

path = './output_logs/Qwen2.5-Coder-7B_20241113_024348/results.json'

with open(path, 'r') as f:
    data = json.load(f)

results = data['results']
for t in results:
    e = t['error_message']
    if isinstance(e, str):
        if e == "":
            t['error_message'] = []
        else:
            t['error_message'] = [e]

print(results)
ds = Dataset.from_list(results)
ds.push_to_hub("miniF2F-eval-qwen25-coder-7b", revision='main', private=True)
