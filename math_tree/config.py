import os
import sys
from datasets import load_dataset

proj_root = ''

ckpt_root = ''

# llama_3_8b = os.path.join(ckpt_root, 'Llama-3.1-8B-Instruct') 
llama_3_70b = os.path.join(ckpt_root, 'Llama-3.1-70B-Instruct') 
qwen_coder = os.path.join(ckpt_root, 'Qwen25Coder7B')
ds_prover = sys.path.append(os.path.join(ckpt_root, 'DeepSeek-Prover-V1.5-SFT'))

results_path = os.path.join(proj_root, 'results')

block_tree_path = os.path.join(results_path, 'block_tree')

proof_path = os.path.join(results_path, 'proof')

temp_results_path = os.path.join(results_path, 'temp')

log_path = os.path.join(results_path, 'logs')

data_path = os.path.join(proj_root, 'data')

if not os.path.exists(results_path):
    os.makedirs(results_path)

if not os.path.exists(temp_results_path):
    os.makedirs(temp_results_path)

if not os.path.exists(log_path):
    os.makedirs(log_path)

openai_api_key = ''
openai_api_base = ''
openai_model = 'gpt-4o'

