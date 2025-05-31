#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate autoformalization

model_name='gpt-4o'

python3 block_tree/refine.py --model_name $model_name --use_openai