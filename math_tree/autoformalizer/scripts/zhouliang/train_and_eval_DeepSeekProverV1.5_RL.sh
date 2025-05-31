#!/bin/bash

: '
This script runs training and evaluation for the whole proof model with one cmd command. Currently,
the script is set to run a proof pipeline, but the user can adopt the code for their desired
use case by checking the variables below in accordance with the documentation.
own use case. 
'
source ~/miniconda3/etc/profile.d/conda.sh
conda activate autoformalization
cd /lustre/fast/fast/txiao/zly/lean/math_tree/autoformalizer
YAML_FILE="/lustre/fast/fast/txiao/zly/lean/math_tree/autoformalizer/train/config/zhouliang/DeepSeekProverV1.5_RL_SFT.yaml"

# Run training
# cd /lustre/fast/fast/txiao/zly/lean/math_tree/autoformalizer/train
# FORCE_TORCHRUN=1 llamafactory-cli train "$YAML_FILE"

# # Run inference
cd /lustre/fast/fast/txiao/zly/lean/math_tree/autoformalizer/evaluation
python evaluator.py --config /lustre/fast/fast/txiao/zly/lean/math_tree/autoformalizer/evaluation/configs/slow_eval/config_minif2f.yaml --output-dir eval_logs


