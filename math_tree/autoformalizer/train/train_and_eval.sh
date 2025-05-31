#!/bin/bash

: '
This script runs training and evaluation for the Autoformalizer model with one cmd command. Currently,
the script is set to run a bootstrapping pipeline, but the user can adopt the code for their desired
use case by checking the variables below in accordance with the documentation.
own use case. 
'

BOOTSTRAP_NAME="V2B0"
YAML_FILE="/home/mert/autoformalizer/train/config/mert/autof_bootstrap/Qwen7B_Autof${BOOTSTRAP_NAME}.yaml"
MODEL_DIR="/home/mert/models/Qwen7BCoder_Autoformalizer${BOOTSTRAP_NAME}/"
EVAL_DATASET="AI-MO/AutoformalizationEvalV2"
SAVE_DATASET="$EVAL_DATASET"
MODEL_NAME="Qwen7BCoder_Autoformalizer${BOOTSTRAP_NAME}"
SAVE_BRANCH="$MODEL_NAME"
MODEL_SAVE_PATH="AI-MO/Qwen7BCoder_AutoformalizerBootstrap"
MODEL_SAVE_BRANCH="$BOOTSTRAP_NAME"

# EXECUTION

# Run training
cd /home/mert/autoformalizer/train
FORCE_TORCHRUN=1 llamafactory-cli train "$YAML_FILE"

# Run inference
cd /home/mert/autoformalizer
python -m autoformalizer.model_utils.infer_hf_dataset \
--model_path="$MODEL_DIR"  \
--dataset_id="$EVAL_DATASET" \
--output_dataset_id="$SAVE_DATASET" \
--output_dataset_branch="$SAVE_BRANCH" \

# Run Evaluation
python -m autoformalizer.eval_utils.all_feedback \
--input_dataset_id="$EVAL_DATASET" \
--input_dataset_branch="$SAVE_BRANCH" \
--output_dataset_id="$EVAL_DATASET" \
--output_dataset_branch="$SAVE_BRANCH" 