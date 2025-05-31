#!/bin/bash

# Set parameters
BOOTSTRAP_ID=5
BOOTSTRAP_NAME="V1B4"
SAVE_DS_BRANCH="V1B5"

MODEL_DIR="/home/mert/models/Qwen7BCoder_Autoformalizer${BOOTSTRAP_NAME}/"       
REF_DS="AI-MO/autoformalization-olympiads-v0.1-filtered"

BS_DS="AI-MO/AutoformalizationV1B0"
BS_DS_BRANCH="$BOOTSTRAP_NAME"

SAVE_DS="$BS_DS"
SAVE_DS_BRANCH="$SAVE_DS_BRANCH"

SAVE_ALPACA_DS="AI-MO/Autoformalization${SAVE_DS_BRANCH}_Alpaca"

N_SAMPLES=10000

# Run bootstrapping
cd /home/mert/autoformalizer
python -m autoformalizer.bootstrap.bootstrap \
    --model_path="$MODEL_DIR" \
    --ref_ds="$REF_DS" \
    --bs_ds="$BS_DS" \
    --bs_ds_branch="$BS_DS_BRANCH" \
    --save_ds="$SAVE_DS" \
    --save_ds_branch="$SAVE_DS_BRANCH" \
    --n_samples="$N_SAMPLES" \
    --bootstrap_id="$BOOTSTRAP_ID" \
    --model_name="claude-3-5-sonnet-latest"

# Convert to Alpaca
python -m autoformalizer.data_utils.convert_alpaca \
    --dataset_id="$SAVE_DS" \
    --split='train' \
    --dataset_branch="$SAVE_DS_BRANCH" \
    --output_dataset_id="$SAVE_ALPACA_DS" 