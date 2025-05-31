#!/bin/bash
MODEL_NAME=${1:-"gpt-4o"}
INPUT_BRANCH=${2:-"main"}
INPUT_DATASET=${3:-"AI-MO/formalized_preview_241210"}


HOME_DIR=$HOME
cd ${HOME_DIR}/autoformalizer

#gpt
export OPENAI_API_KEY=""

#claude
#export ANTHROPIC_API_KEY=""

python3 -m evaluation.judge_model.judge_model_eval \
    --input_dataset_id "$INPUT_DATASET" \
    --input_branch "$INPUT_BRANCH" \
    --model_name "$MODEL_NAME" \
    --verbose