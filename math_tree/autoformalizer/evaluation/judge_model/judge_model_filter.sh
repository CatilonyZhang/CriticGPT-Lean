#!/bin/bash

# Default arguments
TEMPERATURE=${1:-0.0}
NUM_VOTES=${2:-1}
INPUT_BRANCH=${3:-"main"}
INPUT_DATASET=${4:-"AI-MO/formalized_preview_241210"}

# Configuration variables
export BASE_SERVICE_PORT=5001
export FLASK_PORT=12700
export NUM_VLLM_WORKERS=1
export MODEL_PATH="/DATA/disk2/lujianqiao/models/judgemodel/Qwen7BCoder_JudgemodelV1"
export HOME_DIR=$HOME

# Function to check if a port is ready
check_port() {
    local port=$1
    local max_attempts=$2
    local attempt=1

    echo "Waiting for port $port to become available..."
    while [ $attempt -le $max_attempts ]; do
        if nc -z localhost $port; then
            echo "Port $port is ready!"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: Port $port not ready yet, waiting..."
        sleep 5
        attempt=$((attempt + 1))
    done

    echo "Error: Port $port did not become available after $max_attempts attempts"
    return 1
}


# Create logs directory
cd "${HOME_DIR}/autoformalizer" || exit 1
mkdir -p logs

# Start the load balancer
echo "Starting load balancer..."
nohup python3 autoformalizer/balancer/load_balancer.py \
    "$MODEL_PATH" \
    --num_vllm_worker="${NUM_VLLM_WORKERS}" \
    --base_service_port="${BASE_SERVICE_PORT}" \
    --flask_port="${FLASK_PORT}" \
    --api_key="EMPTY" \
    > logs/load_balance.log 2>&1 &

# Wait for load balancer and all worker ports to be ready
echo "Waiting for services to start..."

# Check Flask port
check_port $FLASK_PORT 12 || exit 1

# Check all worker ports
for ((worker=0; worker<NUM_VLLM_WORKERS; worker++)); do
    port=$((BASE_SERVICE_PORT + worker))
    check_port $port 20 || exit 1
done

echo "All services are ready! Starting evaluation..."

# Run evaluation
python3 -m evaluation.judge_model.judge_model_filter \
    --input_dataset_id "$INPUT_DATASET" \
    --input_branch "$INPUT_BRANCH" \
    --temperature "$TEMPERATURE" \
    --sample_size 10 \
    --num_votes "$NUM_VOTES" \
    --columns_to_use problem autoformalization_preview \
    --verbose

# Cleanup will be automatically called by the trap