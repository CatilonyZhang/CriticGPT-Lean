# Step prover

## Step 1. VLLM serve model

Serve the DeepSeek-Prover-V1.5-RL with the vllm. And obtain the `base_url` and `model_id`

## Step 2. Setup the MOONSHOT Client API_KEY
export MOONSHOT_LEAN_API_KEY=xxxxx

## Step 3. Run the prover with command:
```bash
python prover/run.py --config path/to/your/config/file.yaml
```