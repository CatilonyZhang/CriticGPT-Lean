# Autoformalizer and Prover Pipeline

This project consists of two main components: **Autoformalizer** and **Prover**, which work together in an end-to-end pipeline for automatically formalizing mathematical statements, generating proofs, and verifying them.

---

## Launching vLLM Server

First, you need to launch the vLLM server to serve the **autoformalizer** and **prover** models on the designated GPUs.

### Autoformalizer Model

Run the following command to start the **autoformalizer** model:

```bash
CUDA_VISIBLE_DEVICES=6,7 vllm serve --host 0.0.0.0 \
    --port 8081 AI-MO/Qwen7BCoder_Autoformalizer \
    --revision V2B5 \
    --tensor-parallel-size 2
```

- `CUDA_VISIBLE_DEVICES=6,7`: Assigns GPUs 6 and 7 to this model.
- `--host 0.0.0.0`: Makes the server accessible on all network interfaces.
- `--port 8081`: The port on which the autoformalizer model will be served.
- `--tensor-parallel-size 2`: Specifies the number of GPUs for tensor parallelism.

### Prover Model

Next, start the **prover** model:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 vllm serve --host 0.0.0.0 \
    --port 8082 deepseek-ai/DeepSeek-Prover-V1.5-RL \
    --tensor-parallel-size 4
```

- `CUDA_VISIBLE_DEVICES=2,3,4,5`: Assigns GPUs 2, 3, 4, and 5 to the prover model.
- `--host 0.0.0.0`: Makes the server accessible on all network interfaces.
- `--port 8082`: The port on which the prover model will be served.
- `--tensor-parallel-size 4`: Specifies the number of GPUs for tensor parallelism.

---

## Setting Up the Configuration

### Step 1: Create Your Configuration

You need to create a configuration file for the pipeline. You can copy one of the available templates located in `autoformalizer/jobs/configs/`.

For example, you can copy the `numina_config_template.yaml` to create your custom configuration.

## Running the Pipeline

Once your configuration is ready, you can run the pipeline with the following command:

```bash
python -m autoformalizer.jobs.main autoformalizer/jobs/configs/numina_config_template.yaml
```

This command will:

1. **Shard Input Data**: The dataset will be split into smaller shards for efficient processing.
2. **Autoformalization**: Each shard will be processed to formalize the mathematical statements.
3. **Verification**: Formalized statements will be verified for correctness.
4. **Proof Generation**: Proofs for the formalized statements will be generated using the prover.
5. **Proof Verification**: Generated proofs will be verified for validity.

### Monitor the Pipeline

During the process, you can monitor the progress by running the following command in a separate terminal:

```bash
python autoformalizer/jobs/monitoring.py {working_dir}
```

This will give you real-time updates on the status of the pipeline.

---

## Performance Metrics and Statistics

Once the pipeline completes, you can calculate some statistics to evaluate the performance of the pipeline.

### Key Metrics:

- **Valid Rate**: The percentage of verified proofs that are valid.
- **Proof Rate by UUID**: The percentage of unique identifiers (UUIDs) that have at least one valid proof.
- **Negative Statement Prove Rate**: The success rate of proving negative statements.

Example output:

```
Valid rate: 0.36
Prove rate by uuid: 0.55
Negative statement prove rate: 0.35
Total processing time: 80.88 seconds
```

---

## TODO: Data Balancer

We still need to implement a **data balancer** to ensure that the data is distributed evenly across shards. This will help in optimizing the processing time by balancing the workload between different workers.

--- 

### Helpful Commands:

1. **Sharding**: Split your dataset into manageable chunks.
2. **Monitoring**: Track the progress of the pipeline.
3. **Logging**: Logs are stored in `pipeline.log` for debugging and performance analysis.

