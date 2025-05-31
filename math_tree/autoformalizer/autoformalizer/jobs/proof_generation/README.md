This pipeline is for searching proofs with human annotated statements.

In order to run this pipeline, you need to first have a inference server set up as in `autoformalization/jobs/README.md`

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 vllm serve --host 0.0.0.0 \
    --port 8082 deepseek-ai/DeepSeek-Prover-V1.5-RL \
    --tensor-parallel-size 4
```

Then you run 

```bash
python -m autoformalizer.jobs.proof_generation.main create_proof_record \
    autoformalizer/jobs/proof_generation/annotated_dataset_template.yaml \
    --proof_record_path path/to/proof_records.jsonl \
    --num_proc 4
```