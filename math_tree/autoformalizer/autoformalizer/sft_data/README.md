# List of all SFT dataset

- mathlib: https://huggingface.co/datasets/AI-MO/mathlib-sft
- auto_gen: data generated using expert iteration pipeline.
    - https://huggingface.co/datasets/AI-MO/auto-sft-moon-santa-v1beta16
    - https://huggingface.co/datasets/AI-MO/auto-sft-moon-santa-v1beta4
    - https://huggingface.co/datasets/AI-MO/auto-sft-moon-santa-v1beta1
    - https://huggingface.co/datasets/AI-MO/auto-sft-moon-santa-v2beta1
    - https://huggingface.co/datasets/AI-MO/auto-sft-moon-santa-v3-base(with 3 most different proofs)
    - https://huggingface.co/datasets/AI-MO/auto-sft-moon-flying-ant-v1-base(with 3 most different proofs)

- human_statements: data generated using proof gen pipeline with human annotated data
    - https://huggingface.co/datasets/AI-MO/human-statements-sft-v1pick1
    - https://huggingface.co/datasets/AI-MO/human-statements-sft-v1pick10
    - https://huggingface.co/datasets/AI-MO/human-statements-sft-v1pick100

- human_proofs: proofs gathered from the numina platform, annotated by human
    - https://huggingface.co/datasets/AI-MO/human-proofs-sft



# Dataset format (TODo)

# Creating source dataset

Set up sft data directory

```bash
export SFT_DATA_DIR="/dev/shm/workspace/will/dataset/sft"
```

Then you need to run the data loaders to initialize the source datasets.
We merge all the input datasets as one and download it to the local folder(SFT_DATA_DIR). 
input datasets:
- AI-MO/auto-statements-moon-flying-ant-prover-v1
- AI-MO/auto-statements-moon-santa-prover-v1

```bash
python autoformalizer/sft_data/source/run_loaders.py
```

Then create your own filter logic similar to `autoformalizer/sft_data/pipelines/auto_gen/base.py`
You can take a look at the bottom how to run it.

# Generate tagger dataset

- Statement judge tag: The current judge service needs to be started through a temporary code. This will be updated in the future when @Jianqiao updates the code.
    - Step 1: Start judge servers:
        ```bash
        cd evaluation/judge_model
        bash launch_servers.sh
        ```
        
        Set up the JUDGE_YAML env with absolute path:
        
        ```bash
        export JUDGE_YAML="/dev/shm/workspace/will/autoformalizer/evaluation/judge_model/config/Qwen7BCoder_Judgemodel.yaml"
        ```
        
    - Step 2: The current service client has not been merged, so you need to copy autoformalizer/eval_utils/model_feedback.py from the jq/judge_model_eval_v2 branch to your current branch.

    - Run the tagger code: 
        ```bash
        python -m autoformalizer.sft_data.pipelines.auto_gen.tagger_judgement run_pipeline
        ```


  