# Usage

## General Evaluation

To run the evaluation from the command line:

```bash
python mp_evaluator.py --config configs/mpconfig_inhouse.yaml --output-dir eval_logs
```

This command evaluates the model specified in the configuration file (AI-MO/Qwen_Prover_V1_241227 in this case) on the test dataset (AI-MO/wholeproof-test-data-241224) with sampling parameters specified in the configuration file. Results are pushed to a Hugging Face dataset. Additionally, results will be stored locally in the eval_logs directory. Currently there are evaluation config files:

- `evaluation/configs/mpconfig_inhouse.yaml`: Evaluate on in-house evaluation dataset.
- `evaluation/configs/mpconfig_minif2f.yaml`: Evaluate on minif2f.
- `evaluation/configs/mpconfig_proofnet.yaml`: Evaluate on proofnet.

## Overwriting Configuration Values

To overwrite model and sampling configurations programmatically, refer to the example in `call_evaluator.py`. This script demonstrates how to:

- Set a custom model path, e.g., `deepseek-ai/DeepSeek-Prover-V1.5-RL`.
- Change sampling parameters, e.g., temperature to 1.2.
- Specify GPUs for evaluation using CUDA_VISIBLE_DEVICES.

To use this, run:

```bash
python call_evaluator.py
```

This will evaluate the model with the updated configurations.

## Results

### Hugging Face Results

Evaluation results are automatically pushed to a Hugging Face dataset derived from the test dataset name, appending `-results` to it. For example, if the test dataset is `AI-MO/minif2f_test`, results will be in the new Hugging Face dataset `AI-MO/minif2f_test-results`. There are 2 branches where the results are pushed:

1. Full Results Branch
- Branch name: `<model_name>_<identifier>_full`
-	Contains detailed evaluation results for all problems, including:
-	Problem ID.
-	Formal statements.
-	Proof attempts.
-	Feedback from Lean.
-	Number of correct proofs and whether the problem was solved.
2. Summary Results Branch
-	Branch name: `<model_name>_<identifier>_summary`
-	Contains a high-level summary of the evaluation:
-	Total problems evaluated.
-	Number of successful proofs.
-	Overall success rate.
-	Pass@N metrics for the sampling settings.

### Identifier Format

The identifier used in the branch names is derived as follows:
-	Model Name: The last component of the model_path (e.g., `DeepSeek-Prover-V1.5-RL`).
-	Sampling Parameters:
-	n: Number of samples.
-	temperature: Sampling temperature.
-	top_p: Top-p sampling parameter.
-	max_tokens: Maximum tokens to generate.

The format is: `<n_samples>_<temperature>_<top_p>_<max_tokens>`

For example, if the sampling parameters are `n=10`, `temperature=1.2`, `top_p=0.9`, and `max_tokens=256`, the identifier becomes `10_1.2_0.9_256`. Combined with the model name, a full branch name could look like:
`DeepSeek-Prover-V1.5-RL_10_1.2_0.9_256_full`.