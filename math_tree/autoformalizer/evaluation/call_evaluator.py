from mp_evaluator import ProverEvaluator

if __name__ == "__main__":
    # Path to the configuration YAML file
    config_path = "configs/mpconfig_inhouse.yaml"

    # Path to the output directory
    output_dir = "eval_logs"

    # Overwrite options
    overwrite_options = {
        "model_path": "deepseek-ai/DeepSeek-Prover-V1.5-RL",
        "sampling_config.temperature": 1.2,
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    }

    # Create and run the evaluator
    evaluator = ProverEvaluator(config_path, output_dir, **overwrite_options)
    evaluator.evaluate()
