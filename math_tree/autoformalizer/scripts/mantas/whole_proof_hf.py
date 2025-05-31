from autoformalizer.repl_lean_feedback.create_dataset import (
    create_dataset_from_hf_dataset,
)

if __name__ == "__main__":
    # create_dataset_from_hf_dataset(
    #     dataset_name="AI-MO/numina-math-lean4",
    #     split="train",
    #     code_column="lean4_solution",
    #     output_file="scripts/mantas/numina-math-lean4.jsonl",
    #     timeout_threshold=600,
    #     mode = "parent",
    #     num_proc=32
    # )

    create_dataset_from_hf_dataset(
        dataset_name="AI-MO/auto-statements-v1",
        split="train",
        code_column="correct_proof_samples",
        output_file="scripts/mantas/auto-statements-v1.jsonl",
        timeout_threshold=600,
        mode="parent",
        num_proc=32,
        kind="auto",
    )
