from autoformalizer.repl_lean_feedback.create_dataset import create_dataset_from_folder

if __name__ == "__main__":
    create_dataset_from_folder(
        "mathlib4",
        "scripts/marco/mathlib4.jsonl",
        recurse=True,
        num_workers=16,
    )
    # create_dataset_from_hf_dataset(
    #     dataset_name="AI-MO/numina-math-lean4-241122",
    #     split="train",
    #     code_column="lean4_solution",
    #     output_file="scripts/marco/numina-math-lean4-241122.jsonl",
    #     timeout_threshold=900,
    #     num_proc=16
    # )
