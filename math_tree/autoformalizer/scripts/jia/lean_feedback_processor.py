import fire
from datasets import load_dataset
from autoformalizer.eval_utils.lean_feedback import lean4_feedback


def compute_feedback(date: str):
    """
    Computes Lean feedback for each sample in the dataset and pushes the updated dataset to the Hugging Face Hub.

    Args:
        date (str): The date identifier used to specify the dataset version (e.g., '241025').
    
    Returns:
        None: Prints the dataset with feedback and uploads it to the Hugging Face Hub as a private dataset.
    """
    # Load the dataset specified by date
    ds = load_dataset(f"AI-MO/numina-autof-{date}", split="train")

    # Process each sample, obtaining Lean feedback for the Lean code
    def _process(sample):
        lean_code = sample["lean_code"]
        feedback = lean4_feedback(lean_code)
        return {"lean4_feedback": feedback}

    # Map the feedback processing across the dataset using 16 processes
    ds = ds.map(_process, num_proc=16)
    print(ds)

    # Push the processed dataset back to the Hugging Face Hub
    ds.push_to_hub(f"AI-MO/numina-autof-{date}", private=True)


def filter_errors(date: str):
    """
    Filters out samples containing errors in Lean feedback and uploads the filtered dataset to the Hugging Face Hub.

    Args:
        date (str): The date identifier used to specify the dataset version (e.g., '241025').
    
    Returns:
        None: Prints the filtered dataset and a random sample without errors, then uploads it as a private dataset.
    """
    # Load the dataset specified by date
    ds = load_dataset(f"AI-MO/numina-autof-{date}", split="train")
    print(ds)

    # Filter out samples where 'lean4_feedback' contains the word 'error'
    ds = ds.filter(lambda x: "error" not in str(x['lean4_feedback']))
    print(ds)

    # Print a random sample from the filtered dataset
    sample = ds[220]
    print("lean_code\n", sample['lean_code'])
    print("lean4_feedback\n", sample['lean4_feedback'])

    # Push the filtered dataset back to the Hugging Face Hub
    ds.push_to_hub(f"AI-MO/numina-autof-correct-{date}", private=True)

if __name__ == "__main__":
    """
    Usage:
        python scripts/jia/lean_feedback_processor.py compute_feedback <date>
        python scripts/jia/lean_feedback_processor.py filter_errors <date>

    Examples:
        python scripts/jia/lean_feedback_processor.py compute_feedback 241025
        python scripts/jia/lean_feedback_processor.py filter_errors 241025
    """
    fire.Fire({"compute_feedback": compute_feedback, "filter_errors": filter_errors})
