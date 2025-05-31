from datasets import load_dataset, Dataset, concatenate_datasets
import fire

def merge_autof_datasets(dataset_ids: str, output_dataset_id: str):
    """
    Merges multiple datasets of natural language mathematics statements and their formalized Lean 4 code
    into a single dataset.

    Requirements for the source datasets:
        - The datasets must contain the following columns:
          1. 'natural_language': A string with the mathematical problem in natural language.
          2. 'lean_code': A string with the corresponding formalized Lean 4 code.
          3. 'theorem_names': A list of strings with names of the theorems used in the Lean code.
          4. 'has_header': A boolean indicating whether the Lean code should include headers (import/open statements).
          5. 'source': A string identifying the source of the data.

    All the other columns are omitted in the merged dataset.

    Args:
        dataset_ids (str): Comma-separated list of IDs of the source datasets.
        output_dataset_id (str): ID for uploading the final dataset to the Hugging Face hub.
    """

    # Load the source datasets
    dataset_ids = dataset_ids.split(",")
    datasets = [load_dataset(dataset_id, split = "train") for dataset_id in dataset_ids]

    for i in range(len(datasets)):
        datasets[i] = datasets[i].select_columns(["natural_language", "lean_code", "theorem_names", "has_header", "source"])

    ds_final = concatenate_datasets(datasets)

    # Upload the merged dataset to the Hugging Face hub
    ds_final.push_to_hub(output_dataset_id, 
                         private = True,
                         commit_message = f"Merged of {repr(dataset_ids)} into a single dataset.")

    print(f"Dataset {output_dataset_id} has been uploaded to the Hugging Face hub.")

if __name__ == "__main__":
    """
    Example usage

    python autoformalizer/data_utils/merge_autof_datasets.py \
        --dataset_ids="AI-MO/numina-autof-correct-241025,AI-MO/AutoformalizationV1" \
        --output_dataset_id="AI-MO/AutoformalizationV1.1"
    """
    fire.Fire(merge_autof_datasets)
