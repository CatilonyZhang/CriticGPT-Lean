from datasets import load_dataset, Dataset
from autoformalizer.data_utils.constants import system_prompt
from autoformalizer.data_utils.user_prompt import get_user_prompt
import fire

def convert_alpaca(dataset_id: str, 
                   split: str, 
                   output_dataset_id: str, 
                   dataset_branch: str = "main", 
                   output_dataset_branch: str = "main",
                   include_source = False):
    """
    Converts a dataset of natural language mathematics statements and their formalized Lean 4 code
    into an Alpaca-style dataset format. This format includes system prompts, instructions,
    outputs (Lean code), and inputs (empty in this case).

    Requirements for the source dataset:
        - The dataset must contain the following columns:
          1. 'natural_language': A string with the mathematical problem in natural language.
          2. 'lean_code': The corresponding formalized Lean 4 code.
          3. 'theorem_names': A list of strings with names of the theorems used in the Lean code.
          4. 'has_header': A boolean indicating whether the Lean code should include headers (import/open statements).
          5. 'source': A string identifying the source of the data.

    Args:
        dataset_id (str): ID of the source dataset.
        split (str): Dataset split to be used (e.g., 'train', 'test').
        output_dataset_id (str): ID for uploading the final dataset to the Hugging Face hub.
        include_source (bool): Whether to include the source of the data in the user prompt. (default: False)
    """
    # Load the source dataset
    dataset = load_dataset(dataset_id, split=split, revision=dataset_branch)

    # Prepare the new dataset structure
    data_dict = {
        "system": [],
        "instruction": [],
        "output": [],
        "input": []
    }

    # Iterate over the dataset and construct the new format
    for data in dataset:
        # Prepare user prompt
        user_prompt = get_user_prompt(
            data["natural_language"],
            data["has_header"],
            data["theorem_names"],
            data["source"],
            include_source
        )

        # Append formatted data to the new dataset structure
        data_dict["system"].append(system_prompt)
        data_dict["instruction"].append(user_prompt)
        data_dict["output"].append(data["lean_code"])
        data_dict["input"].append("")

    # Create a Hugging Face Dataset and push to hub
    alpaca_dataset = Dataset.from_dict(data_dict)
    
    alpaca_dataset.push_to_hub(output_dataset_id, 
                               revision=output_dataset_branch,
                               private=True, 
                               commit_message=f"Converted {dataset_id} from {dataset_branch} into Alpaca format.")

if __name__ == "__main__":
    """
    Example usage (from autoformalizer folder):

    python -m autoformalizer.data_utils.convert_alpaca \ 
    --dataset_id='AI-MO/AutoformalizationV1' \
    --split='train' \
    --dataset_branch='main' \
    --output_dataset_id='AI-MO/AutoformalizationV1_Alpaca'

    This will convert the dataset and upload it to the Hugging Face hub under the specified output dataset ID.
    """
    fire.Fire(convert_alpaca)
