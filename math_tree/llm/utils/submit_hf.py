import os
import json
from datasets import Dataset
from huggingface_hub import HfApi

def submit_to_hf(dataset_path: str, repo_id: str, token: str):
    """
    Submit dataset to Hugging Face Hub.
    
    Args:
        dataset_path: Path to the JSON dataset file
        repo_id: Hugging Face repo ID (e.g. 'username/repo-name') 
        token: Hugging Face API token
    """
    # Load JSON data
    with open(dataset_path) as f:
        data = json.load(f)
    
    # Convert to Dataset format
    dataset = Dataset.from_list(data)
    
    # Push to Hub
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=token, repo_type="dataset", exist_ok=True)
    
    dataset.push_to_hub(
        repo_id,
        token=token,
        private=True
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                      help="Path to JSON dataset file")
    parser.add_argument("--repo_id", type=str, required=True,
                      help="Hugging Face repo ID (e.g. username/repo-name)")
    parser.add_argument("--token", type=str, required=True,
                      help="Hugging Face API token")
    
    args = parser.parse_args()
    submit_to_hf(args.dataset_path, args.repo_id, args.token)
