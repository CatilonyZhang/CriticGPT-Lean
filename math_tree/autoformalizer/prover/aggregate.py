import concurrent.futures
import os

import fire
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from prover.config import Config


def load_problem_data(problem_dir):
    """Load the problem files (meta_data.json, state.jsonl, tree.json) into a dictionary."""
    meta_data_path = os.path.join(problem_dir, "meta_data.json")
    state_data_path = os.path.join(problem_dir, "state.json")
    tree_data_path = os.path.join(problem_dir, "tree.json")

    if (
        not os.path.exists(meta_data_path)
        or not os.path.exists(state_data_path)
        or not os.path.exists(tree_data_path)
    ):
        return None

    # Load meta data
    with open(meta_data_path, "r") as f:
        meta_data = f.read()

    # Load state data (jsonl)
    with open(state_data_path, "r") as f:
        state_data = f.read()

    # Load tree data
    with open(tree_data_path, "r") as f:
        tree_data = f.read()

    return {
        "uuid": problem_dir.split("/")[-1],
        "meta_data": meta_data,
        "state_data": state_data,
        "tree_data": tree_data,
    }


def process_and_batch_data(problem_dirs, batch_idx, working_dir):
    """Process data in batches and save intermediate results in parquet format."""
    batch = []
    for problem_dir in problem_dirs:
        # Load problem data
        problem_data = load_problem_data(problem_dir)
        if problem_data is None:
            continue

        # Append to batch
        batch.append(problem_data)

    df = pd.DataFrame(batch)
    parquet_path = os.path.join(working_dir, f"{batch_idx}.parquet")
    df.to_parquet(parquet_path)
    print(f"Batch {batch_idx} saved to {parquet_path}")


def upload_tree_data_to_hub(
    config_path, des_dataset_id, batch_size=100, max_workers=4, cache_dir=None
):
    """Upload tree data to Hugging Face hub."""
    config = Config.from_yaml(config_path)
    tree_data_path = config.verified_proofs_dir
    working_dir = os.path.join(config.working_dir, "hf_tree_data")
    os.makedirs(working_dir, exist_ok=True)

    # Get all problem directories
    problem_dirs = []
    shard_dirs = os.listdir(tree_data_path)
    for shard_dir in shard_dirs:
        shard_dir_path = os.path.join(tree_data_path, shard_dir)
        problem_files = os.listdir(shard_dir_path)
        for problem_file in problem_files:
            if problem_file.endswith(".txt"):
                continue
            problem_file_path = os.path.join(shard_dir_path, problem_file)
            problem_dirs.append(problem_file_path)

    # Create batches from the list of UUIDs
    batches = [
        problem_dirs[i : i + batch_size]
        for i in range(0, len(problem_dirs), batch_size)
    ]

    # Process data in parallel and batch
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        tqdm(
            executor.map(
                process_and_batch_data,
                batches,
                range(len(batches)),
                [working_dir] * len(batches),
            ),
            total=len(batches),
        )

    # Upload to Hugging Face
    # Load dataset directly from the folder containing Parquet files
    dataset = load_dataset(working_dir, cache_dir=cache_dir)

    # Upload to Hugging Face
    dataset.push_to_hub(des_dataset_id, private=True)
    print(f"Data uploaded to Hugging Face dataset {des_dataset_id}")


if __name__ == "__main__":
    """
    Example usage:
    python prover/aggregate.py upload_tree_data_to_hub \
        --config_path prover/configs/moonshot_config_template.yaml \
        --des_dataset_id AI-MO/tree_data_jeasper_step_prover_v1 \
        --batch_size 20 \
        --max_workers 64 \
        --cache_dir /mnt/moonfs/kimina-m2/.cache/ttmmpp
    """

    fire.Fire({"upload_tree_data_to_hub": upload_tree_data_to_hub})
