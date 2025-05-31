import pathlib
from concurrent.futures import ProcessPoolExecutor

import datasets
import fire
import pandas as pd
from tqdm import tqdm


def _clean_one_shard(input_shard_path: str, output_shard_path: str) -> None:
    if output_shard_path.exists():
        return
    shard_df = pd.read_parquet(input_shard_path)
    shard_df["error"] = shard_df["error"].fillna("")
    if "selected_model" in shard_df.columns:
        shard_df["selected_model"] = shard_df["selected_model"].fillna("")
    else:
        shard_df["selected_model"] = ""
    shard_df.to_parquet(output_shard_path, index=False)


def push_auto_proof_to_hub(
    auto_proofs_dir, dataset_name, cache_dir=None, num_proc=4
) -> None:
    """
    Pushes the given auto proof to the hub.
    Merged columns: Index(['statement_id', 'uuid', 'formal_statement', 'proof_id', 'formal_proof',
       'error', 'uuid_verified', 'proof', 'lean_feedback', 'has_error',
       'is_valid_no_sorry', 'is_valid_with_sorry'],
      dtype='object')
    """
    auto_proofs_dir = pathlib.Path(auto_proofs_dir)
    working_dir = auto_proofs_dir.parent
    final_proofs_dir = working_dir / "final_proofs"
    final_proofs_dir.mkdir(exist_ok=True, parents=True)

    futures = {}
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        for shard_path in auto_proofs_dir.glob("*.parquet"):
            output_shard_path = final_proofs_dir / shard_path.name
            shard_id = shard_path.stem
            futures[shard_id] = executor.submit(
                _clean_one_shard, shard_path, output_shard_path
            )
        for shard_id, future in tqdm(futures.items(), desc="Cleaning shards"):
            future.result()

    data = datasets.load_dataset(
        str(final_proofs_dir), cache_dir=cache_dir, num_proc=num_proc
    )
    data.push_to_hub(dataset_name, private=True)


if __name__ == "__main__":
    """
    Example usage:
    python -m autoformalizer.jobs.push_auto_proof_to_hub \
        --auto_proofs_dir /mnt/moonfs/kimina-m2/jobs/new_year_human_statments_prover_v7/verified_proofs/ \
        --dataset_name AI-MO/human-proofs-moon-new-year-prover-v1 \
        --cache_dir /mnt/moonfs/kimina-m2/.cache/ttmmpp \
        --num_proc 128

    python -m autoformalizer.jobs.push_auto_proof_to_hub \
        --auto_proofs_dir ../jobs/santa_prover/verified_proofs \
        --dataset_name AI-MO/auto-proofs-santa-small-dryrun
    """

    fire.Fire(push_auto_proof_to_hub)
