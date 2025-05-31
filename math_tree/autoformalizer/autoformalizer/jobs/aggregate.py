import json
import pathlib
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import datasets
import fire
import numpy as np
import pandas as pd
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm


def _aggregate_proofs(group: pd.DataFrame) -> pd.Series:
    """
    Given a group of proofs that share the same 'statement_id',
    aggregate them by removing duplicates, identifying correct proofs,
    and returning one valid formal proof if any exist.

    Args:
        group (pd.DataFrame): A DataFrame corresponding to a group of proofs
            sharing the same 'statement_id'.

    Returns:
        pd.Series: A Series containing aggregated proof information:
            - uuid
            - formal_statement
            - proof_id (list of IDs)
            - n_proofs (int): total number of proofs found
            - n_correct_proofs (int)
            - correct_proof_samples (list)
            - one_formal_proof (str): an example correct proof if available
    """
    # Collect all unique formal proofs
    all_proofs = list(set(group["formal_proof"].tolist()))
    n_proofs = len(all_proofs)

    # Filter correct proofs
    group_correct = group[group["is_valid_no_sorry"]]

    # Collect correct proof samples along with their proof_id and selected_model
    correct_proof_samples = []
    for _, row in group_correct.iterrows():
        correct_proof_samples.append(
            {
                "formal_proof": row["formal_proof"],
                "proof_id": row["proof_id"],
                # "selected_model": row["selected_model"]
            }
        )
    n_correct_proofs = len(correct_proof_samples)

    # If at least one correct proof exists, pick one
    one_formal_proof = (
        correct_proof_samples[0]["formal_proof"] if n_correct_proofs > 0 else ""
    )

    return pd.Series(
        {
            "uuid": group["uuid"].iloc[0],
            "formal_statement": group["formal_statement"].iloc[0],
            "proof_id": group["proof_id"].tolist(),
            "n_proofs": n_proofs,
            "n_correct_proofs": n_correct_proofs,
            "correct_proof_samples": correct_proof_samples,
            "one_formal_proof": one_formal_proof,
        }
    )


def _aggregate_repeat(group: pd.DataFrame) -> pd.Series:
    """
    Given a group of statements that share the same 'statement_id',
    aggregate them accumulating n_proofs, n_correct_proofs, correct_proof_samples.
    and returning one valid formal proof if any exist.

    Args:
        group (pd.DataFrame): A DataFrame corresponding to a group of statements
            sharing the same 'statement_id'.

    Returns:
        pd.Series: A Series containing aggregated proof information:
            - uuid
            - formal_statement
            - proof_id (list of IDs)
            - n_proofs (int): total number of proofs found
            - n_correct_proofs (int)
            - correct_proof_samples (list)
            - one_formal_proof (str): an example correct proof if available
    """
    # Collect all unique formal proofs
    result = group.iloc[0].copy(deep=True)
    result["n_proofs"] = sum(group["n_proofs"].tolist())
    result["n_correct_proofs"] = sum(group["n_correct_proofs"].tolist())
    result["correct_proof_samples"] = [
        sample
        for samples in group["correct_proof_samples"].tolist()
        for sample in samples
    ]
    result["one_formal_proof"] = (
        result["correct_proof_samples"][0]["formal_proof"]
        if result["n_correct_proofs"] > 0
        else ""
    )
    return result


def _aggregate_statements(group: pd.DataFrame) -> pd.Series:
    """
    Given a group of statements that share the same 'uuid', aggregate them.

    This handles the possibility of multiple statements (including negations).
    It sums correct proofs across valid statements, picks an example
    formal proof if available, and collects statement-level info into a dict.

    Args:
        group (pd.DataFrame): A DataFrame of statements that share the same 'uuid'.

    Returns:
        pd.Series: A Series containing:
            - formal_statements (list): All formal statements
            - n_proofs (int): sum of all non-negation proofs
            - n_correct_proofs (int): sum of correct proofs among non-negation statements
            - formal_statements_info (str): JSON-encoded statement-level details
            - one_formal_proof (str): an example correct proof if available
    """
    # Focus on non-negation statements to compute proof stats
    non_negation = group[~group["is_negation"]]
    n_correct_proofs = non_negation["n_correct_proofs"].sum()
    n_proofs = non_negation["n_proofs"].sum()

    # Pick an example formal proof from any correct proofs
    one_formal_proof = ""
    if n_correct_proofs > 0:
        for _, row in non_negation.iterrows():
            if row["one_formal_proof"]:
                one_formal_proof = row["one_formal_proof"]
                break

    # Build dictionary of statement-wise info
    formal_statements_info_records = group[
        [
            "formal_statement",
            "n_correct_proofs",
            "n_proofs",
            "is_negation",
            "correct_proof_samples",
            "statement_id",
        ]
    ].to_dict(orient="records")

    fs_dict: Dict[str, Any] = {}
    for info in formal_statements_info_records:
        statement = info["formal_statement"]
        if statement in fs_dict:
            fs_dict[statement]["n_correct_proofs"] += info["n_correct_proofs"]
            fs_dict[statement]["n_proofs"] += info["n_proofs"]
            fs_dict[statement]["correct_proof_samples"].extend(
                info["correct_proof_samples"]
            )
        else:
            fs_dict[statement] = deepcopy(info)

    return pd.Series(
        {
            "formal_statements": group["formal_statement"].tolist(),
            "n_proofs": n_proofs,
            "n_correct_proofs": n_correct_proofs,
            "formal_statements_info": json.dumps(fs_dict),
            "one_formal_proof": one_formal_proof,
        }
    )


def statement_only_aggregate_proofs_for_one_shard(
    working_dir: str, shard_id: str, statement_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates proofs, and statements
    """
    # Build filepaths
    shard_id_path = pathlib.Path(shard_id)
    proofs_path = (
        pathlib.Path(working_dir) / "verified_proofs" / f"{shard_id_path}.parquet"
    )

    if not proofs_path.exists():
        logger.warning(f"Missing data for shard_id {shard_id}")
        return None

    # Load data
    df_proofs = pd.read_parquet(proofs_path)

    # Group proofs
    grouped_df = (
        df_proofs.groupby("statement_id")
        .apply(_aggregate_proofs, include_groups=False)
        .reset_index()
    )

    # Merge statement df with statement_info
    if statement_info is None:
        return grouped_df

    grouped_df["adjusted_statement_id"] = grouped_df["statement_id"].str.replace(
        r"^neg_", "", regex=True
    )
    df_statements_merged_info = pd.merge(
        grouped_df,
        statement_info,
        left_on="adjusted_statement_id",
        right_on="statement_id",
        how="left",
    )
    df_statements_merged_info = df_statements_merged_info.drop(
        columns=["adjusted_statement_id", "statement_id_y"]
    )
    df_statements_merged_info = df_statements_merged_info.rename(
        columns={"statement_id_x": "statement_id"}
    )

    return df_statements_merged_info


def aggregate_proofs_for_one_shard(
    working_dir: str, shard_id: str, statement_info: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates proofs, statements, and problems for one shard.

    Args:
        working_dir (str): The base directory path containing problems, statements, and proofs subfolders.
        shard_id (str): The ID of the shard to process (e.g., '000', '001').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - A DataFrame (problems) with aggregated proof data
            - A DataFrame (statements) with aggregated proof data
    """
    # Build filepaths
    shard_id_path = pathlib.Path(shard_id)
    proofs_path = (
        pathlib.Path(working_dir) / "verified_proofs" / f"{shard_id_path}.parquet"
    )
    statements_path = (
        pathlib.Path(working_dir) / "verified_statements" / f"{shard_id_path}.parquet"
    )
    problems_path = pathlib.Path(working_dir) / "problems" / f"{shard_id_path}.parquet"

    if (
        not problems_path.exists()
        or not statements_path.exists()
        or not proofs_path.exists()
    ):
        logger.warning(f"Missing data for shard_id {shard_id}")
        return None, None

    # Load data
    df_statements = pd.read_parquet(statements_path)
    df_proofs = pd.read_parquet(proofs_path)
    df_problems = pd.read_parquet(problems_path)

    # Group proofs
    grouped_df = (
        df_proofs.groupby("statement_id")
        .apply(_aggregate_proofs, include_groups=False)
        .reset_index()
    )

    # Ensure we have the same number of grouped results as statements
    assert len(grouped_df) == len(df_statements), (
        f"Grouped proofs ({len(grouped_df)}) != statement rows ({len(df_statements)}) "
        f"for shard_id {shard_id}"
    )

    # Merge grouped proofs with statements
    statements_join = df_statements.loc[
        :, ("statement_id", "natural_language", "is_negation")
    ]
    df_statements_merged = pd.merge(
        statements_join, grouped_df, on="statement_id", how="inner"
    )

    # Aggregate statements by uuid
    grouped_statements = (
        df_statements_merged.groupby("uuid")
        .apply(_aggregate_statements, include_groups=False)
        .reset_index()
    )

    # Merge aggregated statements into problems
    df_problems_merged = pd.merge(
        df_problems, grouped_statements, on="uuid", how="inner"
    )

    return df_problems_merged, df_statements_merged


def temp_function_aggregate_two_runs():
    d1 = datasets.load_dataset(
        "AI-MO/human-statements-moon-new-year-prover-v1", split="train"
    )
    d2 = datasets.load_dataset(
        "AI-MO/human-statements-moon-new-year-prover-v1-samall", split="train"
    )
    d1 = d1.to_pandas()
    d2 = d2.to_pandas()
    merged = pd.concat([d1, d2])
    merged = (
        merged.groupby("statement_id").apply(_aggregate_repeat).reset_index(drop=True)
    )
    merged = datasets.Dataset.from_pandas(merged)
    merged.push_to_hub(
        "AI-MO/human-statements-moon-new-year-prover-v1-merged", private=True
    )


def aggregate_proofs(
    working_dir: str,
    des_statements_dataset_id: str,
    des_problems_dataset_id: str = None,
    num_proc: int = 1,
    cache_dir=None,
    recompute_all: bool = False,
    aggregate_repeat: bool = False,
    statements_only: bool = False,
    info_dataset: str = None,
    info_columns: List[str] = None,
) -> None:
    """
    Aggregates proofs from multiple shards, possibly in parallel, creates
    Hugging Face datasets, and pushes them to the Hub.

    Args:
        working_dir (str): The base directory containing subfolders
                           (problems, verified_statements, verified_proofs).
        des_statements_dataset_id (str): The ID of the destination dataset for statements.
        des_problems_dataset_id (str, optional): The ID of the destination dataset for problems.
        num_proc (int, optional): Number of processes to use for parallelization.
                                  Defaults to 1 (no parallelism).
        cache_dir (str, optional): The directory to cache for hunggingface datasets.
        recompute_all (bool, optional): Whether to recompute all shards.
        aggregate_repeat (bool, optional): Whether to aggregate repeat statements, only use for statements only.
        statements_only (bool, optional): Whether to aggregate statements only (For proof generation pipeline).
        info_dataset (str, optional): The dataset to load statement info from (for statements only mode).
        info_columns (List[str], optional): The columns to select from the info dataset.
    """
    print(info_columns)
    # Create output directories
    aggregated_statements_dir = pathlib.Path(working_dir) / "aggregated_statements"
    aggregated_statements_dir.mkdir(parents=True, exist_ok=True)
    if not statements_only:
        aggregated_problems_dir = pathlib.Path(working_dir) / "aggregated_problems"
        aggregated_problems_dir.mkdir(parents=True, exist_ok=True)

    # Identify all shards
    if not statements_only:
        problems_dir = pathlib.Path(working_dir) / "problems"
        shard_ids = [p.stem for p in problems_dir.glob("*.parquet")]
    else:
        statements_dir = pathlib.Path(working_dir) / "statements"
        shard_ids = [p.stem for p in statements_dir.glob("*.parquet")]

    # Filter out shards that already have aggregated data
    if not recompute_all and not statements_only:
        shard_ids = list(
            filter(
                lambda id: not (aggregated_statements_dir / f"{id}.parquet").exists()
                or not (aggregated_problems_dir / f"{id}.parquet").exists(),
                shard_ids,
            )
        )

    if not recompute_all and statements_only:
        shard_ids = list(
            filter(
                lambda id: not (aggregated_statements_dir / f"{id}.parquet").exists(),
                shard_ids,
            )
        )

    print("Shards to process: ", len(shard_ids))

    # Load statement info dataset
    if info_dataset is not None:
        info_dataset = datasets.load_dataset(
            info_dataset, split="train", cache_dir=cache_dir
        )
        info_dataset = info_dataset.select_columns(info_columns)
        statement_info = info_dataset.to_pandas()
    else:
        statement_info = None

    # We will store future -> shard_id so we know which shard is done
    futures = {}
    aggregate_method = (
        statement_only_aggregate_proofs_for_one_shard
        if statements_only
        else aggregate_proofs_for_one_shard
    )

    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        # Submit jobs to the executor
        for shard_id in shard_ids:
            futures[shard_id] = executor.submit(
                aggregate_method, working_dir, shard_id, statement_info
            )

        # Collect results as they complete
        for shard_id in tqdm(shard_ids, desc="Aggregating shards"):
            if statements_only:
                df_statements = futures[shard_id].result()
                if df_statements is not None:
                    df_statements.to_parquet(
                        aggregated_statements_dir / f"{shard_id}.parquet"
                    )
            else:
                df_problems, df_statements = futures[shard_id].result()
                # Save aggregated problems and statements
                if df_problems is not None and df_statements is not None:
                    df_problems.to_parquet(
                        aggregated_problems_dir / f"{shard_id}.parquet"
                    )
                    df_statements.to_parquet(
                        aggregated_statements_dir / f"{shard_id}.parquet"
                    )

    # Create a Hugging Face dataset from aggregated statements and push
    auto_statements_ds = load_dataset(
        str(aggregated_statements_dir), split="train", cache_dir=cache_dir
    )
    if aggregate_repeat:
        assert (
            statements_only
        ), "aggregate_repeat should only be used for statements_only"
        auto_statements_ds = auto_statements_ds.to_pandas()
        auto_statements_ds = (
            auto_statements_ds.groupby("statement_id")
            .apply(_aggregate_repeat)
            .reset_index(drop=True)
        )
        auto_statements_ds = datasets.Dataset.from_pandas(auto_statements_ds)

    auto_statements_ds.push_to_hub(des_statements_dataset_id, private=True)

    if statements_only:
        return

    # Create a Hugging Face dataset from aggregated problems and push
    auto_problems_ds = load_dataset(str(aggregated_problems_dir), cache_dir=cache_dir)
    auto_problems_ds.push_to_hub(des_problems_dataset_id, private=True)

    # Compute fraction of problems with at least one correct proof
    n_correct_proofs = np.array(auto_problems_ds["train"]["n_correct_proofs"])
    rate = np.mean(n_correct_proofs > 0)
    logger.info(f"Rate of uuid with at least one correct proof: {rate:.2%}")


if __name__ == "__main__":
    """
    Usage:
        python -m autoformalizer.jobs.aggregate \
            --working_dir ../santa_prover/ \
            --des_statements_dataset_id AI-MO/auto-statements-santa-small-dryrun \
            --des_problems_dataset_id AI-MO/auto-problems-santa-small-dryrun \
            --num_proc 4 \
            --recompute_all True

        For moonshot:
        python -m autoformalizer.jobs.aggregate \
            --working_dir /mnt/moonfs/kimina-m2/jobs/santa_prover_v1/ \
            --des_statements_dataset_id AI-MO/auto-statememts-moon-santa-prover-v1 \
            --des_problems_dataset_id AI-MO/auto-problems-moon-santa-prover-v1 \
            --num_proc 128 \
            --cache_dir /mnt/moonfs/kimina-m2/.cache

        For statement only:
        python -m autoformalizer.jobs.aggregate \
            --working_dir /mnt/moonfs/kimina-m2/jobs/new_year_human_statments_prover_v7 \
            --des_statements_dataset_id AI-MO/human-statements-moon-new-year-prover-v1 \
            --num_proc 128 \
            --cache_dir /mnt/moonfs/kimina-m2/.cache \
            --statements_only True \
            --aggregate_repeat True \
            --info_dataset AI-MO/human-statements-dataset-v1-20250103 \
            --info_columns "statement_id,ground_truth,natural_language,tags"

        For statement only:
        python -m autoformalizer.jobs.aggregate \
            --working_dir /mnt/moonfs/kimina-m2/jobs/new_year_human_statments_prover_mathlib_v6 \
            --des_statements_dataset_id AI-MO/human-statements-moon-new-year-prover-mathlib-v1 \
            --num_proc 128 \
            --cache_dir /mnt/moonfs/kimina-m2/.cache \
            --statements_only True \
            --aggregate_repeat True \
            --info_dataset AI-MO/human-statements-dataset-v1-mathlib-20250106 \
            --info_columns "statement_id,ground_truth,natural_language,tags"

        python -m autoformalizer.jobs.aggregate \
            --working_dir /mnt/moonfs/kimina-m2/jobs/flying_ant_prover_v1 \
            --des_statements_dataset_id AI-MO/auto-statements-moon-flying-ant-prover-v1 \
            --num_proc 32 \
            --cache_dir /mnt/moonfs/kimina-m2/.cache \
            --statements_only True \
            --aggregate_repeat True \
            --info_dataset AI-MO/auto-statements-moon-flying-ant-v1-20250110 \
            --info_columns "statement_id,natural_language,is_negation"

        python -m autoformalizer.jobs.aggregate \
            --working_dir /mnt/moonfs/kimina-m2/jobs/jeasper_prover_v1 \
            --des_statements_dataset_id AI-MO/human-statements-moon-jeasper-prover-stage1-v1 \
            --num_proc 64 \
            --cache_dir /mnt/moonfs/kimina-m2/.cache \
            --statements_only True \
            --aggregate_repeat True \
            --info_dataset AI-MO/human-statements-dataset-v2-20250118 \
            --info_columns "statement_id,natural_language,is_negation"

        python -m autoformalizer.jobs.aggregate \
            --working_dir /mnt/moonfs/kimina-m2/jobs/jeasper_stage2_prover_v1 \
            --des_statements_dataset_id AI-MO/human-statements-moon-jeasper-prover-stage2-v1 \
            --num_proc 64 \
            --cache_dir /mnt/moonfs/kimina-m2/.cache \
            --statements_only True \
            --aggregate_repeat True \
            --info_dataset AI-MO/human-statements-dataset-v3-20250121 \
            --info_columns "statement_id,natural_language,is_negation"

    """
    fire.Fire(aggregate_proofs)
    # temp_function_aggregate_two_runs()
