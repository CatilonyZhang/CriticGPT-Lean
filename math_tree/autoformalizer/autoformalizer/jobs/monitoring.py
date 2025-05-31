import datetime
import pathlib
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

import fire
import pandas as pd
from loguru import logger
from tqdm import tqdm


def estimate_time_to_process_shards(
    working_dir: pathlib.Path, finished_shards: List[str], all_shard_ids: List[str]
) -> None:
    """
    Estimate the remaining time to process all shards based on the average time per processed shard.

    Args:
        working_dir (pathlib.Path): The base directory path containing the pipeline log.
        finished_shards (List[str]): List of shard IDs that have been processed.
        all_shard_ids (List[str]): List of all shard IDs to be processed.
    """
    log_file = working_dir / "pipeline.log"

    # Read the log file and extract the start time (the first relevant log entry)
    start_time = None
    try:
        with open(log_file, "r") as log:
            for line in log:
                # Find the first log entry indicating the start of processing
                if "Creating working directories" in line:
                    timestamp_str = line.split(" | ")[
                        0
                    ]  # Extract timestamp part of the line
                    start_time = datetime.datetime.strptime(
                        timestamp_str, "%Y-%m-%d %H:%M:%S.%f"
                    )
                    break
    except FileNotFoundError:
        logger.error(f"Log file {log_file} does not exist.")
        return

    if start_time is None:
        logger.error("Could not find start time in log file.")
        return

    # Calculate the elapsed time since the start
    elapsed_time = datetime.datetime.now() - start_time
    num_shards_processed = len(finished_shards)

    if num_shards_processed == 0:
        logger.warning("No finished shards found. Cannot estimate remaining time.")
        return

    # Calculate average time per shard (in seconds)
    avg_time_per_shard = elapsed_time.total_seconds() / num_shards_processed

    # Estimate time remaining (in seconds) to process all remaining shards
    remaining_shards = len(all_shard_ids) - num_shards_processed
    estimated_remaining_time = avg_time_per_shard * remaining_shards

    # Convert remaining time to a human-readable format (hours, minutes, seconds)
    remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))

    logger.info(f"Elapsed time: {str(elapsed_time)}")
    logger.info(f"Average time per shard: {avg_time_per_shard:.2f} seconds")
    logger.info(f"Estimated time remaining: {remaining_time_str}")


def verify_negation(statements: pd.DataFrame) -> None:
    """
    Verify if negation statements compile and display grouped counts.

    Args:
        statements (pd.DataFrame): DataFrame containing statements with their validity.
    """
    # Separate original and negated statements
    original_statements = statements.loc[~statements["is_negation"]].copy()
    original_statements = original_statements[
        [
            "uuid",
            "statement_id",
            "natural_language",
            "formal_statement",
            "is_valid_with_sorry",
        ]
    ].rename(columns={"is_valid_with_sorry": "is_original_valid"})

    negative_statements = statements.loc[statements["is_negation"]].copy()
    negative_statements = negative_statements[
        ["statement_id", "is_valid_with_sorry"]
    ].rename(columns={"is_valid_with_sorry": "is_negation_valid"})
    negative_statements["statement_id_clean"] = negative_statements[
        "statement_id"
    ].str.replace("neg_", "", regex=False)

    # Merge negation validity back to original statements
    original_statements = original_statements.merge(
        negative_statements[["statement_id_clean", "is_negation_valid"]],
        left_on="statement_id",
        right_on="statement_id_clean",
        how="left",
    ).drop(columns=["statement_id_clean"])

    # Fill NaN values where there are no corresponding negations
    original_statements["is_negation_valid"] = original_statements[
        "is_negation_valid"
    ].fillna(False)

    # Display grouped counts
    grouped_counts = (
        original_statements.groupby(["is_original_valid", "is_negation_valid"])
        .size()
        .reset_index(name="count")
    )
    logger.info(f"Grouped Counts:\n{grouped_counts}")


def compute_metrics(
    shard_id: str, verified_proofs_dir: pathlib.Path
) -> Dict[str, Dict[str, Any]]:
    """
    Compute various metrics for a given shard.

    Args:
        shard_id (str): The ID of the shard to process.
        verified_proofs_dir (pathlib.Path): Directory containing verified proofs.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing computed metrics.
    """
    # Load verified proofs
    verified_proofs_path = verified_proofs_dir / f"{shard_id}.parquet"
    verified_proofs = pd.read_parquet(verified_proofs_path)
    df = verified_proofs.copy()
    df["is_negation"] = df["statement_id"].str.startswith("neg_")
    df["statement_id_clean"] = df["statement_id"].str.replace("neg_", "", regex=False)

    # --- 1. STATEMENT-LEVEL METRICS ---
    statement_metrics = (
        df.groupby(["statement_id_clean", "is_negation"])["is_valid_no_sorry"]
        .sum()
        .unstack(fill_value=0)
    )
    statement_metrics.columns = ["num_valid_proofs", "num_negation_valid_proofs"]
    statement_metrics = statement_metrics.reset_index()

    n_valid = (statement_metrics["num_valid_proofs"] > 0).sum()
    n_negation_valid = (statement_metrics["num_negation_valid_proofs"] > 0).sum()

    # --- 2. UUID-LEVEL METRICS ---
    uuid_agg = df.loc[~df.is_negation].groupby("uuid")["is_valid_no_sorry"].sum()
    uuid_metrics = (uuid_agg > 0).astype(int).rename("is_uuid_proved").reset_index()

    n_proved_uuids = uuid_metrics["is_uuid_proved"].sum()
    n_total_uuids = len(uuid_metrics)

    return {
        "uuid_info": {
            "valid": int(n_proved_uuids),
            "total": int(n_total_uuids),
        },
        "statement_info": {
            "valid": int(n_valid),
            "total": int(statement_metrics.shape[0]),
        },
        "negation_info": {
            "valid": int(n_negation_valid),
            "total": int(statement_metrics.shape[0]),
        },
        "proof_info": {
            "total": int(df.shape[0]),
            "valid": int(df["is_valid_no_sorry"].sum()),
        },
    }


def analyse_jobs(working_dir: str, num_proc: int = 1) -> None:
    """
    Analyze job progress by computing metrics for finished shards in parallel.

    Args:
        working_dir (str): The base directory containing various subdirectories.
        num_proc (int, optional): Number of parallel processes to use. Defaults to 1.
    """
    working_dir = pathlib.Path(working_dir)
    problems_dir = working_dir / "problems"
    statements_dir = working_dir / "statements"
    verified_statements_dir = working_dir / "verified_statements"
    proofs_dir = working_dir / "proofs"
    verified_proofs_dir = working_dir / "verified_proofs"

    step_outputs = [
        problems_dir,
        statements_dir,
        verified_statements_dir,
        proofs_dir,
        verified_proofs_dir,
    ]
    all_shard_ids = [x.stem for x in problems_dir.glob("*.parquet")]

    # Log the processing status of each step
    for step in step_outputs[1:]:
        _n_shards = len(list(step.glob("*.parquet")))
        logger.info(
            f"Processing at {step.name}, {_n_shards}/{len(all_shard_ids)} have finished"
        )

    # Identify finished shards
    finished_shards = [x.stem for x in verified_proofs_dir.glob("*.parquet")]

    # Quit if no shards have been finished
    if not finished_shards:
        logger.info("No finished shards, quitting")
        return

    # Verify negation for a random finished shard
    shard_id_sample = random.choice(finished_shards)
    logger.info(f"Analyzing shard: {shard_id_sample}")

    # Load statements for verification
    statements_path = statements_dir / f"{shard_id_sample}.parquet"
    statements = pd.read_parquet(statements_path)
    logger.debug("Loaded statements")
    logger.debug(f"Statements columns: {statements.columns.tolist()}")

    verify_negation(statements)

    # Load verified statements and proofs
    verified_statements_path = verified_statements_dir / f"{shard_id_sample}.parquet"
    proofs_path = proofs_dir / f"{shard_id_sample}.parquet"
    verified_proofs_path = verified_proofs_dir / f"{shard_id_sample}.parquet"

    verified_statements = pd.read_parquet(verified_statements_path)
    logger.debug("Loaded verified_statements")
    logger.debug(f"Verified Statements columns: {verified_statements.columns.tolist()}")

    proofs = pd.read_parquet(proofs_path)
    logger.debug(f"Loaded proofs, {proofs.shape[0]} rows")
    logger.debug(f"Proofs columns: {proofs.columns.tolist()}")

    verified_proofs = pd.read_parquet(verified_proofs_path)
    logger.debug(f"Loaded proofs, {verified_proofs.shape[0]} rows")
    logger.debug(f"Verified proofs columns: {verified_proofs.columns.tolist()}")
    one_valid_proof = (
        verified_proofs.loc[verified_proofs["is_valid_no_sorry"]].sample(1).iloc[0]
    )
    logger.info(f"Formal statement:\n{one_valid_proof['formal_statement']}")
    logger.info(f"Valid proof:\n{one_valid_proof['proof']}")

    # Display example of a valid negative proof
    negative_proofs = verified_proofs.loc[
        verified_proofs["statement_id"].str.startswith("neg_")
    ]
    if not negative_proofs.empty:
        one_valid_negative_proof = (
            negative_proofs.loc[negative_proofs["is_valid_no_sorry"]].sample(1).iloc[0]
        )
        n_statement_id = one_valid_negative_proof["statement_id"]
        o_statement_id = n_statement_id.replace("neg_", "")
        o_statement = statements.loc[statements["statement_id"] == o_statement_id].iloc[
            0
        ]
        logger.info(
            f"uuid: {one_valid_negative_proof['uuid']}, statement_id: {n_statement_id}"
        )
        logger.info(f"Original Natural statement:\n{o_statement['natural_language']}")
        logger.info(f"Original Formal statement:\n{o_statement['formal_statement']}")
        logger.info(
            f"Negated Formal statement:\n{one_valid_negative_proof['formal_statement']}"
        )
        logger.info(f"Negated Valid proof:\n{one_valid_negative_proof['proof']}")
    else:
        logger.info("No valid negative proofs found in the sampled shard.")

    # Show a small part of finished shards
    logger.info(f"Finished shards (last 5): {finished_shards[-5:]}")

    # Initialize metrics accumulator
    all_metrics = {
        "uuid_info": {"valid": 0, "total": 0},
        "statement_info": {"valid": 0, "total": 0},
        "negation_info": {"valid": 0, "total": 0},
        "proof_info": {"total": 0, "valid": 0},
    }

    # Parallel computation of metrics
    logger.info(
        f"Computing metrics for {len(finished_shards)} finished shards using {num_proc} processes."
    )
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        # Submit all compute_metrics tasks
        future_to_shard = {
            executor.submit(compute_metrics, shard_id, verified_proofs_dir): shard_id
            for shard_id in finished_shards
        }

        for future in tqdm(
            as_completed(future_to_shard),
            total=len(finished_shards),
            desc="Computing metrics",
        ):
            shard_id = future_to_shard[future]
            try:
                metrics = future.result()
                # Aggregate metrics
                for category, values in metrics.items():
                    for key, value in values.items():
                        all_metrics[category][key] += value
            except Exception as e:
                logger.error(f"Error processing shard {shard_id}: {e}")

    logger.info("Aggregated metrics from all shards")
    all_metrics_df = pd.DataFrame(all_metrics).T
    all_metrics_df["valid_ratio"] = all_metrics_df["valid"] / all_metrics_df["total"]
    logger.info(f"All Metrics:\n{all_metrics_df}")

    # If processing is not finished, estimate remaining time
    if len(finished_shards) < len(all_shard_ids):
        estimate_time_to_process_shards(working_dir, finished_shards, all_shard_ids)


if __name__ == "__main__":
    """
    Usage:
        python -m autoformalizer.jobs.monitoring \
            --working_dir ../santa_prover/ \
            --num_proc 4
    """
    fire.Fire(analyse_jobs)
