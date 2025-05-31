#!/usr/bin/env python3
import concurrent.futures
import glob
import json
import multiprocessing as mp
import os
import pathlib
import random
import sys
from time import time
from typing import Any, Dict, List, Tuple

import fire
import pandas as pd
import yaml
from datasets import concatenate_datasets, load_dataset
from loguru import logger
from tqdm import tqdm

from autoformalizer.jobs.input_sharder import InputSharder
from autoformalizer.jobs.processor import Negator
from autoformalizer.jobs.prover import ProofGenerator, ProofVerifier, write_proof_record


class Config:
    def __init__(self, config_path: str):
        """
        Initialize and load the configuration from a YAML file.
        """
        self.config_path = config_path
        self.config = {}
        self.load_config()

    def load_config(self):
        """Load the YAML file and replace placeholders."""
        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_path} not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

        # Get the user's home directory
        home_dir = os.path.expanduser("~")

        # Replace placeholders in working_dir
        job_id = self.config.get("job_id", "")
        self.config["working_dir"] = pathlib.Path(
            self.config["working_dir"].format(home=home_dir, job_id=job_id)
        )

        # Replace placeholders in derived directories
        self.config["input_dir"] = pathlib.Path(
            self.config["input_dir"].format(working_dir=self.config["working_dir"])
        )
        self.config["statements_dir"] = pathlib.Path(
            self.config["statements_dir"].format(working_dir=self.config["working_dir"])
        )
        self.config["proofs_dir"] = pathlib.Path(
            self.config["proofs_dir"].format(working_dir=self.config["working_dir"])
        )
        self.config["verified_proofs_dir"] = pathlib.Path(
            self.config["verified_proofs_dir"].format(
                working_dir=self.config["working_dir"]
            )
        )
        self.config["proof_record_dir"] = pathlib.Path(
            self.config["proof_record_dir"].format(
                working_dir=self.config["working_dir"]
            )
        )
        # create proof record file if not exist
        self.config["proof_record_lock"] = mp.Manager().Lock()

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        return self.config[key]

    def __repr__(self):
        return f"Config({self.config})"


def setup_logging(log_file: pathlib.Path):
    logger.remove()  # Remove default stderr logger
    logger.add(sys.stdout, level="INFO")  # Add stdout logger
    logger.add(
        log_file, rotation="1 week", retention="4 weeks", level="DEBUG"
    )  # Add file logger


def setup_directories(config: Config):
    logger.info(f"Creating working directories at {config['working_dir']}")
    config["working_dir"].mkdir(parents=True, exist_ok=True)
    for dir_path in [
        config["statements_dir"],
        config["proofs_dir"],
        config["verified_proofs_dir"],
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
    logger.debug("All directories created successfully.")


def shard_input_data(
    datasets_list: List[Dict[str, Any]], repeat: int, config: Config
) -> List[Tuple[str, pathlib.Path]]:
    """
    Concatenate all datasets and shard the combined dataset.

    Args:
        datasets_dict (Dict[str, Dict[str, Any]]): Dictionary of datasets to load and concatenate.
            Example structure:
                {
                    "dataset1": {
                        "path": "path_or_huggingface_dataset_name_1",
                        "split": "train",
                        "select_range": 120
                    },
                    "dataset2": {
                        "path": "path_or_huggingface_dataset_name_2",
                        "split": "train",
                    },
                    ...
                }
        repeat (int): Number of times to repeat the sharding process.
        config (Config): Loaded configuration object.

    Returns:
        List[Tuple[str, pathlib.Path]]: A list of tuples containing shard_id and shard_path.
    """
    logger.info("Starting dataset concatenation and sharding process...")

    datasets = []
    for ds_info in datasets_list:
        ds_name = ds_info["name"]
        logger.info(f"Loading dataset: {ds_name}")
        splits = ds_info.get("splits", "train")
        # Select a range if specified (for testing purposes)
        select_range = ds_info.get("select_range")

        columns_to_read = [
            "uuid",
            "statement_id",
            "formal_statement",
        ]
        for split in splits.split(","):
            # Load the dataset
            loaded_ds = load_dataset(
                ds_name,
                split=split,
                cache_dir=config.get("huggingface_cache_dir", None),
            )

            if select_range:
                loaded_ds = loaded_ds.select(range(min(select_range, len(loaded_ds))))
                logger.info(
                    f"Selected first {min(select_range, len(loaded_ds))} samples from {ds_name}"
                )

            # Select specific columns
            loaded_ds = loaded_ds.select_columns(columns_to_read)
            datasets.append(loaded_ds)
            logger.info(f"Dataset {ds_name} loaded with {len(loaded_ds)} samples.")

    # Concatenate all datasets
    concatenated_ds = concatenate_datasets(datasets)
    # assert all statement id are unique
    assert len(concatenated_ds["statement_id"]) == len(
        set(concatenated_ds["statement_id"])
    )
    logger.info(f"All datasets concatenated. Total samples: {len(concatenated_ds)}")

    # Shard the concatenated dataset
    sharder = InputSharder(
        dataset_path=concatenated_ds,
        output_path=config["input_dir"],
        shard_size=config.get("shard_size", 10_000),
        columns_to_read=[
            "uuid",
            "statement_id",
            "formal_statement",
        ],
        repeat=repeat,
    )

    logger.info(f"Total shards created: {len(sharder.shards)}")
    return sharder.shards


def process_shard(shard: Tuple[str, pathlib.Path], config: Config) -> bool:
    """
    Process a single shard: Autoformalization, Verification, Proof Generation, and Proof Verification.

    Args:
        shard (Tuple[str, pathlib.Path]): Tuple containing shard_id and shard_path.
        config (Config): Configuration object.

    Returns:
        bool: True if processing is successful, False otherwise.
    """
    shard_id, shard_path = shard
    logger.info(f"Processing shard {shard_id} at {shard_path}")

    try:
        # Step 0: Negation
        negator = Negator()
        status, _ = negator(
            shard_id=shard_id,
            input_dir=config["input_dir"],
            output_dir=config["statements_dir"],
            add_negation=config["add_negation"],
        )

        if not status:
            logger.warning(f"Negation failed for shard {shard_id}")
            return False

        # Step 1: Proof Generation
        prover_params = config["prover_params"]
        prover = ProofGenerator(
            openai_api_bases=prover_params["url"],
            models=prover_params["model_path"],
            openai_api_key="EMPTY",
            enable_proof_filtering=config["enable_proof_filtering"],
            proof_record_dir=config["proof_record_dir"],
            proof_record_lock=config["proof_record_lock"],
            filter_threshold=config["filter_threshold"],
            is_mathlib="mathlib" in config["job_id"],
        )

        prompt_template = (
            "Complete the following Lean 4 code:\n\n```lean4\n{formal_statement}"
        )
        cot_prompt_template = (
            "Complete the following Lean 4 code with explanatory "
            + "comments preceding each line of code::\n\n```lean4\n{formal_statement}"
        )

        sampling_params = {
            "max_tokens": prover_params.get("max_tokens", 2048),
            "temperature": 1.0,
            "n": prover_params.get("num_samples_per_statement", 1),
            "num_threads": prover_params.get("num_threads", 1),
        }
        status, _ = prover(
            shard_id=shard_id,
            input_dir=config["statements_dir"],
            output_dir=config["proofs_dir"],
            prompt_template=prompt_template,
            cot_prompt_template=cot_prompt_template,
            **sampling_params,
        )
        if not status:
            logger.warning(f"Proof generation failed for shard {shard_id}")
            return False

        # Step 4: Proof Verification
        verifier_params = config["verifier_params"]

        proof_verifier = ProofVerifier(
            client_url=verifier_params["url"],
            client_api_key=None,
            timeout=verifier_params["timeout"],
            num_threads=verifier_params["num_threads"],
            batch_size=verifier_params["batch_size"],
            enable_proof_filtering=config["enable_proof_filtering"],
            proof_record_dir=config["proof_record_dir"],
            proof_record_lock=config["proof_record_lock"],
        )

        status, _ = proof_verifier(
            shard_id=shard_id,
            input_dir=config["proofs_dir"],
            output_dir=config["verified_proofs_dir"],
        )
        if not status:
            logger.warning(f"Proof verification failed for shard {shard_id}")
            return False

        logger.info(f"Shard {shard_id} processed successfully.")
        return True

    except Exception as e:
        logger.error(f"Exception while processing shard {shard_id}: {e}")
        return False


def compute_statistics(config: Config):
    """
    Compute and log various statistics based on the verified proofs.

    Args:
        config (Config): Configuration object.
    """
    try:
        logger.info("Computing statistics on verified proofs...")

        # Load verified proofs dataset
        verified_ds = load_dataset(str(config["verified_proofs_dir"]), split="train")
        verified_ds = verified_ds.filter(
            lambda x: not x["statement_id"].startswith("neg_")
        )
        logger.info(
            f"Verified dataset (non-negative statements) size: {len(verified_ds)}"
        )

        # Calculate Valid Rate
        if len(verified_ds) == 0:
            logger.warning("No verified non-negative statements found.")
            valid_rate = 0.0
        else:
            valid_rate = sum(verified_ds["is_valid_no_sorry"]) / len(verified_ds)
        logger.info(f"Valid rate: {valid_rate:.4f}")

        # Prove rate by uuid
        verified_df = verified_ds.select_columns(
            ["uuid", "is_valid_no_sorry"]
        ).to_pandas()
        uuid_proofs = verified_df.groupby("uuid").sum()
        prove_rate = (
            len(uuid_proofs[uuid_proofs["is_valid_no_sorry"] > 0]) / len(uuid_proofs)
            if len(uuid_proofs) > 0
            else 0.0
        )
        logger.info(f"Prove rate by uuid: {prove_rate:.4f}")

        # Negative statement prove rate
        neg_verified_ds = load_dataset(
            str(config["verified_proofs_dir"]), split="train"
        )
        neg_verified_ds = neg_verified_ds.filter(
            lambda x: x["statement_id"].startswith("neg_")
        )
        logger.info(
            f"Verified dataset (negative statements) size: {len(neg_verified_ds)}"
        )

        if len(neg_verified_ds) == 0:
            logger.warning("No verified negative statements found.")
            neg_prove_rate = 0.0
        else:
            neg_verified_df = neg_verified_ds.select_columns(
                ["statement_id", "is_valid_no_sorry"]
            ).to_pandas()
            neg_proofs = neg_verified_df.groupby("statement_id").sum()
            neg_prove_rate = (
                sum(neg_proofs["is_valid_no_sorry"]) / len(neg_proofs)
                if len(neg_proofs) > 0
                else 0.0
            )
        logger.info(f"Negative statement prove rate: {neg_prove_rate:.4f}")

    except Exception as e:
        logger.error(f"Error during statistics computation: {e}")


def run(config_path: str):
    """
    Main function to execute the proof generation pipeline.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # Log the total processing time
    start_time = time()

    # Initialize configuration
    config = Config(config_path=config_path)

    # Setup logging
    log_file = config["working_dir"] / "pipeline.log"
    setup_logging(log_file)

    # Setup directories
    setup_directories(config)

    # Step 1: Shard the input data
    datasets_list = config.get("datasets", {})
    repeat = config.get("repeat", 1)
    shards = shard_input_data(datasets_list, repeat, config)

    # Define the number of workers based on your system's resources
    max_workers = config.get("max_workers", 1)  # Example: Adjust as needed

    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all shard processing tasks
        future_to_shard = {
            executor.submit(process_shard, shard, config): shard for shard in shards
        }

        # Collect results
        for future in concurrent.futures.as_completed(future_to_shard):
            shard = future_to_shard[future]
            shard_id, _ = shard
            try:
                result = future.result()
                if result:
                    logger.info(f"Shard {shard_id} completed successfully.")
                else:
                    logger.warning(f"Shard {shard_id} failed during processing.")
            except Exception as exc:
                logger.error(f"Shard {shard_id} generated an exception: {exc}")

    # After all shards are processed, perform final aggregation or cleanup
    logger.info("All shards have been processed.")

    # Step 2: Compute and log statistics
    compute_statistics(config)

    # Log the total processing time
    end_time = time()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")


def find_proved_statement_shard(
    shard_path: pathlib.Path, remove_statements: set = None
) -> List[str]:
    """
    Return all the valid statement id
    """
    df = pd.read_parquet(shard_path)
    if remove_statements is not None:
        df = df[~df["statement_id"].isin(remove_statements)]
    valid_statements = df[df["is_valid_no_sorry"] > 0]["statement_id"].unique().tolist()
    all_statements = df["statement_id"].unique().tolist()
    df_pos = df[~df["statement_id"].str.startswith("neg_")]
    valid_uuids = df_pos[df_pos["is_valid_no_sorry"] > 0]["uuid"].unique().tolist()
    all_uuids = df["uuid"].unique().tolist()
    metrics = {
        "valid_uuids": valid_uuids,
        "all_uuids": all_uuids,
        "valid_statements": valid_statements,
        "all_statements": all_statements,
        "n_proofs": len(df),
        "n_valid_proofs": len(df[df["is_valid_no_sorry"] > 0]),
    }
    del df
    return metrics


def monitor(config_path: str, removed_statements_path: str = None, num_proc: int = 4):
    """
    Monitor the proof generation pipeline.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # Initialize configuration
    config = Config(config_path=config_path)

    # Setup logging
    log_file = config["working_dir"] / "pipeline.log"
    setup_logging(log_file)

    # check the number of shards at each stage. Take input shards as the base
    input_shards = list(config["input_dir"].glob("*.parquet"))
    logger.info(f"Number of input shards: {len(input_shards)}")
    steps = ["statements_dir", "proofs_dir", "verified_proofs_dir"]
    for step in steps:
        _n_shards = len(list(config[step].glob("*.parquet")))
        logger.info(
            f"Processing at {step}, {_n_shards}/{len(input_shards)} have finished"
        )

    # Compute metrics
    all_metrics = {
        "uuid_info": {"valid": 0, "total": 0},
        "statement_info": {"valid": 0, "total": 0},
        "negation_info": {"valid": 0, "total": 0},
        "proof_info": {"total": 0, "valid": 0},
    }

    verified_proofs_dir = config["verified_proofs_dir"]
    finished_shards = [x.stem for x in verified_proofs_dir.glob("*.parquet")]
    if not finished_shards:
        logger.warning("No finished shards found.")
        return

    logger.info(
        f"Computing metrics for {len(finished_shards)} finished shards using {num_proc} processes."
    )

    all_shard_paths = [
        verified_proofs_dir / f"{shard}.parquet" for shard in finished_shards
    ]

    # pick random one shard
    one_shard = pd.read_parquet(random.choice(all_shard_paths))
    logger.info(one_shard.columns)
    # find all valid statement_id
    valid_statements = (
        one_shard[one_shard["is_valid_no_sorry"] > 0]["statement_id"].unique().tolist()
    )
    if valid_statements:
        one_valid_statement = random.choice(valid_statements)
        one_proof = one_shard[
            (one_shard["statement_id"] == one_valid_statement)
            & (one_shard["is_valid_no_sorry"] > 0)
        ].iloc[0]["proof"]
        logger.info(f"Valid proof for statement_id {one_valid_statement}: {one_proof}")

    all_valid_statements = set()
    all_statements = set()
    all_uuids = set()
    all_valid_uuids = set()
    n_proofs = 0
    n_valid_proofs = 0

    if removed_statements_path and pathlib.Path(removed_statements_path).exists():
        with open(removed_statements_path, "r") as f:
            removed_statements = json.load(f)
            removed_statements = set(removed_statements)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
        # Submit all shard processing tasks
        future_to_shard = {
            executor.submit(
                find_proved_statement_shard, shard_path, removed_statements
            ): shard_path
            for shard_path in all_shard_paths
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_shard), total=len(future_to_shard)
        ):
            shard_metrics = future.result()
            all_valid_statements.update(shard_metrics["valid_statements"])
            all_statements.update(shard_metrics["all_statements"])
            all_uuids.update(shard_metrics["all_uuids"])
            all_valid_uuids.update(shard_metrics["valid_uuids"])
            n_proofs += shard_metrics["n_proofs"]
            n_valid_proofs += shard_metrics["n_valid_proofs"]

    all_metrics["proof_info"]["total"] = n_proofs
    all_metrics["proof_info"]["valid"] = n_valid_proofs

    # Compute metrics for uuids
    all_metrics["uuid_info"]["total"] = len(all_uuids)
    all_metrics["uuid_info"]["valid"] = len(all_valid_uuids)

    # Compute metrics for positive statements
    all_metrics["statement_info"]["total"] = len(
        [x for x in all_statements if not x.startswith("neg_")]
    )
    all_metrics["statement_info"]["valid"] = len(
        [x for x in all_valid_statements if not x.startswith("neg_")]
    )

    # same thing for negations
    all_metrics["negation_info"]["total"] = len(
        [x for x in all_statements if x.startswith("neg_")]
    )
    all_metrics["negation_info"]["valid"] = len(
        [x for x in all_valid_statements if x.startswith("neg_")]
    )

    # logger.info("Aggregated metrics from all shards")
    all_metrics_df = pd.DataFrame(all_metrics).T
    all_metrics_df["valid_ratio"] = all_metrics_df["valid"] / all_metrics_df["total"]
    logger.info(f"All Metrics:\n{all_metrics_df}")


def create_proof_record(config_path, proof_record_path, num_proc=4):
    """
    Process parquet files in the verified_proof_path and create proof records
    in proof_record_path using multiprocessing and show progress with tqdm.
    """

    def process_file(file, proof_record_path, lock):
        """
        This function processes a single parquet file, reads the data,
        and writes the proof record to the jsonl file.
        """
        print(f"Processing file: {file}")
        df = pd.read_parquet(file)
        write_proof_record(df, proof_record_path, lock)

    # Initialize configuration
    config = Config(config_path=config_path)

    # Get all the parquet files in the verified proof path
    files = glob.glob(config["verified_proofs_dir"] + "/*.parquet")

    # Create a lock using multiprocessing Manager
    lock = mp.Manager().Lock()

    # Use ProcessPoolExecutor to process files concurrently
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
        # Wrap the files iterable with tqdm to show progress
        futures = [
            executor.submit(process_file, file, proof_record_path, lock)
            for file in files
        ]

        # Use tqdm to show progress bar as tasks are completed
        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Creating proof records",
        ):
            pass

    print(f"All files processed and proof records created at {proof_record_path}.")


if __name__ == "__main__":
    """
    Usage:
    # run
    python -m autoformalizer.jobs.proof_generation.main run \
        autoformalizer/jobs/proof_generation/annotated_dataset_template.yaml

    python -m autoformalizer.jobs.proof_generation.main run \
        autoformalizer/jobs/proof_generation/annotated_dataset_moonshot.yaml

    python -m autoformalizer.jobs.proof_generation.main run \
        autoformalizer/jobs/proof_generation/annotated_dataset_moonshot_mathlib.yaml

    python -m autoformalizer.jobs.proof_generation.main run \
        autoformalizer/jobs/proof_generation/annotated_dataset_moonshot_flying_ant.yaml

    python -m autoformalizer.jobs.proof_generation.main run \
        autoformalizer/jobs/proof_generation/annotated_dataset_moonshot_jeasper.yaml

    python -m autoformalizer.jobs.proof_generation.main run \
        autoformalizer/jobs/proof_generation/annotated_dataset_moonshot_jeasper_deepseek_baseline.yaml

    python -m autoformalizer.jobs.proof_generation.main monitor \
        autoformalizer/jobs/proof_generation/cot_proofs_diff.yaml \
        --num_proc 4

    Example log output:
    2025-01:compute_statistics:278 - Verified dataset (non-negative statements)size: 1120
    2025-01:compute_statistics:300 - Prove rate by uuid: 0.2429
    2025-01:compute_statistics:288 - Valid rate: 0.1018
    2025-01:compute_statistics:309 - Verified dataset (negative statements) size: 1088
    2025-01:compute_statistics:326 - Negative statement prove rate: 0.0000
    2025-01:main:387 - Total processing time: 74.54 seconds

    # monitor
    python -m autoformalizer.jobs.proof_generation.main monitor \
        autoformalizer/jobs/proof_generation/annotated_dataset_template.yaml \
        --num_proc 4
    python -m autoformalizer.jobs.proof_generation.main monitor \
        autoformalizer/jobs/proof_generation/annotated_dataset_moonshot.yaml \
        --num_proc 128
    python -m autoformalizer.jobs.proof_generation.main monitor \
        autoformalizer/jobs/proof_generation/annotated_dataset_moonshot_mathlib.yaml \
        --num_proc 128

    python -m autoformalizer.jobs.proof_generation.main monitor \
        autoformalizer/jobs/proof_generation/annotated_dataset_moonshot_flying_ant.yaml \
        --num_proc 128

    python -m autoformalizer.jobs.proof_generation.main monitor \
        autoformalizer/jobs/proof_generation/annotated_dataset_moonshot_jeasper.yaml \
        --num_proc 64

    python -m autoformalizer.jobs.proof_generation.main monitor \
        autoformalizer/jobs/proof_generation/annotated_dataset_moonshot_jeasper_deepseek_baseline.yaml \
        --num_proc 32

    python -m autoformalizer.jobs.proof_generation.main monitor \
        autoformalizer/jobs/proof_generation/annotated_dataset_moonshot_jeasper_stage2.yaml \
        --num_proc 32



    # create_proof_record
    The proof record is used to filter problems that have already been solved multiple times
    (e.g., when the count exceeds the filter_threshold in the config).
    It also filters out (non-)negated problems that have been solved.

    You only need to run this step if you wish to inherit the proof record from a previous run,
    and if the previous run does not have the proof record at the specified proof_record_dir file.

    Place the generated proof_records.jsonl file into the working_dir of the new run before starting the job.


    python -m autoformalizer.jobs.proof_generation.main create_proof_record \
        autoformalizer/jobs/proof_generation/annotated_dataset_template.yaml \
        --proof_record_path path/to/proof_records.jsonl \
        --num_proc 4

    python -m autoformalizer.jobs.proof_generation.main create_proof_record \
        autoformalizer/jobs/proof_generation/annotated_dataset_template.yaml \
        --proof_record_path path/to/proof_records.jsonl \
        --num_proc 4
    """
    fire.Fire(
        {
            "run": run,
            "monitor": monitor,
            "create_proof_record": create_proof_record,
        }
    )
