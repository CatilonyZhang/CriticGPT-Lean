import multiprocessing
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import fire
import numpy as np
import pandas as pd
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from autoformalizer.data_utils import process_proof, process_statement
from autoformalizer.sft_data.util import _proof_using_0, find_different_proof


class AutoSftDatasetPipeline:

    def __init__(
        self,
        source,
        beta: float = 1,
        informal_prefix: bool = False,
        push_name: str = None,
    ):
        self.SOURCE = source
        self.working_dir = pathlib.Path(os.environ["SFT_DATA_DIR"])
        self.input_dir = self.working_dir / self.SOURCE / "source"
        self.output_dir = self.working_dir / self.SOURCE / "processed"

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Source directory {self.input_dir} not found.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.beta = beta
        self.push_name = push_name
        self.informal_prefix = informal_prefix

    def process_shard(self, shard_id: int):
        """
        Process a single shard: apply filters, compute shortest_proof, and limit statements per uuid.

        Args:
            shard_id (int): The ID of the shard to process.
        """
        input_path = self.input_dir / f"shard_{shard_id}.parquet"
        output_path = self.output_dir / f"shard_{shard_id}.parquet"

        df = pd.read_parquet(input_path)
        logger.info(f"Processing shard {shard_id} with {len(df)} samples.")

        # Apply filters
        if "n_correct_proofs" in df.columns.tolist():
            df = df[df["n_correct_proofs"] > 0]
            logger.debug(
                f"Shard {shard_id}: After n_correct_proofs filter: {len(df)} samples."
            )

        if (
            "n_correct_proofs" in df.columns.tolist()
            and "is_negation" in df.columns.tolist()
        ):
            df["proof_rate"] = df["n_correct_proofs"] / df["n_proofs"]
            df["is_negation"] = df["is_negation"].fillna(False)
            # create a new column with random number 0-1
            df["random"] = np.random.rand(len(df))
            # Apply proof_rate filter
            sampling_proba = (1 - df["is_negation"].astype(int) * 0.8) * (
                1 - 0.95 * df["proof_rate"]
            ) ** self.beta
            df = df[sampling_proba > df["random"]]
            logger.debug(
                f"Shard {shard_id}: After proof_rate filter: {len(df)} samples."
            )

        # Apply 'use 0' tactic filter
        if "is_negation" in df.columns.tolist():
            df = df[
                df.apply(lambda x: x["is_negation"] or (not _proof_using_0(x)), axis=1)
            ]
            logger.debug(
                f"Shard {shard_id}: After 'use 0' tactic filter: {len(df)} samples."
            )

        # Apply 'sorry' count filter
        df = df[df["formal_statement"].str.count("sorry") == 1]
        logger.debug(
            f"Shard {shard_id}: After 'sorry' count filter: {len(df)} samples."
        )

        # Compute shortest_proof
        if "correct_proof_samples" in df.columns.tolist():
            df["shortest_proof"] = df["correct_proof_samples"].apply(
                lambda proofs: min(
                    (process_proof.remove_comments(p) for p in proofs), key=len
                )
            )
        elif "formal_proof" in df.columns.tolist():
            df["shortest_proof"] = df["formal_proof"]

        # Limit to 3 statements per uuid
        df = df.drop_duplicates(["uuid", "shortest_proof"])
        logger.debug(
            f"Shard {shard_id}: After drop_duplicates shortest_proof per uuid: {len(df)} samples."
        )
        n = 3
        df = (
            df.groupby("uuid")
            .apply(lambda x: find_different_proof(x, n))
            .reset_index(drop=True)
        )
        logger.debug(
            f"Shard {shard_id}: After limiting to {n} statements per uuid: {len(df)} samples."
        )

        # Write processed shard to output
        df.to_parquet(output_path, index=False)
        logger.info(f"Processed shard {shard_id} written to {output_path}.")

    def consolidate(self, num_proc: int = None):
        """
        Consolidate all processed shards into a single dataset.
        """
        logger.info("Consolidating processed shards.")
        final_ds = load_dataset(str(self.output_dir), split="train", num_proc=num_proc)
        logger.info(f"Consolidated dataset has {len(final_ds)} samples.")

        ds_len = len(final_ds)
        final_ds = final_ds.filter(
            lambda x: process_proof.is_proof_splitable(x["shortest_proof"]),
            num_proc=num_proc,
        )
        logger.info(
            f"Filtered {ds_len - len(final_ds)} samples with unsplitable proofs."
        )

        # insert informal problem as comment before the formal statement
        if self.informal_prefix:

            def _insert_informal(sample):
                natural_language = (
                    ""
                    if sample["natural_language"] is None
                    else sample["natural_language"]
                )
                informal = natural_language.split("The final answer is")[0].strip()
                if not sample["is_negation"]:
                    shortest_proof = process_statement.insert_informal(
                        sample["shortest_proof"], informal
                    )
                else:
                    shortest_proof = sample["shortest_proof"]
                return {
                    "shortest_proof": shortest_proof,
                }

            final_ds = final_ds.map(
                _insert_informal,
                num_proc=num_proc,
            )

        def _split_proof(sample):
            proof_input, proof_out = process_proof.split_proof(sample["shortest_proof"])
            return {
                "proof_input": proof_input,
                "proof_output": proof_out,
                "formal_proof": sample["shortest_proof"],
            }

        final_ds = final_ds.map(_split_proof, num_proc=num_proc)

        columns = [
            "uuid",
            "statement_id",
            "formal_statement",
            "proof_input",
            "proof_output",
            "formal_proof",
        ]
        if "source" in final_ds.features:
            columns.append("source")
        final_ds = final_ds.select_columns(columns)
        if self.push_name:
            final_ds.push_to_hub(self.push_name, private=True)
        logger.info("Consolidation complete.")

    def process(self, max_workers: int = None):
        """
        Shard the dataset and process each shard concurrently.

        Args:
            uuids_per_shard (int): Number of unique UUIDs per shard.
            max_workers (int, optional): The maximum number of worker processes. Defaults to number of CPUs.
        """
        # Identify all shard IDs
        shard_files = list(self.input_dir.glob("shard_*.parquet"))
        shard_ids = [int(f.stem.split("_")[1]) for f in shard_files]
        logger.info(f"Found {len(shard_ids)} shards to process.")

        # Process shards concurrently
        logger.info("Starting concurrent shard processing.")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_shard, shard_id): shard_id
                for shard_id in shard_ids
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing shards"
            ):
                shard_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing shard {shard_id}: {e}")

        logger.info("All shards have been processed.")

        # Consolidate processed shards
        self.consolidate(num_proc=max_workers)


def run_pipeline(beta, source, push_name=None, informal_prefix=False):
    """
    Run the full pipeline.
    """
    pipeline = AutoSftDatasetPipeline(source, beta, informal_prefix, push_name)

    # fix random seed
    np.random.seed(37)

    # Start the processing pipeline
    pipeline.process(max_workers=multiprocessing.cpu_count())


def run_one(source):
    """
    Run one for debugging.
    """
    pipeline = AutoSftDatasetPipeline(source)
    pipeline.process_shard(0)


if __name__ == "__main__":
    """Usage:
    python -m autoformalizer.sft_data.pipelines.auto_gen.base run_one \
        --source auto-statements-moon-santa-prover-v1.2

    python -m autoformalizer.sft_data.pipelines.auto_gen.base run_pipeline \
        --beta 1 \
        --source auto-statements-moon-flying-ant-prover-v1 \
        --push_name AI-MO/auto-sft-moon-flying-ant-prover-v1-base \
        --informal_prefix True

    python -m autoformalizer.sft_data.pipelines.auto_gen.base run_pipeline \
        --beta 1 \
        --source auto-statements-moon-santa-prover-v1 \
        --push_name AI-MO/auto-sft-moon-santa-prover-v3-base \
        --informal_prefix True

    """
    fire.Fire({"run_pipeline": run_pipeline, "run_one": run_one})
