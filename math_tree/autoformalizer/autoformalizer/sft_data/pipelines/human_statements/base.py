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

from autoformalizer.data_utils import process_proof
from autoformalizer.sft_data.util import find_different_proof


class HumanSftDatasetPipeline:

    def __init__(self, source, pick_n, push_name: str = None):
        self.SOURCE = source
        self.working_dir = pathlib.Path(os.environ["SFT_DATA_DIR"])
        self.input_dir = self.working_dir / self.SOURCE / "source"
        self.output_dir = self.working_dir / self.SOURCE / "processed"

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Source directory {self.input_dir} not found.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pick_n = pick_n
        self.push_name = push_name

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

        if "tags" in df.columns.tolist():
            df = df[~df["tags"].isna()]
            logger.debug(f"Shard {shard_id}: After tags na filter: {len(df)} samples.")
            tag_remove = "human-statements-source:PutnamBench-lean4-train"
            df = df[df["tags"].map(lambda x: tag_remove not in x)]
            logger.debug(
                f"Shard {shard_id}: After tag_remove filter: {len(df)} samples."
            )

        samples = []
        for i, sample in df.iterrows():
            proofs = list(
                set([x["formal_proof"] for x in sample["correct_proof_samples"]])
            )
            selected = proofs
            for proof in selected:
                if len(process_proof.get_statement_split_indexes(proof)) > 0:
                    proof_input, proof_out = process_proof.split_proof(proof)
                    samples.append(
                        {
                            "uuid": sample["uuid"],
                            "formal_statement": sample["formal_statement"],
                            "statement_id": sample["statement_id"],
                            "proof_input": proof_input,
                            "proof_output": proof_out,
                            "formal_proof": proof,
                        }
                    )
                else:
                    print(sample["uuid"])
        df = pd.DataFrame(samples)

        # Limit to 10 statements per uuid
        n = self.pick_n
        df = (
            df.groupby("uuid")
            .apply(lambda x: find_different_proof(x, n, "formal_proof"))
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

        final_ds = final_ds.select_columns(
            [
                "uuid",
                "statement_id",
                "formal_statement",
                "proof_input",
                "proof_output",
                "formal_proof",
            ]
        )
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


def run_pipeline(pick_n, source, push_name=None):
    """
    Run the full pipeline.
    """
    pipeline = HumanSftDatasetPipeline(source, pick_n, push_name)

    # fix random seed
    np.random.seed(37)

    # Start the processing pipeline
    pipeline.process(max_workers=multiprocessing.cpu_count())


def run_one():
    """
    Run one for debugging.
    """
    pipeline = HumanSftDatasetPipeline()
    pipeline.process_shard(0)


if __name__ == "__main__":
    """Usage:
    python -m autoformalizer.sft_data.pipelines.human_statements.base run_one \
        --source human-statements-moon-new-year-prover-v1-merged


    python -m autoformalizer.sft_data.pipelines.human_statements.base run_pipeline \
        --source human-statements-moon-new-year-prover-v1-merged \
        --pick_n 10 \
        --push_name AI-MO/human-statements-sft-v1pick10

    """
    fire.Fire({"run_pipeline": run_pipeline, "run_one": run_one})
