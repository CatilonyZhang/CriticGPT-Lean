import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import fire
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from autoformalizer.eval_utils.model_feedback import create_feedback_service


class JudgeTaggerPipeline:

    SOURCE = "auto_statements_moon_santa_prover_v1"

    def __init__(self):
        self.working_dir = pathlib.Path(os.environ["SFT_DATA_DIR"])
        self.judge_yaml = pathlib.Path(os.environ["JUDGE_YAML"])
        self.input_dir = self.working_dir / self.SOURCE / "source"
        self.output_dir = self.working_dir / self.SOURCE / "tag"
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Source directory {self.input_dir} not found.")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_shard(self, shard_id: int):
        input_path = self.input_dir / f"shard_{shard_id}.parquet"
        output_path = self.output_dir / f"shard_{shard_id}.parquet"

        df = pd.read_parquet(input_path)
        logger.info(f"Processing shard {shard_id} with {len(df)} samples.")

        # statement judge model
        service = create_feedback_service(self.judge_yaml)
        feedbacks = service.process_parallel_feedback(
            df["formal_statement"].tolist(),
            df["natural_language"].tolist(),
            temperature=0.0,
            num_votes=1,
        )
        df["tag_statement_judge"] = np.array([x[0] for x in feedbacks])

        # Write processed shard to output
        df.to_parquet(output_path, index=False)
        logger.info(f"Processed shard {shard_id} written to {output_path}.")

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


def run_pipeline(max_workers=8):
    """
    Run the full pipeline.
    """
    pipeline = JudgeTaggerPipeline()

    # fix random seed
    np.random.seed(37)

    # Start the processing pipeline
    pipeline.process(max_workers=max_workers)


def run_one():
    """
    Run one for debugging.
    """
    pipeline = JudgeTaggerPipeline()
    pipeline.process_shard(0)


if __name__ == "__main__":
    """Usage:
    python -m autoformalizer.sft_data.pipelines.auto_gen.tagger_judgement run_one

    python -m autoformalizer.sft_data.pipelines.auto_gen.tagger_judgement run_pipeline
    """
    fire.Fire({"run_pipeline": run_pipeline, "run_one": run_one})
