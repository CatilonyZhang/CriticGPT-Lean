import multiprocessing
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import fire
import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk
from loguru import logger
from tqdm import tqdm

from autoformalizer.sft_data import util
from autoformalizer.sft_data.pipelines.auto_gen.base import AutoSftDatasetPipeline


class CustomFilterPipeline(AutoSftDatasetPipeline):
    SOURCE = "AI-MO/wholeproof-pt-250119-v3"

    def __init__(
        self,
        beta: float = 1,
        informal_prefix: bool = False,
        push_name: str = None,
        local_filters: list[Callable] = None,
        local_filter_kwargs: list[dict] = None,
        global_filters: list[Callable] = None,
        global_filter_kwargs: list[dict] = None,
    ):
        self.working_dir = pathlib.Path(os.environ["SFT_DATA_DIR"])
        self.input_dir = self.working_dir / self.SOURCE / "data"
        self.shard_count = len(list(self.input_dir.glob("*.parquet")))
        self.shard_output_dir = self.working_dir / "shards" / self.SOURCE
        self.final_output_dir = self.working_dir / "processed" / self.SOURCE

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Source directory {self.input_dir} not found.")
        self.shard_output_dir.mkdir(parents=True, exist_ok=True)
        self.final_output_dir.mkdir(parents=True, exist_ok=True)
        self.beta = beta
        self.push_name = push_name
        self.informal_prefix = informal_prefix
        self.local_filters = local_filters or []
        self.local_filter_kwargs = local_filter_kwargs or []
        self.global_filters = global_filters or []
        self.global_filter_kwargs = global_filter_kwargs or []

    def process_shard(self, shard_id: int):
        """
        Apply filters to a single shard.

        Args:
            shard_id (int): The ID of the shard to process.
        """
        input_path = (
            self.input_dir / f"train-{shard_id:05d}-of-{self.shard_count:05d}.parquet"
        )
        output_path = (
            self.shard_output_dir
            / f"train-{shard_id:05d}-of-{self.shard_count:05d}.parquet"
        )

        df = pd.read_parquet(input_path)
        logger.info(f"Processing shard {shard_id} with {len(df)} samples.")

        # Apply filters
        for filter_fn, filter_kwargs in zip(
            self.local_filters, self.local_filter_kwargs
        ):
            df = getattr(util, filter_fn)(df, **filter_kwargs)
            logger.debug(
                f"Shard {shard_id}: After filter {filter_fn}: {len(df)} samples."
            )

        # Write processed shard to output
        df.to_parquet(output_path, index=False)
        logger.info(f"Processed shard {shard_id} written to {output_path}.")

    def consolidate(self, num_proc: int = None):
        """
        Consolidate all processed shards into a single dataset.
        """
        logger.info("Consolidating processed shards.")
        try:
            final_ds = load_dataset(
                str(self.shard_output_dir), split="train", num_proc=num_proc
            )
        except ValueError:
            final_ds = load_from_disk(str(self.shard_output_dir))
        logger.info(f"Consolidated dataset has {len(final_ds)} samples.")

        # Apply filters
        for filter_fn, filter_kwargs in zip(
            self.global_filters, self.global_filter_kwargs
        ):
            filter_kwargs.update(
                {"source_name": self.SOURCE, "output_path": self.final_output_dir}
            )
            final_ds = getattr(util, filter_fn)(final_ds, **filter_kwargs)
            logger.debug(
                f"Consolidated dataset after global filter {filter_fn}: {len(final_ds)} samples."
            )

        final_ds = final_ds.select_columns(
            [
                "uuid",
                "text",
                "proof_input",
                "proof_output",
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
        logger.info(f"Found {self.shard_count} shards to process.")

        # Process shards concurrently
        logger.info("Starting concurrent shard processing.")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_shard, shard_id): shard_id
                for shard_id in range(self.shard_count)
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


def run_pipeline(**kwargs):
    """
    Run the full pipeline.
    """
    pipeline = CustomFilterPipeline(**kwargs)

    # fix random seed
    np.random.seed(37)

    # Start the processing pipeline
    pipeline.process(max_workers=multiprocessing.cpu_count())


def run_one(**kwargs):
    """ """
    pipeline = CustomFilterPipeline(**kwargs)
    pipeline.process_shard(0)


if __name__ == "__main__":
    """Usage:
    python -m autoformalizer.sft_data.pipelines.auto_gen.auto_filter \
        run_one \
      --local_filters \
        '[filter_by_str_length, \
          filter_by_line_length, \
          filter_by_token_length]' \
      --local_filter_kwargs \
      '[{"min_length": 3, "max_length": 3000}, \
        {"min_length": 2, "max_length": 150}, \
        {"min_length": 3, "max_length": 1000}]'

    python -m autoformalizer.sft_data.pipelines.auto_gen.auto_filter \
        run_pipeline \
      --push_name "AI-MO/wholeproof-pt-250119-v3_filtered_gn10" \
      --global_filters \
        '[filter_by_near_dedup]' \
      --global_filter_kwargs \
        '[{"ngram": 10, "dedup_column": "proof_output"}]'

    python -m autoformalizer.sft_data.pipelines.auto_gen.auto_filter \
        run_pipeline \
      --push_name "AI-MO/wholeproof-pt-250119-v3_openai-embedding" \
      --local_filters \
          '[filter_by_str_length, \
          filter_by_line_length]' \
      --local_filter_kwargs \
        '[{"min_length": 3, "max_length": 3000}, \
        {"min_length": 2, "max_length": 150}]' \
      --global_filters \
        '[filter_by_gt_embedding]' \
      --global_filter_kwargs \
        '[{"target_column": "proof_output", \
          "gt_dataset": "AI-MO/human-proofs-sft", \
          "similar_threshold": 0.7, \
          "model": "bge-m3", \
          "base_url": "http://localhost/v1/", \
          "api_key": "EMPTY", \
          "num_parallel": 1}]'
    """
    fire.Fire({"run_pipeline": run_pipeline, "run_one": run_one})
