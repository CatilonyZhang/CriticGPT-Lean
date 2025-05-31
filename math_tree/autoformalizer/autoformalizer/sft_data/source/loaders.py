# uuid_dataset_loader.py

import multiprocessing
import pathlib
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from autoformalizer.data_utils.hf_datasets import IndexDataset


class UuidDatasetLoader:
    """
    Loader for datasets indexed by a UUID column.
    All data points with the same UUID are sharded together.
    """

    def __init__(
        self,
        dataset_name: str,
        uuid_column: str,
        shard_dir: str,
        uuids_per_shard: int,
        num_processes: int = None,
    ):
        """
        Initialize the loader.

        Args:
            dataset_name (str): HuggingFace dataset name or path.
            uuid_column (str): Column name to index by UUID.
            output_dir (str): Directory where shards will be saved.
        """
        self.dataset_name = dataset_name
        self.uuid_column = uuid_column
        self.shard_dir = pathlib.Path(shard_dir)
        if not self.shard_dir.exists():
            self.shard_dir.mkdir(parents=True, exist_ok=True)
        if num_processes is None:
            self.num_processes = multiprocessing.cpu_count()
        else:
            self.num_processes = num_processes
        self.dataset = None
        self.index_dataset = None
        self.uuids_per_shard = uuids_per_shard

    def load_data(self):
        """
        Load the HuggingFace dataset and create an index based on the UUID column.
        """
        logger.info(f"Loading dataset '{self.dataset_name}'...")
        self.dataset = load_dataset(self.dataset_name, split="train")

        if self.dataset_name == "AI-MO/auto-statements-moon-flying-ant-prover-v1":
            self.dataset = self.dataset.map(
                lambda x: {
                    "correct_proof_samples": [
                        e["formal_proof"] for e in x["correct_proof_samples"]
                    ]
                }
            )

        logger.info(f"Dataset loaded with {len(self.dataset)} samples.")

        logger.info(f"Creating index on '{self.uuid_column}'...")
        self.index_dataset = IndexDataset(self.dataset, indices=[self.uuid_column])
        logger.info("Indexing complete.")

    def shard(self):
        """
        Shard the dataset into Parquet files based on UUIDs.
        """
        uuids_per_shard = self.uuids_per_shard
        if self.index_dataset is None:
            logger.error("Data not loaded. Call `load_data()` before sharding.")
            return

        unique_uuids = list(
            set(self.index_dataset._index_to_row[self.uuid_column].keys())
        )
        logger.info(f"Total unique '{self.uuid_column}': {len(unique_uuids)}")

        shards = [
            unique_uuids[i : i + uuids_per_shard]
            for i in range(0, len(unique_uuids), uuids_per_shard)
        ]
        logger.info(f"Total shards to create: {len(shards)}")

        args = [(shard_id, shard_uuids) for shard_id, shard_uuids in enumerate(shards)]

        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            list(
                tqdm(
                    executor.map(self._shard_worker, args),
                    total=len(args),
                    desc="Sharding datasets",
                )
            )

        logger.info(f"Sharding complete. Shards saved in '{self.shard_dir}'.")

    def _shard_worker(self, args):
        """
        Worker function to create a single shard from a list of UUIDs.

        Args:
            args (tuple): (shard_id, shard_uuids)
        """
        shard_id, shard_uuids = args
        shard_data = []

        for uuid in shard_uuids:
            rows = self.index_dataset.get_rows_by_index(self.uuid_column, uuid)
            if rows:
                shard_data.extend(rows)

        if shard_data:
            shard_df = pd.DataFrame(shard_data)
            shard_path = self.shard_dir / f"shard_{shard_id}.parquet"
            shard_df.to_parquet(shard_path, index=False)
            logger.info(f"Shard {shard_id} written with {len(shard_df)} rows.")
        else:
            logger.warning(f"Shard {shard_id} is empty. No file created.")
