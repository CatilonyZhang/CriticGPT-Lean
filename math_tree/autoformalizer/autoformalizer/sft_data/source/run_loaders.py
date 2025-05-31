import os
import pathlib
import sys

import fire
from loguru import logger

from autoformalizer.sft_data.source import loaders


# config datasets
def get_datasets_config():
    if "SFT_DATA_DIR" not in os.environ:
        raise ValueError("SFT_DATA_DIR environment variable must be set")
    else:
        SFT_DATA_DIR = os.environ["SFT_DATA_DIR"]

    datasets_to_load = [
        {
            "loader": loaders.UuidDatasetLoader,
            "args": {
                "dataset_name": "AI-MO/auto-statements-moon-flying-ant-prover-v1",
                "uuid_column": "uuid",
                "shard_dir": SFT_DATA_DIR
                + "/auto-statements-moon-flying-ant-prover-v1/source",
                "uuids_per_shard": 500,
            },
        },
        {
            "loader": loaders.UuidDatasetLoader,
            "args": {
                "dataset_name": "AI-MO/auto-statements-moon-santa-prover-v1",
                "uuid_column": "uuid",
                "shard_dir": SFT_DATA_DIR
                + "/auto-statements-moon-santa-prover-v1/source",
                "uuids_per_shard": 500,
            },
        },
    ]
    return datasets_to_load


# Configure logging
def setup_logging(log_dir: str = "logs"):
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add("logs/sft_loader.log", rotation="1 week", level="INFO")
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time}</green> | <level>{message}</level>",
    )


def main():
    setup_logging()
    logger.info("Starting dataset loaders...")
    datasets_to_load = get_datasets_config()
    for dataset_config in datasets_to_load:
        shard_dir = pathlib.Path(dataset_config["args"]["shard_dir"])
        if shard_dir.exists():
            logger.info(f"Shard directory {shard_dir} already exists. Skipping.")
            continue
        loader = dataset_config["loader"](**dataset_config["args"])
        # if exist, skip
        loader.load_data()
        loader.shard()


if __name__ == "__main__":
    fire.Fire(main)
