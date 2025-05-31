#!/usr/bin/env python3

import concurrent.futures
import multiprocessing as mp
import pathlib
import sys
from dataclasses import dataclass, field
from time import time
from typing import Any, List, Tuple

import fire
import yaml
from datasets import load_dataset
from loguru import logger

from autoformalizer.jobs import statement_formalizer
from autoformalizer.jobs.input_sharder import InputSharder
from autoformalizer.jobs.prover import ProofGenerator, ProofVerifier


@dataclass
class Config:
    config_file: str  # Path to the config file

    # Common parameters
    platform: str = field(init=False)
    job_id: str = field(init=False)
    input_dataset_path: str = field(init=False)
    shard_size: int = field(init=False)
    select_range: int = field(init=False)
    max_workers: int = field(init=False)
    repeat: int = field(init=False)

    # Proof filtering
    enable_proof_filtering: bool = field(init=False)
    filter_threshold: float = field(init=False)
    proof_record_dir: pathlib.Path = field(init=False)
    proof_record_lock: Any = field(init=False)

    # Platform-specific parameters
    autoformalizer_params: dict = field(init=False)
    prover_params: dict = field(init=False)
    verifier_params: dict = field(init=False)

    # Working directories
    working_dir: pathlib.Path = field(init=False)
    problems_dir: pathlib.Path = field(init=False)
    statements_dir: pathlib.Path = field(init=False)
    verified_statements_dir: pathlib.Path = field(init=False)
    proofs_dir: pathlib.Path = field(init=False)
    verified_proofs_dir: pathlib.Path = field(init=False)
    huggingface_cache_dir: pathlib.Path = field(init=False)

    def __post_init__(self):
        # Load the configuration from the specified YAML file
        self.load_config()

        home_dir = pathlib.Path.home()

        # Format the working directory with job_id
        self.working_dir = pathlib.Path(
            self.working_dir.format(home=home_dir, job_id=self.job_id)
        ).expanduser()

        # Format derived directories with working_dir
        self.problems_dir = pathlib.Path(
            self.problems_dir.format(working_dir=str(self.working_dir))
        )
        self.statements_dir = pathlib.Path(
            self.statements_dir.format(working_dir=str(self.working_dir))
        )
        self.verified_statements_dir = pathlib.Path(
            self.verified_statements_dir.format(working_dir=str(self.working_dir))
        )
        self.proofs_dir = pathlib.Path(
            self.proofs_dir.format(working_dir=str(self.working_dir))
        )
        self.verified_proofs_dir = pathlib.Path(
            self.verified_proofs_dir.format(working_dir=str(self.working_dir))
        )
        self.proof_record_dir = pathlib.Path(
            self.proof_record_dir.format(working_dir=self.working_dir)
        )
        if self.huggingface_cache_dir:
            self.huggingface_cache_dir = pathlib.Path(self.huggingface_cache_dir)
        # create proof record file if not exist
        self.proof_record_lock = mp.Manager().Lock()

    def load_config(self):
        """Load configuration data from the specified YAML file."""
        try:
            with open(self.config_file, "r") as file:
                config_data = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_file} not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

        # Assign common parameters
        self.platform = config_data.get("platform")
        self.max_workers = config_data.get("max_workers", 1)
        if self.platform not in ["numina", "moonshot"]:
            raise ValueError(f"Unsupported platform: {self.platform}")

        self.job_id = config_data.get("job_id")
        self.input_dataset_path = config_data.get("input_dataset_path")
        self.shard_size = config_data["shard_size"]
        self.select_range = config_data.get("select_range")
        self.repeat = config_data.get("repeat", 1)

        # Proof filtering
        self.enable_proof_filtering = config_data.get("enable_proof_filtering", False)
        self.filter_threshold = config_data.get("filter_threshold", 4)
        self.proof_record_dir = config_data.get("proof_record_dir")

        # Assign platform-specific parameters
        self.autoformalizer_params = config_data.get("autoformalizer_params", {})
        assert isinstance(self.autoformalizer_params["add_negation"], bool)
        # Assign default values for add_informal
        self.autoformalizer_params["add_informal"] = config_data.get(
            "add_informal", True
        )
        self.prover_params = config_data.get("prover_params", {})
        self.verifier_params = config_data.get("verifier_params", {})

        # Assign working directory and derived directories
        self.working_dir = config_data.get("working_dir")
        self.problems_dir = config_data.get("problems_dir")
        self.statements_dir = config_data.get("statements_dir")
        self.verified_statements_dir = config_data.get("verified_statements_dir")
        self.proofs_dir = config_data.get("proofs_dir")
        self.verified_proofs_dir = config_data.get("verified_proofs_dir")
        self.huggingface_cache_dir = config_data.get("huggingface_cache_dir", None)


def setup_logging(log_file: pathlib.Path):
    logger.remove()  # Remove default stderr logger
    logger.add(sys.stdout, level="INFO")  # Add stdout logger
    logger.add(
        log_file, rotation="1 week", retention="4 weeks", level="DEBUG"
    )  # Add file logger


def setup_directories(config: Config):
    logger.info(f"Creating working directories at {config.working_dir}")
    config.working_dir.mkdir(parents=True, exist_ok=True)
    for dir_path in [
        config.problems_dir,
        config.statements_dir,
        config.verified_statements_dir,
        config.proofs_dir,
        config.verified_proofs_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
    logger.debug("All directories created successfully.")


def shard_input_data(config: Config) -> List[Tuple[str, pathlib.Path]]:
    logger.info("Sharding input data...")
    ds = load_dataset(
        config.input_dataset_path, split="train", cache_dir=config.huggingface_cache_dir
    )
    if config.select_range:
        ds = ds.select(range(config.select_range))  # For testing
    sharder = InputSharder(
        ds,
        output_path=config.problems_dir,
        shard_size=config.shard_size,
        repeat=config.repeat,
    )
    logger.info(f"Total shards created: {len(sharder.shards)}")
    return sharder.shards


def process_shard(shard: Tuple[str, pathlib.Path], config: Config) -> bool:
    shard_id, shard_path = shard
    logger.info(f"Processing shard {shard_id} at {shard_path}")

    try:
        # Step 2: Autoformalization Inference
        formalizer = statement_formalizer.StatementFormalizer(
            model=config.autoformalizer_params["model_path"],
            openai_api_base=config.autoformalizer_params["url"],
            openai_api_key="EMPTY",
        )

        sampling_params = {
            "n_samples": config.autoformalizer_params[
                "num_samples_per_informal_statement"
            ],
            "temperature": 0.8,
            "max_tokens": 1024,
            "num_threads": config.autoformalizer_params["num_threads"],
            "add_negation": config.autoformalizer_params["add_negation"],
            "add_informal": config.autoformalizer_params["add_informal"],
        }

        status, row = formalizer(
            shard_id, config.problems_dir, config.statements_dir, **sampling_params
        )
        if not status:
            logger.warning(f"Autoformalization failed for shard {shard_id}")
            return False

        # Step 3: Verification of Autoformalized Statements
        verifier = statement_formalizer.StatementVerifier(
            client_url=config.verifier_params["url"],
            client_api_key=None,
            timeout=config.verifier_params["timeout"],
            num_threads=config.verifier_params["num_threads"],
            batch_size=config.verifier_params["batch_size"],
        )

        status, row = verifier(
            shard_id=shard_id,
            input_dir=config.statements_dir,
            output_dir=config.verified_statements_dir,
        )
        if not status:
            logger.warning(f"Verification failed for shard {shard_id}")
            return False

        # Step 4: Proof Generation
        prover = ProofGenerator(
            openai_api_base=config.prover_params["url"],
            model=config.prover_params["model_path"],
            openai_api_key="EMPTY",  # Replace with actual key if needed
            enable_proof_filtering=config.enable_proof_filtering,
            proof_record_dir=config.proof_record_dir,
            proof_record_lock=config.proof_record_lock,
            filter_threshold=config.filter_threshold,
        )

        prompt_template = (
            "Complete the following Lean 4 code:\n\n```lean4\n{formal_statement}"
        )

        sampling_params = {
            "max_tokens": config.prover_params.get("max_tokens", 2048),
            "temperature": 1.0,
            "n": config.prover_params.get("num_samples_per_statement", 1),
            "num_threads": config.prover_params.get("num_threads", 1),
        }

        status, row = prover(
            shard_id=shard_id,
            input_dir=config.verified_statements_dir,
            output_dir=config.proofs_dir,
            prompt_template=prompt_template,
            **sampling_params,
        )
        if not status:
            logger.warning(f"Proof generation failed for shard {shard_id}")
            return False

        # Step 5: Proof Verification
        proof_verifier = ProofVerifier(
            client_url=config.verifier_params["url"],
            client_api_key=None,
            timeout=config.verifier_params["timeout"],
            num_threads=config.verifier_params["num_threads"],
            batch_size=config.verifier_params["batch_size"],
            enable_proof_filtering=config.enable_proof_filtering,
            proof_record_dir=config.proof_record_dir,
            proof_record_lock=config.proof_record_lock,
        )

        status, sid = proof_verifier(
            shard_id=shard_id,
            input_dir=config.proofs_dir,
            output_dir=config.verified_proofs_dir,
        )
        if not status:
            logger.warning(f"Proof verification failed for shard {shard_id}")
            return False

        # Step 6: Prover Training (TODO)
        # Placeholder for future implementation

        logger.info(f"Shard {shard_id} processed successfully.")
        return True

    except Exception as e:
        logger.error(f"Exception while processing shard {shard_id}: {e}")
        return False


def main(config_path: str):
    # log the total processing time
    start_time = time()

    # Initialize configuration
    config = Config(config_file=config_path)

    # Setup logging
    log_file = config.working_dir / "pipeline.log"
    setup_logging(log_file)

    # Setup directories
    setup_directories(config)

    # Step 1: Shard the input data
    shards = shard_input_data(config)

    # Define the number of workers based on your system's resources
    # You may need to adjust this based on the resource consumption of each step
    max_workers = config.max_workers  # Example: Adjust as needed

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

    # After all shards are processed, you can perform any final aggregation or cleanup
    logger.info("All shards have been processed.")

    # run some statistics on the verified proofs
    verified_ds = load_dataset(str(config.verified_proofs_dir), split="train")
    verified_ds = verified_ds.filter(lambda x: not x["statement_id"].startswith("neg_"))
    print(verified_ds)
    valid_rate = sum(verified_ds["is_valid_no_sorry"]) / len(verified_ds)
    logger.info(f"Valid rate: {valid_rate}")

    # prove rate by uuid
    verified_df = verified_ds.select_columns(["uuid", "is_valid_no_sorry"]).to_pandas()
    uuid_proofs = verified_df.groupby("uuid").sum()
    # find uuid with at least one proof
    prove_rate = len(uuid_proofs[uuid_proofs["is_valid_no_sorry"] > 0]) / len(
        uuid_proofs
    )
    logger.info(f"Prove rate by uuid: {prove_rate}")

    # negative statement prove rate
    neg_verified_ds = load_dataset(str(config.verified_proofs_dir), split="train")
    neg_verified_ds = neg_verified_ds.filter(
        lambda x: x["statement_id"].startswith("neg_")
    )
    neg_verified_df = neg_verified_ds.select_columns(
        ["statement_id", "is_valid_no_sorry"]
    ).to_pandas()
    neg_proofs = neg_verified_df.groupby("statement_id").sum()
    neg_prove_rate = sum(neg_proofs["is_valid_no_sorry"]) / len(neg_proofs)
    logger.info(f"Negative statement prove rate: {neg_prove_rate}")

    # log the total processing time
    end_time = time()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    """
    Usage:
    python -m autoformalizer.jobs.main autoformalizer/jobs/configs/numina_config_template.yaml

    should give something like:
    2024-12-26 05:05:26.239 | INFO     | __main__:main:294 - Valid rate: 0.3599537037037037
    2024-12-26 05:05:26.246 | INFO     | __main__:main:303 - Prove rate by uuid: 0.5526315789473685
    2024-12-26 05:05:26.294 | INFO     | __main__:main:315 - Negative statement prove rate: 0.34953703703703703
    2024-12-26 05:05:26.295 | INFO     | __main__:main:319 - Total processing time: 80.88 seconds
    """
    fire.Fire(main)
