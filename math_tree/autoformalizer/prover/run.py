import datetime
import glob
import json
import os
import pathlib
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, current_process
from pprint import pprint

# Import the Config class
from config import Config
from datasets import concatenate_datasets, load_dataset
from loguru import logger
from tqdm import tqdm

from autoformalizer.clients.lean4_client import Lean4Client
from autoformalizer.jobs.input_sharder import InputSharder
from prover.bfs_prover import BestFirstSearchProver, SearchResult
from prover.search_tree import Status
from prover.tactic_generator import APITacticGenerator


def setup_logging(config: Config):
    """
    Setup logging using Loguru.

    Args:
        config (Config): Configuration object.
    """
    logger.remove()  # Remove default logger

    # Ensure logging directory exists
    os.makedirs(config.logging_dir, exist_ok=True)

    # Add a global logger to stdout
    logger.add(sys.stdout, level="INFO")

    # Individual process loggers will be handled in `run_prover`


def shard_input_data(datasets_list: list, config: Config) -> list:
    """
    Concatenate all datasets and shard the combined dataset.

    Args:
        datasets_list (list): List of dataset configurations.
        repeat (int): Number of times to repeat the sharding process.
        config (Config): Loaded configuration object.

    Returns:
        list: A list of tuples containing shard_id and shard_path.
    """
    logger.info("Starting dataset concatenation and sharding process...")

    concatenated_ds = None
    datasets = []
    for ds_info in datasets_list:
        logger.info(f"Loading dataset: {ds_info.name}, split: {ds_info.split}")
        try:
            loaded_ds = load_dataset(ds_info.name, split=ds_info.split)
            loaded_ds = loaded_ds.select_columns(
                ["uuid", "statement_id", "formal_statement"]
            )
            datasets.append(loaded_ds)
            logger.info(f"Dataset {ds_info.name} loaded with {len(loaded_ds)} samples.")
        except Exception as e:
            logger.error(f"Failed to load or process dataset {ds_info.name}: {e}")
            continue

    if not datasets:
        logger.error("No datasets loaded successfully. Exiting sharding process.")
        return []

    # Concatenate all datasets
    try:
        concatenated_ds = concatenate_datasets(datasets)
        logger.info(f"All datasets concatenated. Total samples: {len(concatenated_ds)}")
    except Exception as e:
        logger.error(f"Failed to concatenate datasets: {e}")
        return []

    # Ensure all statement_id are unique
    try:
        assert len(concatenated_ds["statement_id"]) == len(
            set(concatenated_ds["statement_id"])
        )
        logger.info("All statement_id values are unique.")
    except AssertionError:
        logger.error("Duplicate statement_id found in the concatenated dataset.")
        return []

    # Shard the concatenated dataset
    sharder = InputSharder(
        dataset_path=concatenated_ds,
        output_path=config.statement_dir,
        shard_size=config.shard_size if config.shard_size else 10000,
        columns_to_read=["uuid", "statement_id", "formal_statement"],
        repeat=config.pass_at,
    )

    shards = []
    for shard_id, shard_path in sharder:
        shards.append((shard_id, shard_path))
        logger.info(f"Shard {shard_id} saved to {shard_path}")

    logger.info(f"Total shards created: {len(shards)}")
    return shards


def process_shard(
    shard_id: int, shard_path: pathlib.Path, config: Config, result_queue, debug=False
) -> None:
    """
    Process a single shard: load tasks and run the prover.

    Args:
        shard_id (int): Identifier for the shard.
        shard_path (pathlib.Path): Path to the shard parquet file.
        config (Config): Configuration object.
        result_queue (multiprocessing.Queue): Queue to put results.

    Returns:
        None
    """
    # Load shard data
    try:
        shard = load_dataset("parquet", data_files=str(shard_path), split="train")
        logger.info(
            f"Loaded shard {shard_id} from {shard_path}, containing {len(shard)} tasks."
        )
        if debug and (
            "ec7da13d-4842-5530-95ab-d3f8c4ef1701" not in shard["statement_id"]
        ):
            return
    except Exception as e:
        logger.error(f"Failed to load shard {shard_path}: {e}")
        return

    # set up path for saving the data
    output_dir = pathlib.Path(config.verified_proofs_dir) / f"shard_{shard_id}/"
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_finished_file = output_dir / "shard_finished.txt"
    searched_statement_ids = []
    if shard_finished_file.exists():
        # get all the meta.json files
        for file in glob.glob(
            str(output_dir / "**" / "meta_data.json"), recursive=True
        ):
            with open(file, "r") as f:
                meta = json.load(f)
                statement_id = pathlib.Path(file).parent.name
                if meta["search_result"] is not None:
                    result_queue.put(
                        {
                            "statement_id": statement_id,
                            "result": SearchResult.from_dict(meta["search_result"]),
                            "error": None,
                        }
                    )
                    searched_statement_ids.append(statement_id)
        if len(searched_statement_ids) == len(shard):
            logger.info(f"Shard {shard_id} already processed. Skipping.")
            return

    # Setup logger for each process
    logger.remove()

    process_id = current_process()._identity[0] if current_process()._identity else 1
    log_file = pathlib.Path(config.logging_dir) / f"prover-{process_id}.log"

    if config.verbose:
        logger.add(log_file, level="DEBUG")
    else:
        logger.add(log_file, level="INFO")

    # Initialize components
    tactic_generator = APITacticGenerator(
        model_id=config.model_params.model_path[0],  # Assuming single model path
        base_url=config.model_params.url[0],
        tokenizer_path=config.model_params.tokenizer_path,
    )

    lean4_client = Lean4Client(
        url=config.verifier_params.url,
    )

    # Initialize Prover
    if config.search_params.search_method.lower() == "bfs":
        prover = BestFirstSearchProver(
            tac_gen=tactic_generator,
            lean4_client=lean4_client,
            timeout=config.search_params.search_timeout,
            num_sampled_tactics=config.search_params.num_sampled_tactics,
            max_expansions=config.search_params.max_expansions,
            prompt_type=config.model_params.prompt_type,
            temperature=config.model_params.temperature,
            max_length=config.model_params.max_length,
            debug=config.verbose,
            checkpoint_dir=None,
            serialize_interval=config.search_params.serialize_interval,
            resume_from_checkpoint=config.search_params.resume_from_checkpoint,
        )
    else:
        raise ValueError(
            f"Unsupported search method: {config.search_params.search_method}"
        )

    for task in shard:
        if task["statement_id"] in searched_statement_ids:
            continue
        try:
            logger.info(f"Processing task: {task}")
            checkpoint_dir = output_dir / f"{task['statement_id']}/"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            prover.checkpoint_dir = checkpoint_dir

            search_result = prover.search(task)
            result_queue.put(
                {
                    "statement_id": task["statement_id"],
                    "result": search_result,
                    "error": None,
                }
            )
            logger.info(
                f"Task: {[task['formal_statement']]}, Status: {search_result.status}"
            )

        except Exception as e:
            logger.error(f"Error processing task {task}: {e}")
            traceback.print_exc()
            result_queue.put({"statement_id": task["statement_id"], "error": str(e)})

    # Mark shard as finished
    with open(shard_finished_file, "w") as f:
        f.write("Shard processing completed.")

    logger.info(f"Shard {shard_id} completed.")


def print_status(search_results):
    """
    Print the status of the proof search.

    Args:
        status (Status): Status object.
    """
    status_for_statement_id = defaultdict(list)
    for res in search_results:
        if res["error"] is not None:
            status_for_statement_id[res["statement_id"]].append(Status.SYSTEM_ERROR)
        else:
            status_for_statement_id[res["statement_id"]].append(res["result"].status)
    total_status = {
        Status.PROVED: 0,
        Status.FAILED: 0,
        Status.OPEN: 0,
        Status.SYSTEM_ERROR: 0,
        Status.INIT_FAILED: 0,
    }
    for _, statuses in status_for_statement_id.items():
        if Status.PROVED in statuses:
            total_status[Status.PROVED] += 1
        elif Status.FAILED in statuses:
            total_status[Status.FAILED] += 1
        elif Status.OPEN in statuses:
            total_status[Status.OPEN] += 1
        elif Status.SYSTEM_ERROR in statuses:
            total_status[Status.SYSTEM_ERROR] += 1
        elif Status.INIT_FAILED in statuses:
            total_status[Status.INIT_FAILED] += 1
    logger.info(f"Status counts: {total_status}")
    logger.info(
        f"Prove rate: {total_status[Status.PROVED]}/{len(status_for_statement_id)},"
        + f" {total_status[Status.PROVED] / len(status_for_statement_id):.2f}"
    )


def run_experiment(config: Config, debug=False):
    """
    Run the proof generation experiment.

    Args:
        config (Config): Configuration object.
    """
    # Print configuration
    print("Running experiment with the following configuration:")
    pprint(config)

    # save the config
    with open(config.working_dir + "/config.json", "w") as f:
        json.dump(str(config), f)

    # Setup logging
    setup_logging(config)

    # Create necessary directories
    working_dir = pathlib.Path(config.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    statement_dir = pathlib.Path(config.statement_dir)
    statement_dir.mkdir(parents=True, exist_ok=True)

    verified_proofs_dir = pathlib.Path(config.verified_proofs_dir)
    verified_proofs_dir.mkdir(parents=True, exist_ok=True)

    # Shard the input data
    logger.info(
        f"Sharding datasets with shard_size={config.shard_size}, repeat={config.pass_at}..."
    )
    shards = shard_input_data(config.datasets, config=config)
    if not shards:
        logger.error("No shards created. Exiting experiment.")
        return
    logger.info(f"Total shards to process: {len(shards)}")

    # Initialize Manager and Queue
    manager = Manager()
    result_queue = manager.Queue()

    # Prepare output directory with timestamp
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        pathlib.Path(config.logging_dir).parent / f"{config.job_id}_{time_stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update logging_dir to include timestamped directory
    config.logging_dir = str(output_dir / "log/proof_search/")
    os.makedirs(config.logging_dir, exist_ok=True)

    # Re-setup logging to include timestamped directory
    setup_logging(config)

    # Start ProcessPoolExecutor
    all_search_results = []

    if debug:
        for shard_id, shard_path in shards:
            process_shard(shard_id, shard_path, config, result_queue, debug=debug)
        exit(0)
    # process_shard(shards[0][0], shards[0][1], config, result_queue)

    logger.info(f"Starting multiprocessing with {config.max_workers} workers...")
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        # Submit all run_prover tasks
        futures = []
        for shard_id, shard_path in shards:
            future = executor.submit(
                process_shard, shard_id, shard_path, config, result_queue
            )
            futures.append(future)

        # Optionally, use as_completed to track progress
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing shards"
        ):
            try:
                future.result()  # To catch any exceptions raised in run_prover
                while not result_queue.empty():
                    res = result_queue.get()
                    logger.info(f"Received result: {res}")
                    all_search_results.append(res)
                    print_status(all_search_results)
            except Exception as e:
                logger.error(f"Exception in run_prover: {e}")
                traceback.print_exc()

    # Print summary
    logger.info("Experiment completed.")
    logger.info(f"Total tasks processed: {len(all_search_results)}")


def main():
    """
    Main entry point for the script.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run gronet solution generation experiment with specified configurations"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="prover/configs/moonshot_config_miniF2F-test.yaml",
        help="Path to the YAML configuration file",
    )

    args = parser.parse_args()

    # Load configuration from YAML
    config = Config.from_yaml(args.config)

    # Run the experiment
    run_experiment(config, debug=False)


if __name__ == "__main__":
    main()
