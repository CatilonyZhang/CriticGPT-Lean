import json
import multiprocessing as mp
import os
import pathlib
import random
import sys
import time
from functools import partial

import pandas as pd
import requests
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from autoformalizer.eval_utils import lean_feedback


class Lean4Client(object):

    def __init__(self, url, api_key) -> None:
        self.url = url
        self.api_key = api_key
        self.contexts = {}
        self.process_id = None

    def one_pass_verify(self, code, timeout):
        """verify the proof code and get result"""
        json_data = {
            "method": "one_pass_verify",
            "code": code,
            "timeout": timeout,
        }
        return self._query(json_data, n_retries=3)

    def _query(
        self, json_data: dict, n_retries: int = 0, retry_on_timeout=False
    ) -> dict:
        """One single method for sending all requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        method = json_data["method"]
        for attemp_i in range(n_retries + 1):
            try:
                response = requests.post(
                    f"{self.url}/{method}", headers=headers, json=json_data
                )
                res = json.loads(response.text)
                if res.get("error") == "Lean process timed out" and retry_on_timeout:
                    time.sleep(0.1)
                    continue
                break
            except json.decoder.JSONDecodeError:
                res = {"error": f"JSONDecodeError with text: {response.text}"}
                if attemp_i < n_retries:
                    time.sleep(0.1 * attemp_i**2)
            except requests.exceptions.Timeout:
                res = {"error": "Server Inner TimeoutError"}
                if attemp_i < n_retries:
                    time.sleep(0.1)
            except requests.exceptions.ConnectionError:
                res = {"error": "Connection Error"}
                if attemp_i < n_retries:
                    time.sleep(0.1)
        return res


def test_connection(client):
    """
    Tests the connection to the server by verifying a simple proof.

    Args:
        client (Lean4Client): The client to use for verification.

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    code = """import Mathlib\n\ndef f := 2\nexample : f = 2 := rfl"""
    res = client.one_pass_verify(code, timeout=60)
    logger.info(f"Test connection: {res}")
    if res.get("error") is None and res.get("response") is not None:
        return True
    else:
        return False


def verify_proof(client, timeout, working_dir, sample):
    """
    Verifies the proof contained in the sample using the client and logs detailed information.

    Args:
        sample (dict): A dictionary containing the proof to be verified.

    Returns:
        dict: A dictionary containing server feedback and validity status.
    """
    start_time = time.time()
    code = sample["proof"]
    filepath = working_dir / f"{sample['uuid']}_{sample['proof_id']}.json"

    if filepath.exists():
        try:
            with open(filepath, "r") as f:
                response = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error in file: {filepath}")
            raise e
    else:
        # Perform verification with timeout
        response = client.one_pass_verify(code, timeout=timeout)
        with open(filepath, "w") as f:
            json.dump(response, f)
    elapsed_time = time.time() - start_time

    # Log errors and response details
    error_message = response.get("error", None)
    if error_message is not None:
        logger.error(
            f"Time: {elapsed_time:.2f}s, Proof verification failed: {error_message}"
        )

    json_response = response.get("response", None)
    if error_message is None and json_response is None:
        logger.error(f"Time: {elapsed_time:.2f}s, Missing response: {response}")
    elif "error decoding json response in leanrepl" in str(response):
        logger.error(
            f"Time: {elapsed_time:.2f}s, JSON decoding error in LeanRepl: {response}"
        )

    # Log response details occasionally
    if random.random() < 0.001:
        logger.info(
            f"Time: {elapsed_time:.2f}s, Verification response: {response}, Proof: {[code]}"
        )

    # Determine if there are any errors
    has_error = bool(error_message) or lean_feedback.has_error(json_response)
    is_valid_server = (not has_error) and ("sorry" not in code)

    # Prepare the result
    sample.update(
        {
            "server_feedback": json.dumps(response),
            "is_valid_server": is_valid_server,
        }
    )
    return sample


def parallel_verify_proof(client, samples, timeout, num_proc, working_dir):
    """
    Verifies the proofs in the samples in parallel using the client and logs detailed information.

    Args:
        samples (list): A list of dictionaries containing the proofs to be verified.

    Returns:
        dict: A dictionary containing server feedback and validity status for each proof.
    """
    # Perform parallel verification
    func = partial(verify_proof, client, timeout, working_dir)

    with mp.Pool(processes=num_proc) as pool:
        results = list(tqdm(pool.imap_unordered(func, samples), total=len(samples)))
    return results


def save_chunks_to_parquet(data, output_dir, chunk_size=1000):
    """
    Save a list of dictionaries into Parquet files in chunks.

    Parameters:
        data (list): List of dictionaries to save.
        output_dir (str): Directory to save the Parquet files.
        chunk_size (int): Number of elements per chunk (default is 1000).

    Returns:
        list: List of file paths for the saved Parquet files.
    """
    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        raise ValueError("Input data must be a list of dictionaries.")

    # Ensure the output directory exists
    output_dir_path = pathlib.Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    file_paths = []

    # Split the data into chunks and save each chunk as a Parquet file
    for i in tqdm(range(0, len(data), chunk_size)):
        chunk = data[i : i + chunk_size]
        df = pd.DataFrame(chunk)
        file_name = output_dir_path / f"chunk_{i // chunk_size}.parquet"
        df.to_parquet(file_name, index=False)
        file_paths.append(str(file_name))

    return file_paths


if __name__ == "__main__":
    client = Lean4Client(
        "https://kimina.saas.moonshot.cn/lean4-evaluator",
        api_key=os.environ.get("MOONSHOT_LEAN4_API_KEY"),
    )

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    dataset = load_dataset(
        "AI-MO/formalproblemsv0-inference-deepseek-prover-v1.5-rl", split="train"
    )
    n_proofs = 128

    # Test connection
    if not test_connection(client):
        logger.error("Connection test failed. Exiting...")
        sys.exit(1)

    batch_size = 500
    working_dir = pathlib.Path("./scripts/jia/formal_problems_v0")
    result_dir = pathlib.Path("./scripts/jia/formal_problems_v0_results")
    for offset in range(0, len(dataset), batch_size):
        # for offset in range(0, 1000, batch_size):
        # for problem_dict in tqdm(dataset.select(range(1000))):
        # for problem_dict in tqdm(dataset):
        samples = []
        logger.info(f"Processing problems {offset} to {offset + batch_size}")
        end_batch = min(offset + batch_size, len(dataset))
        for problem_dict in tqdm(dataset.select(range(offset, end_batch))):
            proof_ids = list(range(n_proofs))
            for _id in proof_ids:
                sample = {}
                sample["proof_id"] = _id
                sample["uuid"] = f"temporal_id_{problem_dict['problem_id']}"
                sample["problem_id"] = problem_dict["problem_id"]
                sample["proof"] = problem_dict[f"proof_{_id}"]
                samples.append(sample)

        # Verify proofs in parallel
        timeout = 60
        num_proc = 100
        results = parallel_verify_proof(client, samples, timeout, num_proc, working_dir)

        # Log statistics
        total_problems = len(samples)
        valid_problems = sum(result["is_valid_server"] for result in results)
        logger.info(f"Total unique proofs: {total_problems}")
        logger.info(f"Valid unique proofs: {valid_problems}")

        # save_chunks_to_parquet(
        #     aggregate_results, "./scripts/jia/formal_problems_v0_results", chunk_size=100000
        # )
        # sort the results by uuid and proof_id
        results = sorted(results, key=lambda x: (x["uuid"], x["proof_id"]))
        df = pd.DataFrame(results)
        df.to_parquet(result_dir / f"chunk_{offset // batch_size}.parquet", index=False)

    ds = load_dataset(
        "parquet", data_files="./scripts/jia/formal_problems_v0_results/chunk_*.parquet"
    )
    ds.push_to_hub("AI-MO/formal-problems-v0-proofs", private=True)
    ds = ds["train"]
    valid_ds = ds.filter(lambda x: x["is_valid_server"])
    # log the number of valid problems
    logger.info(f"Total problems: {len(set(ds['uuid']))}")
    logger.info(f"Valid problems: {len(set(valid_ds['uuid']))}")
    logger.info("Results pushed to the hub.")
