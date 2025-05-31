import asyncio
import json
import logging
import multiprocessing as mp
import os
import queue
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import aiohttp
import httpx
import pandas as pd
from datasets import load_dataset
from loguru import logger
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from autoformalizer.clients.infotree.process_infotree import extract_data
from autoformalizer.eval_utils import lean_feedback
from autoformalizer.repl_lean_feedback.repl_utils import (
    get_messages_for_lines,
    parse_lean_response,
    split_logprobs,
    split_proof_header,
)
from autoformalizer.repl_lean_feedback.state import State


def get_healthy_consul_services(
    service_name: str,
    raise_exception: bool = False,
) -> list[dict] | None:
    service_query_url = f"https://consul-lan.msh.team/v1/health/checks/{service_name}"
    try:
        response = httpx.get(service_query_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        svcs = []
        seen = set()
        for item in data:
            address = "http://" + ":".join(item["CheckID"].split(":")[-2:])
            status = item["Status"]
            if status == "passing" and address not in seen:
                svcs.append(address)
                seen.add(address)

        return svcs  # noqa: TRY300
    except (httpx.HTTPError, ValueError) as e:
        logger.error(f"Failed to get healthy consul services for {service_name}: {e}")
        if raise_exception:
            raise e  # noqa: TRY201
        return []


class Lean4Client(object):

    def __init__(self, url, api_key=None) -> None:
        self.url = url
        if api_key is None:
            api_key = os.getenv("MOONSHOT_LEAN_API_KEY")
        self.api_key = api_key
        self.contexts = {}
        self.process_id = None
        self.all_address = None

    def refresh_all_address(self):
        if self.url == "https://kimina.saas.moonshot.cn/lean4-evaluator":
            self.all_address = ["https://kimina.saas.moonshot.cn/lean4-evaluator"]
        else:
            service_name = self.url.split("//")[-1].split(".")[0]
            all_adresses = []
            while len(all_adresses) == 0:
                all_adresses = get_healthy_consul_services(service_name)
                time.sleep(0.5)
                logger.error(
                    f"Failed to get healthy consul services for {service_name}"
                )
            logger.info(f"Refreshed lean4 server with {len(all_adresses)} servers")
            self.all_address = all_adresses

    def one_pass_verify(self, code, timeout):
        return asyncio.run(self.async_one_pass_verify(code, timeout))

    def one_pass_verify_batch(self, codes, timeout, infotree_type=None):
        return asyncio.run(
            self.async_one_pass_verify_batch(codes, timeout, infotree_type)
        )

    def init_theorem(self, code, timeout):
        ori_code = code
        code = code.rstrip()
        if code.endswith("by"):
            code = code + " sorry"
        elif code.endswith(":="):
            code = code + " by sorry"
        else:
            raise ValueError(f"Invalid code: {code}")

        codes = [{"code": code, "custom_id": str(uuid.uuid4())}]
        response = self.one_pass_verify_batch(codes, timeout, infotree_type="original")

        # handle the response to get error message and infotree
        try:
            infotree, _ = (
                response["results"][0]["response"]["infotree"],
                response["results"][0]["response"],
            )
        except (KeyError, TypeError):
            logger.error(f"Error in init theorem response: {response}, code: {[code]}")
            return None

        _, context = split_proof_header(code)
        tactics = extract_data(infotree, context)

        # get the last tactic
        tactic = tactics[-1]
        if "sorry" not in tactic["tactic"] and len(tactic["goalsAfter"]) != 0:
            logger.error(
                f"Error in init theorem response: {response}, code: {[code]}, tactics: {tactics}"
            )
            return None

        return State.from_seq_apply(
            messages=None,
            goals=tactic["goalsBefore"],
            parent_state=None,
            last_tactic=None,
            statement=ori_code,
        )

    def apply_seq_tactic(
        self, code_prefix, seq_tactics, logprobs, state=None, timeout=60, indent=2
    ):
        # temporarily add a "skip" tactic at the end of the proof
        full_code = code_prefix + seq_tactics + "\n" + " " * indent + "skip"
        code = [{"code": full_code, "custom_id": str(uuid.uuid4())}]
        response = self.one_pass_verify_batch(code, timeout, infotree_type="original")

        # handle the response to get error message and infotree
        try:
            infotree, lean_response = (
                response["results"][0]["response"]["infotree"],
                response["results"][0]["response"],
            )
        except (KeyError, TypeError):
            logger.error(f"Error in response: {response}")
            return [], False, full_code
        is_valid_no_sorry = not lean_feedback.has_error(
            lean_response, accept_sorry=False
        )
        # logger.debug(f"Result [{is_valid_no_sorry}] Verifing full proof: {[full_code]}")
        _, context = split_proof_header(full_code)
        tactics = extract_data(infotree, context)
        line_num_to_message = parse_lean_response(lean_response)

        # lean error
        if line_num_to_message.get(0, {}).get("severity", "") == "error":
            return [], is_valid_no_sorry, full_code

        # remove the last "skip" tactic added before
        full_code = full_code[: -len("\n" + " " * indent + "skip")]

        # Get the correct initial tactic
        begin_output_tactic_index = 0
        _, split_code_prefix = split_proof_header(code_prefix)
        for idx, tactic in enumerate(tactics):
            tactic_text = tactic["tactic"]
            if tactic_text in split_code_prefix:
                continue

            tactic_lines = tactic["tactic"].split("\n")
            last_index = len(tactic_lines)
            while "\n".join(tactic_lines[:last_index]) not in code_prefix:
                last_index -= 1

            tactics[idx]["tactic"] = "\n" + "\n".join(tactic_lines[last_index:])
            begin_output_tactic_index = idx
            break
        line_num = split_code_prefix.count("\n") + 1

        messages, has_error, is_unsolved_goals = get_messages_for_lines(
            line_num_to_message, 0, line_num
        )
        if has_error and not is_unsolved_goals:
            logger.error(
                f"Error in the prefix code: {[code_prefix]}, seq_tactics: {[seq_tactics]}, messages: {messages}"
            )
            return [], is_valid_no_sorry, full_code

        # add one to line_num to match the first line of tactic
        line_num += 1

        split_logprob = split_logprobs(logprobs, tactics[begin_output_tactic_index:])
        if split_logprob is None:
            return [], is_valid_no_sorry, full_code

        # Get the state's result
        results = []
        last_state = state
        for idx, tactic in enumerate(tactics[begin_output_tactic_index:]):
            tactic_text = tactic["tactic"]
            if (
                "skip" in tactic_text
                and idx == len(tactics[begin_output_tactic_index:]) - 1
            ):
                break
            # Set the current goals to the "goalsBefore" of the next tactic if available.
            # This is more robust then the "goalsAfter" in general
            goals = (
                tactics[idx + 1]["goalsBefore"]
                if idx + 1 < len(tactics)
                else tactic["goalsAfter"]
            )
            messages, has_error, _ = get_messages_for_lines(
                line_num_to_message, line_num, line_num + tactic_text.count("\n") - 1
            )
            new_state = State.from_seq_apply(
                messages=messages,
                goals=goals,
                parent_state=last_state,
                last_tactic=tactic_text,
                statement=code_prefix,
            )
            if new_state.is_solved() and not is_valid_no_sorry:
                logger.error("!!! No goals but invalid!")
                break
            new_state.full_code = full_code
            # new_state.full_code_lean_response = lean_response
            logprob = split_logprob[idx]
            results.append((new_state, logprob))
            last_state = new_state
            line_num += tactic_text.count("\n")

            if has_error:
                break
            if new_state.is_solved():
                break

        return results, is_valid_no_sorry, full_code

    async def async_one_pass_verify(self, code, timeout, url=None):
        """verify the proof code and get result

        Args:
            code (str): The lean 4 code to verify.
            timeout (int): The timeout in seconds.

        Returns:
            response (dict): The response from the server.
                It contains the following keys:
                    - error: A string with the error message from the lean server.
                    - response: A dictionary with the response from the LEAN REPL.

        Example:
            >>> client.one_pass_verify("import Mathlib\n\nexample : 2 = 2 := rfl", timeout=60)
            {"error": null, "response": {"env": 0}}
        """
        json_data = {
            "method": "one_pass_verify",
            "code": code,
            "timeout": timeout,
        }
        response = await self._query(json_data, url=url)
        return response

    async def async_one_pass_verify_batch(self, codes, timeout, infotree_type=None):
        """verify the proof code and get result

        Args:
            codes (list): The list of lena 4 code to verify.
                Each code is a dict of:
                    - code: The lena 4 code to verify.
                    - custom_id: The custom id of the proof.
            timeout (int): The timeout in seconds.

        Returns:
            response (dict): The response from the server.
                It contains a  key results, which is a list of dictionaries.
                Each dictionary contains the following keys:
                    - code: The custom id of the proof.
                    - error: A string with the error message from the lean server.
                    - response: A dictionary with the response from the LEAN REPL.

        Example:
            >>> client.one_pass_verify("import Mathlib\n\nexample : 2 = 2 := rfl", timeout=60)
            {'results': [{'code': 'test_connection', 'error': None, 'response': {'env': 0}}]}
        """
        json_data = {
            "method": "one_pass_verify_batch",
            "codes": codes,
            "timeout": timeout,
            "infotree_type": infotree_type,
        }
        response = await self._query(json_data)
        return response

    async def _query(
        self, json_data: dict, n_retries: int = 100000, url: str = None
    ) -> dict:
        """
        One single method for sending all requests, with retry behavior controlled by the caller.
        :param json_data: The data to send in the request.
        :param n_retries: Number of retry attempts.
        :return: Response as a dict.
        """

        # Create retry decorator with dynamic n_retries
        @retry(
            stop=stop_after_attempt(
                n_retries
            ),  # Dynamic retries based on the caller's argument
            wait=wait_exponential(multiplier=1, min=1, max=10),  # Exponential backoff
            before_sleep=before_sleep_log(
                logger, logging.ERROR
            ),  # Optional logging of each retry
        )
        async def query_with_retries(url):
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            method = json_data["method"]

            if self.all_address is None or len(self.all_address) == 0:
                logger.info("No address, refresh all address")
                self.refresh_all_address()

            if random.random() < 0.0001:
                logger.info("Randomly refresh all address")
                self.refresh_all_address()

            url = random.choice(self.all_address)

            # Create a session with trust_env set to True
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.post(
                    f"{url}/{method}",
                    headers=headers,
                    json=json_data,  # Directly send the JSON data
                ) as response:
                    # Get the response body asynchronously and parse it as JSON
                    res = await response.json()

            return res

        # Call the query function with retries
        return await query_with_retries(url)


def test_connection_batch(client):
    """
    Tests the connection to the server by verifying a simple proof.

    Args:
        client (Lean4Client): The client to use for verification.

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    code = {
        "code": """import Mathlib\n\ndef f := 2\nexample : f = 2 := rfl""",
        "custom_id": "test_connection",
    }
    time_start = time.time()
    res = client.one_pass_verify_batch([code] * 1, timeout=60)
    logger.info(f"Batch test connection in {time.time() - time_start:.2f}: {res}")
    if res.get("results") is not None:
        return True
    else:
        return False


def test_connection(client):
    """
    Tests the connection to the server by verifying a simple proof.

    Args:
        client (Lean4Client): The client to use for verification.

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    code = """import Mathlib\n\ndef f := 2\nexample : f = 2 := rfl"""
    time_start = time.time()
    res = client.one_pass_verify(code, timeout=60)
    logger.info(f"Test connection in {time.time() - time_start:.2f}: {res}")
    if res.get("error") is None and res.get("response") is not None:
        return True
    else:
        return False


def verify_proof(sample, client, timeout, working_dir):
    """
    Verifies the proof contained in the sample using the client and logs detailed information.
    This function is mainly used for parallel processing.

    Args:
        sample (dict): A dictionary containing the proof to be verified.
            sample must contain the following keys:
                - proof: The proof code to verify.
                - uuid: The uuid of the proof.
                - proof_id: The proof id.
            uuid and proof_id are used to save the verification results in a file.
        client (Lean4Client): The client to use for verification.
        timeout (int): The timeout in seconds.



    Returns:
        sample (dict): The updated sample with the following keys:
            - lean_feedback: A JSON string with the response from the server.
            - has_error: A boolean indicating if the proof has an error.
                we use lean_feedback.has_error to determine if the proof has an error.
    """
    start_time = time.time()
    code = sample["proof"]
    uuid_dir = working_dir / sample["uuid"]
    # use dict to avoid too many files under the same directory
    uuid_dir.mkdir(parents=True, exist_ok=True)

    filepath = uuid_dir / f"{sample['proof_id']}.json"

    if filepath.exists():
        try:
            with open(filepath, "r") as f:
                response = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error in file: {filepath}")
            response = {"error": str(e)}
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
    is_valid_no_sorry = (not bool(error_message)) and (
        not lean_feedback.has_error(json_response, accept_sorry=False)
    )

    # Prepare the result
    sample.update(
        {
            "lean_feedback": json.dumps(response),
            "has_error": has_error,
            "is_valid_no_sorry": is_valid_no_sorry,
        }
    )
    return sample


def threaded_verify_proof(client, samples, timeout, num_threads, working_dir):
    """
    Verifies the proofs in the samples in parallel using threads and logs detailed information.

    Using threads instead of multiprocessing can be beneficial when the operations are I/O bound
    rather than CPU bound, such as when making network requests or file operations.

    Args:
        client: The client object used for verification
        samples (list): A list of dictionaries containing the proofs to be verified
        timeout (int): Timeout duration for each verification
        num_threads (int): Number of threads to use for parallel processing
        working_dir (str): Working directory path for temporary files

    Returns:
        list: A list containing server feedback and validity status for each proof
    """
    # Create the partial function with fixed arguments
    func = partial(
        verify_proof, client=client, timeout=timeout, working_dir=working_dir
    )

    # Initialize results list to maintain order
    results = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks and wrap with tqdm for progress tracking
        futures = [executor.submit(func, sample) for sample in samples]

        # Use tqdm to show progress while collecting results
        for future, sample in tqdm(zip(futures, samples), total=len(samples)):
            try:
                # Try to get the result with a timeout
                result = future.result(timeout=timeout + 30)
                results.append(result)
            except TimeoutError:
                # Log the timeout and return the sample with an error message
                logger.warning(f"Proof verification timed out for sample {sample}")
                feedback = {"error": "Timeout occurred during verification"}
                sample.update(
                    {
                        "lean_feedback": json.dumps(feedback),
                        "has_error": True,
                        "is_valid_no_sorry": False,
                    }
                )
            except Exception as e:
                # Catch any other exceptions and log them
                logger.error(
                    f"Proof verification failed for sample {sample} with error: {e}"
                )
                feedback = {"error": str(e)}
                sample.update(
                    {
                        "lean_feedback": json.dumps(feedback),
                        "has_error": True,
                        "is_valid_no_sorry": False,
                    }
                )
                results.append(sample)
    return results


async def verify_one_batch(client, timeout, working_dir, infotree_type, samples):
    """
    Verifies a single batch of proofs using the batch API endpoint.

    Args:
        - client (Lean4Client): The client to use for verification
        - samples (list): A list of samples to verify in one batch
            samples must contain the following keys:
                - proof: The proof code to verify
                - proof_id: The proof id
        - timeout (int): The timeout in seconds for the entire batch
        - working_dir (Path): Working directory for saving verification results

    Returns:
        - list: List of processed samples with verification results
    """
    # Prepare batch input format
    batch_codes = []
    for sample in samples:
        batch_codes.append(
            {"code": sample["proof"], "custom_id": str(sample["proof_id"])}
        )
    id_to_sample = {str(sample["proof_id"]): sample for sample in samples}

    # Perform batch verification
    start_time = time.time()
    if working_dir:
        # Create directory structure for results
        for sample in samples:
            uuid_dir = working_dir / sample["uuid"]
            uuid_dir.mkdir(parents=True, exist_ok=True)

        # check if all file path exist
        all_file_path = [
            working_dir / sample["uuid"] / f"{sample['proof_id']}.json"
            for sample in samples
        ]
    if working_dir and all([file_path.exists() for file_path in all_file_path]):
        # all files exist, load reponse
        response = []
        for file_path in all_file_path:
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {"error": "JSONDecodeError"}
                if "code" not in data:
                    data["code"] = file_path.stem
                response.append(data)
    else:
        response = await client.async_one_pass_verify_batch(
            batch_codes, timeout=timeout, infotree_type=infotree_type
        )
        response = response.get("results", [])
    elapsed_time = time.time() - start_time

    # Process results
    if len(response) != len(samples):
        logger.error(
            f"""Time: {elapsed_time:.2f}s, Batch verification failed:
            Expected {len(samples)} results, got {len(response)}"""
        )
        # return samples with default values
        results = []
        for sample in samples:
            new_sample = sample.copy()
            new_sample.update(
                {
                    "lean_feedback": json.dumps(
                        {"error": "Missing response from server"}
                    ),
                    "has_error": True,
                    "is_valid_no_sorry": False,
                    "is_valid_with_sorry": False,
                }
            )
            results.append(new_sample)
        return results

    results = []
    for individual_response in response:
        if individual_response.get("error") is not None:
            # ignore timeout error
            if "Lean process timed out" not in individual_response["error"]:
                logger.warning(
                    f"Time: {elapsed_time:.2f}s, Sample in batch failed: {individual_response['error']}"
                )
        elif individual_response.get("response") is None:
            logger.warning(
                f"Time: {elapsed_time:.2f}s, Sample in batch failed: No response found"
            )

        # this should be custom_id, but I haven't fix it in the server.
        proof_id = individual_response["code"]
        sample = id_to_sample.get(proof_id).copy()
        filepath = (
            working_dir / sample["uuid"] / f"{proof_id}.json" if working_dir else None
        )
        if working_dir and (not filepath.exists()):
            with open(filepath, "w") as f:
                json.dump(individual_response, f)

        output = lean_feedback.parse_client_response(individual_response)
        sample.update(
            {
                "lean_feedback": json.dumps(individual_response),
                "has_error": output["has_error"],
                "is_valid_no_sorry": output["is_valid_no_sorry"],
                "is_valid_with_sorry": output["is_valid_with_sorry"],
            }
        )
        results.append(sample)
        # Log occasional results for monitoring
        if random.random() < 0.0001:
            logger.info(
                f"Time: {elapsed_time:.2f}s, Verification response for {proof_id}: {individual_response}"
            )

    return results


async def thread_worker(
    batch_queue, result_queue, client, timeout, working_dir, infotree_type
):
    while not batch_queue.empty():
        try:
            batch = batch_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        # Assuming verify_one_batch is an async function
        try:
            results = await verify_one_batch(
                client, timeout, working_dir, infotree_type, batch
            )
        except Exception as e:
            logger.error(
                f"Batch verification failed for batch {str(batch)[:50]} with error: {e}"
            )
            results = []
            for sample in batch:
                new_sample = sample.copy()
                new_sample.update(
                    {
                        "lean_feedback": json.dumps({"error": str(e)}),
                        "has_error": True,
                        "is_valid_no_sorry": False,
                        "is_valid_with_sorry": False,
                    }
                )
                results.append(new_sample)
        result_queue.put(results)


async def async_process_worker(
    batch_queue,
    result_queue,
    client,
    timeout,
    working_dir,
    num_thread=1000,
    infotree_type=None,
):
    tasks = [
        asyncio.create_task(
            thread_worker(
                batch_queue, result_queue, client, timeout, working_dir, infotree_type
            )
        )
        for _ in range(num_thread)
    ]
    await asyncio.gather(*tasks)


# Wrapper for asyncio event loop
def process_worker(
    batch_queue, result_queue, timeout, working_dir, num_thread, infotree_type, address
):
    client = Lean4Client(url=address)
    asyncio.run(
        async_process_worker(
            batch_queue,
            result_queue,
            client,
            timeout,
            working_dir,
            num_thread,
            infotree_type,
        )
    )


def batch_verify_proof(
    client,
    samples,
    timeout=60,
    num_threads=5,
    batch_size=100,
    working_dir=None,
    infotree_type=None,
):
    """
    Verifies multiple proofs using batched parallel processing.

    Args:
        - client (Lean4Client): The client to use for verification
        - samples (list): List of samples to verify.
            Must contain the following keys:
                - proof_id: The proof id
                - uuid: The uuid of the problem
                - proof: The formal proof to verify
        - timeout (int): Timeout in seconds for each batch
        - num_proc (int): Number of parallel processes to use
        - batch_size (int): Number of proofs to include in each batch
        - working_dir (Path): Working directory for saving verification results
            if None, no cache is applied

    Returns:
        - list: List of all processed samples with verification results
            Each sample contains the following:
                - proof_id: The proof id
                - uuid: The uuid of the problem
                - proof: The formal proof
                - lean_feedback: A JSON string with the response from the server
                - has_error: A boolean indicating if the proof has an error
                - is_valid_no_sorry: A boolean indicating if the proof is valid
                    without any "sorry" statements.
                    This is the output from lean_feedback.parse_client_response.
                - is_valid_with_sorry: A boolean indicating if the proof is valid
                    with "sorry" statements. Use to check formal_statement.
                    This is the output from lean_feedback.parse_client_response.

    """
    # Split samples into batches
    batches = [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]

    # assert proof id is unique
    proof_ids = [sample["proof_id"] for sample in samples]
    assert len(proof_ids) == len(set(proof_ids)), "Proof id must be unique"

    logger.info(
        f"Processing {len(samples)} samples in {len(batches)} batches of size {batch_size}"
    )

    # Create multiprocessing queue
    manager = mp.Manager()
    batch_queue = manager.Queue()
    result_queue = manager.Queue()
    [batch_queue.put(batch) for batch in batches]

    # Process batches in parallel
    all_results = []
    if client.url == "https://kimina.saas.moonshot.cn/lean4-evaluator":
        all_address = ["https://kimina.saas.moonshot.cn/lean4-evaluator"]
    else:
        service_name = client.url.split("//")[-1].split(".")[0]
        all_address = get_healthy_consul_services(service_name)
    logger.info(f"Number of lean server: {len(all_address)}")
    with mp.Pool(processes=len(all_address)) as pool:
        # Here we create a list of jobs, each corresponding to a batch of verification
        func = partial(
            process_worker,
            batch_queue,
            result_queue,
            timeout,
            working_dir,
            num_threads,
            infotree_type,
        )

        # pool.map(func, range(num_proc))
        async_results = [pool.map_async(func, all_address)]

        tqdm_bar = tqdm(total=len(batches), desc="Verifying proofs")
        # Gather the results from the result_queue
        while True:
            time.sleep(1)
            if not any(result.ready() is False for result in async_results):
                break
            while not result_queue.empty():
                all_results.extend(result_queue.get())
                # update tqdm bar to len(all_results)
                tqdm_bar.update(1)

        # Gater the remaining results
        while not result_queue.empty():
            all_results.extend(result_queue.get())
            # update tqdm bar to len(all_results)
            tqdm_bar.update(1)

    del batch_queue
    del result_queue
    manager.shutdown()
    tqdm_bar.close()

    return all_results


def example_batch(client):
    dataset = load_dataset("AI-MO/math-test-inference-results", split="train")
    working_dir = None
    # select the first 500 uuids
    dataset = dataset.select(range(100 * 64))
    print(dataset)

    # make sure we have proof_id, uuid, and proof
    new_samples = []
    for sample in tqdm(dataset):
        new_samples.append(
            {
                # for this dataset, proof_id is not unique, unfortunately
                "proof_id": sample["uuid"] + "_" + str(sample["proof_id"]),
                "uuid": sample["uuid"],
                "proof": sample["proof"],
            }
        )

    # create batch and use one_pass_verify_batch
    batch_size = 1
    results = batch_verify_proof(
        client=client,
        samples=new_samples,
        timeout=60,
        num_threads=250,
        batch_size=batch_size,
        working_dir=working_dir,
    )
    print(len(results))
    print(results[0])

    # calculate valid rate and other stats
    df = []
    for data in results:
        uuid = data["uuid"]
        name = data["proof_id"]
        response = json.loads(data["lean_feedback"])
        error_message = response.get("error", None)
        df.append(
            {
                "uuid": uuid,
                "is_valid_no_sorry": data["is_valid_no_sorry"],
                "name": name,
                "has_connection_error": bool(error_message),
            }
        )
    df = pd.DataFrame(df)
    print(df)

    # calculate valid rate
    valid_rate = df["is_valid_no_sorry"].sum() / len(df)
    print(f"valid rate: {valid_rate}")

    # connection error rate
    connection_error_rate = df["has_connection_error"].sum() / len(df)
    print(f"connection error rate: {connection_error_rate}")

    # calculate valid rate for each uuid using groupby
    uuid_group = df.groupby("uuid")["is_valid_no_sorry"].sum()

    # find all uuids with at least one valid proof
    valid_uuids = uuid_group[uuid_group > 0].index
    print(f"Number of uuids: {len(uuid_group)}")
    print(f"Number of uuids with at least one valid proof: {len(valid_uuids)}")


if __name__ == "__main__":
    """
    Example batch should display something like

    ==========================Output==========================
        valid rate: 0.32890625
        connection error rate: 0.0
        Number of uuids: 100
        Number of uuids with at least one valid proof: 52
    """
    # client = Lean4Client(
    #     "https://kimina.saas.moonshot.cn/lean4-evaluator",
    # )
    # client = Lean4Client(url="https://kimina.saas.moonshot.cn/lean4-evaluator")
    client = Lean4Client(url="http://lean4-evaluator-internal.app.msh.team/")
    if test_connection(client):
        logger.info("Connection successful!")
    else:
        logger.error("Connection failed!")

    if test_connection_batch(client):
        logger.info("Connection successful!")
    else:
        logger.error("Connection failed!")

    example_batch(client)
