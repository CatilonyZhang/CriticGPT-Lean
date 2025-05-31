import concurrent.futures
import json
import logging
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List
from tqdm import tqdm

import fire
import pandas as pd
from datasets import load_dataset

def has_error(
    feedback: dict, accept_sorry: bool = True, return_error_messages: bool = False
):
    """
    Checks if the Lean feedback contains an error.

    Args:
    - feedback: The Lean feedback as a dictionary.
    - accept_sorry: Whether to accept "sorry" statements as "not an error".
    By default, "sorry" statements are not considered errors.
    """

    if "error" in feedback:
        r = (True, [feedback["error"]]) if return_error_messages else True
        return r

    if "stderr" in feedback:
        r = (True, [feedback["stderr"]]) if return_error_messages else True
        return r

    has_error = False
    error_data_values = []
    sorry_data_values = []
    if "messages" in feedback:
        error_data_values = [
            message["data"]
            for message in feedback.get("messages", [])
            if message.get("severity") == "error"
        ]
        has_error = bool(error_data_values)

        if not accept_sorry:
            warning_data_values = [
                message["data"]
                for message in feedback.get("messages", [])
                if message.get("severity") == "warning"
            ]
            sorry_data_values = [
                warning_data
                for warning_data in warning_data_values
                if "declaration uses 'sorry'" in warning_data
            ]
            has_error = has_error or bool(sorry_data_values)

    if return_error_messages:
        return has_error, error_data_values + sorry_data_values
    else:
        return has_error


def parse_client_response(response: dict):
    """Parses the response from the Lean4Client.
    Reponse should be the output of autoformalizer.clients.lean4_client.Lean4Client.one_pass_verify

    Args:
        - response (dict): The response from the Lean4Client.

    Returns:
        - dict: A dictionary containing the following keys:
            - has_error: Whether the response contains an error.
            - is_valid_no_sorry: Whether the response is valid without "sorry" statements.
                this is used for proof verification.
            - is_valid_with_sorry: Whether the response is valid with "sorry.
                this is used for statement verification.
    """
    error_message = response.get("error", None)
    json_response = response.get("response", None)

    error = bool(error_message) or has_error(json_response)
    is_valid_no_sorry = (not bool(error_message)) and (
        not has_error(json_response, accept_sorry=False)
    )
    is_valid_with_sorry = (not bool(error_message)) and (
        not has_error(json_response, accept_sorry=True)
    )

    return {
        "has_error": error,
        "is_valid_no_sorry": is_valid_no_sorry,
        "is_valid_with_sorry": is_valid_with_sorry,
    }


def lean4_feedback(
    lean_code: str,
    header: str = None,
    max_retries: int = 1,
    timeout: int = 60,
    memory_limit: int = 32,
    verbose: bool = False,
):
    """
    Gets Lean 4 feedback for a given Lean code.

    Args:
    - lean_code: The Lean code to get feedback for.
    - header: The header to prepend to the Lean code.
    - max_retries: The maximum number of retries.
    - timeout: The timeout for the Lean feedback command.
    - memory_limit: The memory limit for the Lean feedback command in GB.
    - verbose: Whether to print the feedback.
    """
    from autoformalizer.eval_utils.constants import path_to_mathlib, path_to_repl
    directory = path_to_mathlib + "/tmp/"
    Path(directory).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=directory, mode="w", suffix=".in", delete=True
    ) as in_file, tempfile.NamedTemporaryFile(
        dir=directory, mode="w", suffix=".lean", delete=True
    ) as input_file:

        file_string = f"""{{"path": "{input_file.name}", "allTactics": true}}"""
        in_file.write(file_string)
        in_file.flush()

        if header is not None:
            formalization = header + lean_code
        else:
            formalization = lean_code

        input_file.write(formalization)
        input_file.flush()

        current_env = os.environ.copy()

        mem_limit_bytes = memory_limit * 1024 * 1024  # Convert GB to KB

        command = f"cd {path_to_mathlib} && ulimit -v {mem_limit_bytes} && lake env {path_to_repl} < {in_file.name}"
        data = {"error": "unknown lean4 error"}

        for attempt in range(max_retries):
            try:
                if verbose:
                    logging.info(f"Attempt {attempt + 1} to run command: {command}")

                with subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,
                    text=True,
                    encoding="utf-8",
                    env=current_env,
                ) as process:
                    try:
                        stdout, stderr = process.communicate(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        # Ensure all relevant processes are termination
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        stdout, stderr = (
                            process.communicate()
                        )  # Read any remaining output
                        process.wait()
                        logging.error(
                            f"Lean feedback command timed out on attempt {attempt + 1}."
                        )
                        continue  # Retry if within max_retries

                    if process.returncode != 0:
                        message = f"Command failed with return code {process.returncode} on attempt {attempt + 1}"
                        logging.error(message + f": {stderr.strip()}")
                        if attempt < max_retries - 1:
                            time.sleep(0.1)
                            continue  # Retry
                        else:
                            return {
                                "error": "command_failed",
                                "details": stderr.strip(),
                            }

                    feedback = stdout
                    data = json.loads(feedback)
                    if len(stderr.strip()):
                        # This covers the case {'message': "unknown metavariable '?[anonymous]'"}
                        # in which the REPL outputs a bunch of messages to stderr but still returns a valid JSON
                        # logging.error(f"Command returned stderr on attempt {attempt + 1}: {stderr.strip()}")
                        data["stderr"] = stderr.strip()

                    if verbose:
                        logging.info(f"Command executed successfully: {feedback[:20]}")
                    break  # Success, exit retry loop

            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding error on attempt {attempt + 1}: {e}")
                logging.error(
                    f"Faulty JSON content: {stdout.strip() if 'stdout' in locals() else 'No output'}"
                )
                return {"error": "json_decode_error"}
            except Exception as e:
                logging.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.1)
                else:
                    return {"error": "unexpected_error", "details": str(e)}

    return data


def parallel_lean4_feedback(
    lean_codes: List[str],
    headers: List[str] = None,
    num_workers: int = 30,
    max_retries: int = 1,
    timeout: int = 60,
    memory_limit: int = 32,
):
    """
    Parallelizes getting the lean feedback across multiple CPUs using concurrent.futures.ThreadPoolExecutor.

    Args:
    - lean_codes: A list of Lean 4 codes to get feedback for.
    - headers: A list of headers to prepend to the Lean codes.
    - num_workers: The number of workers to use.
    - max_retries: The maximum number of retries.
    - timeout: The timeout for the Lean feedback command.
    - memory_limit: The memory limit for the Lean feedback command in GB.
    """

    # Create a helper function that combines the code and header and calls lean4_feedback
    def feedback_helper(args):
        code, header = args
        return lean4_feedback(
            lean_code=code,
            header=header,
            max_retries=max_retries,
            timeout=timeout,
            memory_limit=memory_limit,
        )

    # If headers are provided, zip lean_codes and headers together, else zip lean_codes with None
    if headers is None:
        inputs = [(code, None) for code in lean_codes]
    else:
        inputs = zip(lean_codes, headers)

    # Start timing
    start_time = time.time()

    # Use ThreadPoolExecutor with a progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Wrap the executor.map in tqdm for a progress bar
        results = list(tqdm(executor.map(feedback_helper, inputs), 
                            total=len(lean_codes), 
                            desc="Processing Lean 4 Codes"))

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    # Print the total time taken
    logging.info(f"Total time taken: {total_time:.2f} seconds")

    return results


def dataframe_parallel_lean4_feedback(
    input_csv: str, output_csv: str, accept_sorry: bool = True, verbose: bool = True
):
    """
    Gets Lean 4 feedback for each formalization in a given DataFrame in parallel.

    Args:
    - input_csv: Path to the input CSV file.
    - output_csv: Path to the output CSV file.
    - accept_sorry: Whether to accept "sorry" statements as "not an error".
    - verbose: Whether to print the results.

    Requirements:
    The input CSV file must contain the following columns:
    - 'autoformalization_i': The i-th autoformalization to get feedback for.

    The output CSV file will contain the following columns:
    - 'compiler_feedback_i': The feedback for the i-th autoformalization.
    - 'compiler_feedback_i_bool': The status of the i-th autoformalization. (True/False)
    """

    df = pd.read_csv(input_csv)

    for column in df.columns:
        if str(column).startswith("autoformalization_"):
            idx = column.split("_")[-1]
            lean4_codes = df[column].tolist()
            feedbacks = parallel_lean4_feedback(lean4_codes)

            df[f"compiler_feedback_{idx}"] = [str(feedback) for feedback in feedbacks]
            df[f"compiler_feedback_{idx}_bool"] = [
                not has_error(feedback, accept_sorry=accept_sorry)
                for feedback in feedbacks
            ]
            # df[f"compiler_feedback_{idx}_bool"] = ~df[f"compiler_feedback_{idx}"].str.contains("error")

            if verbose:
                val_counts = df[f"compiler_feedback_{idx}_bool"].value_counts()
                print(f"Feedback for autoformalization_{idx}:")
                print(val_counts)
                print(val_counts / val_counts.sum())

    df.to_csv(output_csv, index=False)


def hf_lean4_feedback(
    input_dataset_id: str,
    input_dataset_branch: str,
    output_dataset_id: str,
    output_dataset_branch: str,
    column_startswith: str = "autoformalization_",
    accept_sorry: bool = True,
    verbose: bool = True,
):
    """
    Gathers Lean 4 compiler feedback for each formalization in a given HuggingFace Dataset.

    Args:
    - input_dataset_id: The ID of the input HuggingFace Dataset.
    - input_dataset_branch: The branch of the input HuggingFace Dataset.
    - output_dataset_id: The ID of the output HuggingFace Dataset.
    - output_dataset_branch: The branch of the output HuggingFace Dataset.
    - column_startswith: The prefix of the columns to get feedback for.
    - accept_sorry: Whether to accept "sorry" statements as "not an error".
    - verbose: Whether to print results.

    Requirements:
    The input dataset must contain the following columns:
    - 'natural_language': The natural language statement to formalize.
    - '{column_startswith}_i': The i-th autoformalization to get feedback for.

    The output dataset file will contain the following columns:
    - 'compiler_feedback_i': The feedback for the i-th autoformalization.
    - 'compiler_feedback_i_status': The status of the i-th autoformalization. (True/False)

    The output dataset will be pushed to the HuggingFace Hub.
    """

    # Load the dataset
    ds = load_dataset(input_dataset_id, split="train", revision=input_dataset_branch)

    autof_columns = [
        column
        for column in ds.column_names
        if str(column).startswith(column_startswith)
    ]

    for column in autof_columns:

        if "_" in column:
            idx = column.split("_")[-1]
            # hack to skip columns that are not proof_{number}
            if not idx.isnumeric():
                continue
        else:
            idx = column

        if (
            f"compiler_feedback_{idx}" in ds.column_names
            or f"compiler_feedback_{idx}_bool" in ds.column_names
        ):
            print(f"Skipping {column} as it already has compiler feedback.")
            continue

        lean4_codes = ds[column]

        feedbacks = parallel_lean4_feedback(lean4_codes)
        feedback_bools = [
            not has_error(feedback, accept_sorry=accept_sorry) for feedback in feedbacks
        ]
        feedbacks = [str(feedback) for feedback in feedbacks]

        if verbose:
            # Print the percentage of correct formalizations
            true_count = sum(feedback_bools)
            print(
                f"Compiler Feedback success rate for {column}: {true_count/len(feedback_bools):.3%}."
            )

        if f"compiler_feedback_{idx}" in ds.column_names:
            ds = ds.remove_columns(f"compiler_feedback_{idx}")

        if f"compiler_feedback_{idx}_bool" in ds.column_names:
            ds = ds.remove_columns(f"compiler_feedback_{idx}_bool")

        ds = ds.add_column(f"compiler_feedback_{idx}", feedbacks)
        ds = ds.add_column(f"compiler_feedback_{idx}_bool", feedback_bools)

        # push every time to avoid losing data
        ds.push_to_hub(
            output_dataset_id,
            revision=output_dataset_branch,
            private=True,
            commit_message="Added Lean 4 compiler feedback for autoformalizations.",
        )


if __name__ == "__main__":
    """
    Example usage:
    python -m autoformalizer.eval_utils.lean_feedback \
    dataframe_parallel_lean4_feedback \
    --input_csv="/home/mert/autoformalizer/scripts/mert/Autof181024.csv" \
    --output_csv="/home/mert/autoformalizer/scripts/mert/Autof181024_feedback.csv"

    python -m autoformalizer.eval_utils.lean_feedback \
    hf_lean4_feedback \
    --input_dataset_id="AI-MO/AutoformalizationEvalV2" \
    --input_dataset_branch="Qwen7BCoder_AutoformalizerV1B3" \
    --output_dataset_id="AI-MO/AutoformalizationEvalV2" \
    --output_dataset_branch="Qwen7BCoder_AutoformalizerV1B3"

    python -m autoformalizer.eval_utils.lean_feedback \
    hf_lean4_feedback \
    --input_dataset_id="AI-MO/wholeproof-data-nomathlib-241218" \
    --input_dataset_branch="main" \
    --output_dataset_id="AI-MO/wholeproof-data-nomathlib-241218" \
    --output_dataset_branch="main" \
    --column_startswith="proof_"
    """

    fire.Fire(
        {
            "dataframe_parallel_lean4_feedback": dataframe_parallel_lean4_feedback,
            "hf_lean4_feedback": hf_lean4_feedback,
        }
    )
