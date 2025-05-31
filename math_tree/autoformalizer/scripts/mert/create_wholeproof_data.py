import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Union

import pandas as pd
import tqdm
from datasets import Dataset, load_dataset

VERBOSE = False
NUM_PROC = mp.cpu_count()


def save_to_parquet_with_parallelism(
    data_dict, base_path: str, chunk_size: int = 10000
):
    """
    Saves the data_dict into Parquet files in parallel, splitting the data into chunks.

    Args:
        data_dict (dict): The data dictionary where keys are column names and values are lists of data.
        base_path (str): The directory path where the Parquet shards will be saved.
        chunk_size (int): The size of each chunk (shard). Default is 10,000.
    """

    total_size = len(
        data_dict[next(iter(data_dict))]
    )  # Get the size of the data (using the first column)
    num_shards = math.ceil(total_size / chunk_size)  # Calculate how many shards we need

    # Ensure the directory exists
    os.makedirs(base_path, exist_ok=True)

    # Helper function to save a single shard as Parquet
    def save_shard(shard_idx: int):
        start_idx = shard_idx * chunk_size
        end_idx = min((shard_idx + 1) * chunk_size, total_size)

        # Extract the chunk for the current shard
        shard_data_dict = {
            key: value[start_idx:end_idx] for key, value in data_dict.items()
        }

        # Convert to pandas DataFrame
        df = pd.DataFrame(shard_data_dict)

        # Define the filename for the shard
        shard_filename = os.path.join(base_path, f"shard_{shard_idx:05}.parquet")

        # Save the shard as Parquet file
        df.to_parquet(shard_filename, engine="pyarrow", index=False)
        print(f"Saved shard {shard_idx} to {shard_filename}")

    # Create a ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=NUM_PROC) as executor:
        # Map the function to the number of shards we need
        executor.map(save_shard, range(num_shards))

    print("All shards have been saved successfully!")


def parallel_extract_data(dataset, column_name, num_proc=NUM_PROC):
    """
    Extracts data from a dataset in parallel and keeps the order.

    Args:
        dataset (Dataset): The dataset to extract data from.
        column_name (str): The column name to extract.
        num_proc (int): The number of processes to use. Default is 128.

    Returns:
        List: A list containing the data extracted from the specified column.
    """

    # Function to extract the data for a given chunk (this keeps the logic isolated)
    def extract_chunk(start_idx, end_idx):
        # Simulating extraction from the dataset column (adjust depending on your dataset type)
        return [dataset[i][column_name] for i in range(start_idx, end_idx)]

    # Total number of items in the dataset
    total_size = len(dataset)

    # Create a list to store the result in the correct order
    column_data = [None] * total_size

    # Divide the work into chunks
    chunk_size = (total_size + num_proc - 1) // num_proc  # this ensures rounding up

    # We will use a ProcessPoolExecutor to parallelize
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        # We create future objects for each chunk and map them to their result
        futures = []
        for i in range(0, total_size, chunk_size):
            start_idx = i
            end_idx = min(i + chunk_size, total_size)
            futures.append(executor.submit(extract_chunk, start_idx, end_idx))

        # Collect results from futures, maintaining order
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Extracting data"
        ):
            chunk_result = future.result()
            # Identify the correct place to insert the results in the ordered list
            start_idx = futures[futures.index(future)].args[
                0
            ]  # Retrieve the starting index
            for i, val in enumerate(chunk_result):
                column_data[start_idx + i] = val

    return column_data


def is_in_statement(line: str):
    line = line.strip()
    in_statement = line.startswith("theorem")
    in_statement = in_statement or line.startswith("lemma")
    in_statement = in_statement or line.startswith("/- determine -/ abbrev")

    return in_statement


def _get_index_of_theorem_statement(proof_context: str):
    index = 0
    ends = []
    in_statement = False
    for line in proof_context.split("\n"):
        if is_in_statement(line):
            in_statement = True
        if in_statement:
            if ":= by" in line:
                end_index = index + line.index(":= by") + len(":= by")
                ends.append(end_index)
                in_statement = False
            elif ":=by" in line:
                end_index = index + line.index(":=by") + len(":=by")
                ends.append(end_index)
                in_statement = False
            elif ":=" in line and ":= by" not in line and ":=by" not in line:
                end_index = index + line.index(":=") + len(":=")
                ends.append(end_index)
                in_statement = False
        index += len(line) + 1
    if len(ends) > 0:
        # Get the last match
        return ends[-1]
    return -1


def auto_statements_dataset_handler(dataset: Dataset):
    """
    Handler for the auto-statements dataset.

    Args:
        dataset (Dataset): The dataset.
    """

    def process_example(example):
        formal_statement = example["formal_statement"].strip()[: -len("sorry")].strip()
        proof_context = example["proof"]

        if len(proof_context.strip()) == 0:
            example["proof_input"] = None
            example["proof_output"] = None
            example["proof"] = None
            return example

        # process the proof context
        if len(formal_statement) > 0 and formal_statement in proof_context:
            statement_index = proof_context.index(formal_statement) + len(
                formal_statement
            )
        else:
            statement_index = _get_index_of_theorem_statement(proof_context)

        if statement_index == -1:
            if VERBOSE:
                print(
                    "\n\n=========No match found for proof context. example: ",
                    example["formal_proof"],
                    "\n\n\n",
                )
            example["proof_input"] = None
            example["proof_output"] = None
            example["proof"] = None
            return example

        # Extract the input and output
        proof_input = proof_context[:statement_index].strip() + "\n"
        proof_output = proof_context[statement_index:].rstrip()
        if "\n" in proof_output and proof_output.index("\n") <= 2:
            proof_output = proof_output[proof_output.index("\n") + 1 :]
        proof = proof_input + proof_output

        example["proof_input"] = proof_input
        example["proof_output"] = proof_output
        example["proof"] = proof

        return example

    return dataset.map(process_example, num_proc=NUM_PROC)


def mathlib4_dataset_handler(dataset: Dataset):
    """
    Handler for the mathlib4 dataset.

    Args:
        dataset (Dataset): The dataset.
    """

    def process_example(example):
        proof_input = example["header"] + example["statement"]
        # process the proof context
        statement = example["statement"].strip()[: -len("sorry")].strip()
        whole_proof = example["whole_proof"]

        if len(whole_proof.rstrip()) == 0:
            example["proof_input"] = None
            example["proof_output"] = None
            example["proof"] = None
            return example

        if len(statement) > 0 and statement in whole_proof:
            statement_index = whole_proof.index(statement) + len(statement)
        else:
            statement_index = _get_index_of_theorem_statement(whole_proof)

        if statement_index == -1:
            if VERBOSE:
                print(
                    "\n\n=========No match found for proof context. example: ",
                    example["whole_proof"],
                    "\n\n\n",
                )
            example["proof_input"] = None
            example["proof_output"] = None
            example["proof"] = None
            return example

        # Extract the input and output
        proof_input = proof_input.strip()[: -len("sorry")].strip() + "\n"
        proof_output = whole_proof[statement_index:].rstrip()
        if "\n" in proof_output and proof_output.index("\n") <= 2:
            proof_output = proof_output[proof_output.index("\n") + 1 :]
        proof = proof_input + proof_output

        example["proof_input"] = proof_input
        example["proof_output"] = proof_output
        example["proof"] = proof

        return example

    dataset = dataset.add_column("source", ["mathlib4"] * len(dataset))
    return dataset.map(process_example, num_proc=NUM_PROC)


def general_dataset_handler(dataset: Dataset, column_for_proof_context: str):
    """
    General handler for datasets.

    Args:
        dataset (str): dataset.
        column_for_proof_context (str): The column name of the proof context.
    """

    def process_whole_proof(example):
        proof_context = example[column_for_proof_context]
        # process the proof context
        statement_index = _get_index_of_theorem_statement(proof_context)

        if statement_index == -1:
            if VERBOSE:
                if "uuid" in example:
                    print(f"UUID: {example['uuid']}")
                print(
                    "\n\n=========No match found for proof context. example: ",
                    example[column_for_proof_context],
                    "\n\n\n",
                )
            example["proof_input"] = ""
            example["proof_output"] = ""
            example["proof"] = ""
            return example

        # Extract the input and output
        proof_input = proof_context[:statement_index].rstrip() + "\n"
        proof_output = proof_context[statement_index:].rstrip()
        if "\n" in proof_output and proof_output.index("\n") <= 2:
            proof_output = proof_output[proof_output.index("\n") + 1 :]
        proof = proof_input + proof_output

        example["proof_input"] = proof_input
        example["proof_output"] = proof_output
        example["proof"] = proof

        return example

    if "is_valid_with_sorry" in dataset.column_names:
        dataset = dataset.filter(lambda x: x["is_valid_with_sorry"], num_proc=NUM_PROC)

    return dataset.map(process_whole_proof, num_proc=NUM_PROC)


def create_wholeproof_data(
    dataset_ids: List[str],
    proof_columns: List[str],
    handler_types: List[str],
    test_tags: List[str],
    train_dataset_id: str,
    test_dataset_id: str,
    no_train_tags: List[str] = [],
    columns_to_keep: List[str] = [],
    dataset_splits: Union[List[str], str] = "train",
):
    """
    Generates wholeproof data from multiple HF datasets with formal proofs.

    Args:
        dataset_ids (List[str]): The IDs of the datasets.
        proof_columns (List[str]): The column names of the proofs.
        handler_types (List[str]): The handler types for the datasets.
        test_tags (List[str]): The tags to filter for the test dataset.
        train_dataset_id (str): The ID of the train dataset.
        test_dataset_id (str): The ID of the test dataset.

        Optional:
        columns_to_keep (List[str]): The columns to keep in the output dataset, whenever available.
        dataset_splits (Union[List[str], str]): The splits of the datasets to use.
    """

    # Get the dataset splits
    if isinstance(dataset_splits, str):
        dataset_splits = [dataset_splits] * len(dataset_ids)
    elif len(dataset_splits) != len(dataset_ids):
        raise ValueError(
            "The number of dataset splits must be equal to the number of dataset IDs."
        )

    print(f"Merging datasets: {dataset_ids} with splits: {dataset_splits}")
    print(f"Keeping columns: {columns_to_keep} whenever available.\n")

    whole_proof_train_dict = dict()

    for col_name in columns_to_keep:
        whole_proof_train_dict[col_name] = []

    whole_proof_train_dict["proof_input"] = []
    whole_proof_train_dict["proof_output"] = []
    whole_proof_train_dict["proof"] = []

    test_data_dict = dict()
    test_data_dict["formal_statement"] = []
    test_data_dict["proof_input"] = []
    test_data_dict["proof_output"] = []
    test_data_dict["proof"] = []
    for col_name in columns_to_keep:
        test_data_dict[col_name] = []

    for dataset_id, dataset_split, proof_col, handler_type in zip(
        dataset_ids, dataset_splits, proof_columns, handler_types
    ):
        ds = load_dataset(dataset_id, split=dataset_split)
        if handler_type == "auto_statements":
            ds = auto_statements_dataset_handler(ds)
        elif handler_type == "mathlib4":
            ds = mathlib4_dataset_handler(ds)
        else:
            ds = general_dataset_handler(ds, proof_col)

        lends = len(ds)
        print(f"Number of examples in {dataset_id}: {lends}")

        # filter out empty and None proofs
        ds = ds.filter(
            lambda x: x["proof_input"] is not None
            and len(x["proof_input"]) > 0
            and x["proof_output"] is not None
            and len(x["proof_output"]) > 0,
            num_proc=NUM_PROC,
        )

        # filter out data points for test dataset
        if "tags" in ds.column_names:
            test_ds = ds.filter(
                lambda x: any(tag in x["tags"] for tag in test_tags), num_proc=NUM_PROC
            )
            test_data_dict["formal_statement"].extend(test_ds["proof_input"])
            test_data_dict["proof_input"].extend(test_ds["proof_input"])
            test_data_dict["proof_output"].extend(test_ds["proof_output"])
            test_data_dict["proof"].extend(test_ds["proof"])

            for col_name in columns_to_keep:
                if col_name in test_ds.column_names:
                    test_data_dict[col_name].extend(test_ds[col_name])
                else:
                    test_data_dict[col_name].extend([None] * len(test_ds))

            print(f"Number of test examples in {dataset_id}: {len(test_ds)}")

            # filter some test data points from the training dataset
            ds = ds.filter(
                lambda x: not any(tag in x["tags"] for tag in no_train_tags),
                num_proc=NUM_PROC,
            )

        lends = len(ds)

        print(f"Number of proofs after filtering in {dataset_id}: {lends}")

        whole_proof_train_dict["proof_input"].extend(ds["proof_input"])
        whole_proof_train_dict["proof_output"].extend(ds["proof_output"])
        whole_proof_train_dict["proof"].extend(ds["proof"])

        for col_name in columns_to_keep:
            if col_name in ds.column_names:
                whole_proof_train_dict[col_name].extend(ds[col_name])
            else:
                whole_proof_train_dict[col_name].extend([None] * lends)

    whole_proof_train_ds = Dataset.from_dict(whole_proof_train_dict)
    test_ds = Dataset.from_dict(test_data_dict)

    # print the number of proofs and test data points
    print(f"Number of training proofs generated: {len(whole_proof_train_ds)}")
    print(f"Number of test data points generated: {len(test_ds)}")

    # Save the data
    whole_proof_train_ds.push_to_hub(train_dataset_id, private=True)
    test_ds.push_to_hub(test_dataset_id, private=True)


if __name__ == "__main__":
    """
    This script processes and merges multiple Hugging Face datasets containing formal proofs,
    creating training and testing datasets.

    Args:
        dataset_ids: List of dataset IDs from Hugging Face to process.
        proof_columns: List of column names containing proofs in each dataset.
        handler_types: List of handler types (`auto_statements`, `mathlib4`, `general`) for processing.
        test_tags: Tags to identify examples for the test dataset.
        train_dataset_id: Dataset ID for the merged training dataset on the Hub.
        test_dataset_id: Dataset ID for the test dataset on the Hub.
        no_train_tags: Tags to exclude from the training dataset.
        columns_to_keep: List of additional columns to include in the final datasets.
        dataset_splits: Dataset splits (e.g., "train", "test") to process.
    """

    create_wholeproof_data(
        [
            "AI-MO/human-proofs-sft",
            "AI-MO/auto-sft-moon-santa-v2beta1",
            "AI-MO/human-sft-v1pick100",
        ],
        ["formal_proof", "formal_proof", "formal_proof"],
        [
            "general",
            "general",
            "general",
        ],
        ["source:MATH-train", "source:MATH-test"],
        "AI-MO/wholeproof-data-beta1pick100-250107",
        "AI-MO/wholeproof-test-data-250107",
        no_train_tags=["source:MATH-test"],
        columns_to_keep=["uuid", "problem", "source", "tags"],
    )
