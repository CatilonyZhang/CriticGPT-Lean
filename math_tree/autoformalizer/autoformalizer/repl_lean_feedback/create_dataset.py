import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from func_timeout import FunctionTimedOut, func_timeout
from tqdm import tqdm

from autoformalizer.eval_utils.constants import base
from autoformalizer.repl_lean_feedback.intermediate_states import (
    process_code,
    process_file,
)


def create_dataset_from_folder(
    folder_path,
    path_to_repo,
    output_file,
    recurse=False,
    num_workers=16,
    timeout_threshold=360,
    mode="leaf",
):

    data = []

    # Create the list of files from the folder
    if recurse:
        # Search recursively for .lean files
        file_list = glob.glob(os.path.join(path_to_repo, folder_path, "**/*.lean"), recursive=True)
    else:
        # Search only in the specified folder (not recursive)
        file_list = glob.glob(os.path.join(path_to_repo, folder_path, "*.lean"), recursive=False)
    print(f"Found {len(file_list)} files")

    # Traverse with tqdm progress bar
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        future_to_file = {
            executor.submit(
                process_file_parallel, file_path, path_to_repo, timeout_threshold, mode
            ): file_path
            for file_path in file_list
        }

        # Progress bar
        with tqdm(
            total=len(file_list),
            desc=f"Extracting tactic states from files in repository {path_to_repo.split('/')[-1]}",
        ) as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_data = future.result()
                    data.extend(file_data)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                # Update progress bar
                pbar.update(1)

    # Write to output file
    with open(output_file, "w") as f:
        for entry in data:
            json_line = json.dumps(entry)
            f.write(json_line + "\n")


def has_sorry(theorem_snap):
    return any(
        ("sorry" in step["tactic"] or "admit" in step["tactic"])
        for step in theorem_snap.steps
    )


def process_file_parallel(file_path, path_to_repo, timeout_threshold, mode):
    """Function to process a file and return the processed data."""
    data = []
    if not os.path.isfile(file_path) or not file_path.endswith(".lean"):
        print("Invalid file path: ", file_path)
        return data
    
    try:
        result = func_timeout(
            timeout_threshold, process_file, args=(file_path, path_to_repo, mode)
        )
        for theorem_snap in result:
            theorem_snap.folder = os.path.dirname(os.path.relpath(file_path, base))
            theorem_snap.file = os.path.basename(file_path)
            theorem_snap.has_sorry = has_sorry(theorem_snap)
            data.append(theorem_snap.to_dict())
    except FunctionTimedOut:
        print(f"Processing of file {file_path} timed out.")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return data


def process_code_parallel_numina(
    example, code_column, dataset_name, timeout_threshold, mode
):
    data = []
    try:
        code = example[code_column]
        id = str(example["id"])
        # Use func_timeout to enforce a timeout on process_code
        result = func_timeout(timeout_threshold, process_code, args=(code, mode))
        for theorem_snap in result:
            theorem_snap.folder = dataset_name
            theorem_snap.file = id + ".lean"
            theorem_snap.has_sorry = has_sorry(theorem_snap)
            data.append(theorem_snap.to_dict())
    except FunctionTimedOut:
        print(f"Processing of example ID {id} timed out.")
    except Exception as e:
        print(f"Error processing example ID {id}: {e}")
    return data


def process_code_parallel_auto(
    example, code_column, dataset_name, timeout_threshold, mode
):
    data = []
    try:
        codestr = example[code_column]
        codes = codestr.strip("[]").split('", "')
        # Remove any leading or trailing quotes from each code
        codes = re.findall(r'"(.*?)"', codestr)
        id = str(example["uuid"])
        for i, code in enumerate(codes):
            code = code.replace(r"\n", "\n")
            code = code.encode().decode("unicode_escape")
            # Use func_timeout to enforce a timeout on process_code
            result = func_timeout(timeout_threshold, process_code, args=(code, mode))
            for theorem_snap in result:
                theorem_snap.folder = dataset_name
                theorem_snap.file = id + str(i) + ".lean"
                theorem_snap.has_sorry = has_sorry(theorem_snap)
                data.append(theorem_snap.to_dict())
    except FunctionTimedOut:
        print(f"Processing of example UUID {id} timed out.")
    except Exception as e:
        print(f"Error processing example UUID {id}: {e}")
    return data


def create_dataset_from_hf_dataset(
    dataset_name,
    split,
    code_column,
    output_file,
    timeout_threshold=360,
    mode="leaf",
    num_proc=16,
    kind="numina",
):
    data = []
    dataset = load_dataset(dataset_name)
    print("Dataset has been loaded")
    if split:
        dataset = dataset[split]

    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        # Submit tasks
        if kind == "numina":
            future_to_file = {
                executor.submit(
                    process_code_parallel_numina,
                    example,
                    code_column,
                    dataset_name,
                    timeout_threshold,
                    mode,
                ): example
                for example in dataset
            }
        elif kind == "auto":
            future_to_file = {
                executor.submit(
                    process_code_parallel_auto,
                    example,
                    code_column,
                    dataset_name,
                    timeout_threshold,
                    mode,
                ): example
                for example in dataset
            }

        # Progress bar
        with tqdm(
            total=len(dataset),
            desc=f"Extracting tactic states from dataset {dataset_name}",
        ) as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_data = future.result()
                    data.extend(file_data)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                # Update progress bar
                pbar.update(1)

    # Write to output file
    with open(output_file, "w") as f:
        for entry in data:
            json_line = json.dumps(entry)
            f.write(json_line + "\n")
