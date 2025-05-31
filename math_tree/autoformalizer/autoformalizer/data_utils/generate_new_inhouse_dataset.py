import copy
import hashlib

import datasets
import fire
from transformers import AutoTokenizer

from autoformalizer.clients.lean4_client import Lean4Client, batch_verify_proof
from autoformalizer.data_utils.process_proof import (
    get_statement_split_indexes,
    remove_comments,
)


def hash_to_float(s: str) -> float:
    # Hash the string using SHA-256
    hash_value = hashlib.sha256(s.encode()).hexdigest()

    # Convert the hex hash value to an integer
    int_value = int(hash_value, 16)

    # Normalize the integer value to the range [0, 1]
    return int_value / (2**256 - 1)


def generate_new_inhouse_dataset(
    client_url: str,
    client_batch_size: int,
    client_threads: int,
    dataset_id: str,
    proof_column: str,
    inhouse_dataset_id: str,
):
    """
    Generate a new inhouse dataset from the given dataset by spliting the proof into input and output
    """
    client = Lean4Client(client_url)

    # Tokenizer use for filter out over length input's
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

    # Load the datasets
    dataset = datasets.load_dataset(dataset_id, split="train")

    # Add proof id for verification
    if "proof_id" not in dataset.column_names:
        dataset = dataset.add_column("proof_id", [str(i) for i in range(len(dataset))])
    dataset = dataset.rename_column(proof_column, "proof")

    samples = [sample for sample in dataset]
    results = batch_verify_proof(
        client=client,
        samples=samples,
        batch_size=client_batch_size,
        num_threads=client_threads,
    )

    num_valid = 0
    result_dataset = []
    # split input and output
    for result in results:
        if not result["is_valid_no_sorry"]:
            continue
        num_valid += 1
        indexes = get_statement_split_indexes(result["proof"])
        statement_id = result["uuid"]
        for i, index in enumerate(indexes):
            new_sample = copy.copy(result)
            new_sample["proof_input"] = new_sample["proof"][:index]
            if len(tokenizer.encode(new_sample["proof_input"])) > 1800:
                continue
            new_sample["proof_output"] = new_sample["proof"][index:]
            new_sample["statement_id"] = str(statement_id) + f"_{i:02}"
            new_sample["human_proof"] = new_sample["proof"]
            new_sample["proof_input_no_comment"] = remove_comments(
                new_sample["proof_input"]
            )
            new_sample["proof_id"] = new_sample["statement_id"]
            new_sample["proof"] = new_sample["proof_input"] + " sorry"
            result_dataset.append(new_sample)

    print(f"Number of valid proofs: {num_valid}")
    print(f"Number of new statements: {len(result_dataset)}")

    # Check the splited input again
    result_dataset = batch_verify_proof(
        client=client,
        samples=result_dataset,
        batch_size=client_batch_size,
        num_threads=client_threads,
    )

    filtered_result_dataset = []
    for i, result in enumerate(result_dataset):
        if result["is_valid_with_sorry"] is False:
            print("====================")
            print(result["proof_id"])
            print(result["proof"])
            print("====================")
            print("\n\n\n\n")
        else:
            filtered_result_dataset.append(result)

    result_dataset = datasets.Dataset.from_list(filtered_result_dataset)
    result_dataset = result_dataset.rename_column("proof_input", "formal_statement")
    result_dataset = result_dataset.rename_column(
        "proof_input_no_comment", "formal_statement_no_comment"
    )
    result_dataset = result_dataset.rename_column("proof_output", "ground_truth")
    # TODO: make these columns as parameters
    result_dataset = result_dataset.remove_columns(
        [
            "problem",
            "lean_code",
            "is_valid_no_sorry",
            "is_valid_with_sorry",
            "proof_id",
            "has_error",
            "lean_feedback",
        ]
    )

    train = []
    inhouse_evaluation_math_train = []
    inhouse_evaluation_math_test = []
    inhouse_evaluation_others_10_percent = []
    for samples in result_dataset:
        in_test = False
        if samples["source"] == "MATH-train":
            inhouse_evaluation_math_train.append(samples)
        elif samples["source"] == "MATH-test":
            inhouse_evaluation_math_test.append(samples)
            in_test = True
        else:
            if hash_to_float(samples["statement_id"]) < 0.1:
                inhouse_evaluation_others_10_percent.append(samples)
                in_test = True
        if not in_test:
            train.append(samples)

    inhouse_evaluation_math_train = datasets.Dataset.from_list(
        inhouse_evaluation_math_train
    )
    inhouse_evaluation_math_test = datasets.Dataset.from_list(
        inhouse_evaluation_math_test
    )
    inhouse_evaluation_others_10_percent = datasets.Dataset.from_list(
        inhouse_evaluation_others_10_percent
    )
    train = datasets.Dataset.from_list(train)
    datasets.DatasetDict(
        {
            "train": train,
            "inhouse_evaluation_math_train": inhouse_evaluation_math_train,
            "inhouse_evaluation_math_test": inhouse_evaluation_math_test,
            "inhouse_evaluation_others_10_percent": inhouse_evaluation_others_10_percent,
            "full": result_dataset,
        }
    ).push_to_hub(inhouse_dataset_id, private=True)


if __name__ == "__main__":
    """
    Example usage:
    python -m autoformalizer.data_utils.generate_new_inhouse_dataset \
        --client_url "http://lean4-evaluator.app.msh.team/" \
        --client_batch_size 1 \
        --client_threads 220 \
        --dataset_id "AI-MO/numina-math-lean4" \
        --proof_column "lean4_solution" \
        --inhouse_dataset_id "AI-MO/inhouse-evaluation-v1-20250102"
    """
    fire.Fire(generate_new_inhouse_dataset)
