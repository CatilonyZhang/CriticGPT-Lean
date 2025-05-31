import json
import os
import pathlib
import random
import sys

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from autoformalizer.clients import lean4_client
from autoformalizer.eval_utils import lean_feedback


def main():
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    client = lean4_client.Lean4Client(
        "https://kimina.saas.moonshot.cn/lean4-evaluator",
        api_key=os.environ.get("MOONSHOT_LEAN4_API_KEY"),
    )
    ds = load_dataset("AI-MO/auto-proofs-v2-haiming", split="train", num_proc=10)
    working_dir = pathlib.Path("/home/jia/auto_proofs_v2")
    working_dir.mkdir(parents=True, exist_ok=True)

    # randomly select 100 samples with random seed
    random.seed(42)
    # ds = ds.select(random.sample(range(len(ds)), 2000))
    # randomly select 100 uuid
    # uuids = set(random.sample(ds["uuid"], 100))
    # ds = ds.select(random.sample(range(len(ds)), 2000))
    # offset = 12000 * 1024
    # ds = ds.select(range(offset, offset + 100 * 1024))
    # print(ds)

    # make sure we have proof_id, uuid, and proof
    # use batch processing
    processing_batch_size = 1000000
    for i in range(52 * processing_batch_size, len(ds), processing_batch_size):
        samples = []
        end_batch = min(i + processing_batch_size, len(ds))
        print(f"Processing batch {i} to {end_batch}, total: {len(ds)}")
        for sample in tqdm(ds.select(range(i, end_batch))):
            samples.append(
                {
                    "proof_id": sample["proof_id"],
                    "uuid": sample["uuid"],
                    "proof": sample["formal_proof"],
                }
            )

        # warning, result is not in the same order as samples
        batch_size = 20
        results = lean4_client.batch_verify_proof(
            client,
            samples,
            timeout=60,
            num_threads=300,
            working_dir=working_dir,
            batch_size=batch_size,
        )

        # Log statistics
        total_proofs = len(samples)
        is_valid_proofs = [result["is_valid_no_sorry"] for result in results]
        logger.info(f"Total proofs to verify: {total_proofs}")
        logger.info(f"Total valid proofs: {sum(is_valid_proofs)}")

        # log uuid with at least one valid proof
        valid_uuids = set()
        all_uuids = set(x["uuid"] for x in results)
        for result in results:
            if result["is_valid_no_sorry"]:
                valid_uuids.add(result["uuid"])
        logger.info(f"Number of uuids: {len(all_uuids)}")
        logger.info(
            f"Number of uuids with at least one valid proof: {len(valid_uuids)}"
        )


def push_to_hub():
    working_dir = pathlib.Path("/home/jia/auto_proofs_v2")
    dataset = load_dataset("AI-MO/auto-proofs-v2-haiming", split="train", num_proc=10)

    def _update_proof_status(sample):
        uuid = sample["uuid"]
        proof_id = sample["proof_id"]
        filepath = working_dir / uuid / f"{proof_id}.json"
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    response = json.load(f)
            except json.JSONDecodeError:
                response = {"error": "JSON file corrupted"}
                return {
                    "is_valid_no_sorry": False,
                    "has_connection_error": False,
                    "status": "JSON_FILE_CORRUPTED",
                    "lean_server_feedback": json.dumps(response),
                }
            error_message = response.get("error", None)
            json_response = response.get("response", None)

            is_valid_no_sorry = (not bool(error_message)) and (
                not lean_feedback.has_error(json_response, accept_sorry=False)
            )

            connection_error = bool(error_message) and (
                "Lean process timed out" not in error_message
            )
            data = {
                "is_valid_no_sorry": is_valid_no_sorry,
                "has_connection_error": connection_error,
                "status": "FINISHED",
                "lean_server_feedback": json.dumps(response),
            }
            return data
        else:
            return {
                "is_valid_no_sorry": False,
                "has_connection_error": False,
                "status": "TODO",
                "lean_server_feedback": "",
            }

    dataset = dataset.map(_update_proof_status, num_proc=50)
    # compute some statistics
    finished = dataset.filter(lambda x: x["status"] == "FINISHED", num_proc=30)
    valid_rate = sum(finished["is_valid_no_sorry"]) / len(finished)
    connection_error_rate = sum(finished["has_connection_error"]) / len(finished)
    logger.info(f"Valid rate: {valid_rate}")
    logger.info(f"Connection error rate: {connection_error_rate}")

    dataset.push_to_hub("AI-MO/auto-proofs-v2", private=True)

    df = dataset.to_pandas()
    # calculate valid rate
    valid_rate = df["is_valid_no_sorry"].sum() / len(df)
    logger.info(f"valid rate: {valid_rate}")

    # calculate valid rate for each uuid using groupby
    uuid_group = df.groupby("uuid")["is_valid_no_sorry"].sum()

    # find all uuids with at least one valid proof
    valid_uuids = uuid_group[uuid_group > 0].index
    logger.info(f"Number of uuids: {len(uuid_group)}")
    logger.info(f"Number of uuids with at least one valid proof: {len(valid_uuids)}")


if __name__ == "__main__":
    main()
    push_to_hub()
