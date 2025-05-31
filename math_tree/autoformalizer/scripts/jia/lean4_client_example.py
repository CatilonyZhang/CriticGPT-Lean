import os
import pathlib
import random
import sys

from datasets import load_dataset
from loguru import logger

from autoformalizer.clients import lean4_client


def main():
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    client = lean4_client.Lean4Client(
        "https://kimina.saas.moonshot.cn/lean4-evaluator",
        api_key=os.environ.get("MOONSHOT_LEAN4_API_KEY"),
    )
    ds = load_dataset("AI-MO/math-test-inference-results", split="train")
    working_dir = pathlib.Path("/tmp/math_test_inference_results")
    working_dir.mkdir(parents=True, exist_ok=True)

    # randomly select 100 samples with random seed
    random.seed(42)
    ds = ds.select(random.sample(range(len(ds)), 2000))

    # make sure we have proof_id, uuid, and proof
    samples = []
    for sample in ds:
        samples.append(
            {
                "proof_id": sample["proof_id"],
                "uuid": sample["uuid"],
                "proof": sample["proof"],
            }
        )

    # warning, result is not in the same order as samples
    results = lean4_client.parallel_verify_proof(
        client, samples, timeout=60, num_proc=100, working_dir=working_dir
    )
    # Log statistics
    total_proofs = len(samples)
    no_error_proofs = sum(not result["has_error"] for result in results)
    logger.info(f"Total proofs to verify: {total_proofs}")
    logger.info(f"No error proofs: {no_error_proofs}")


if __name__ == "__main__":
    main()
