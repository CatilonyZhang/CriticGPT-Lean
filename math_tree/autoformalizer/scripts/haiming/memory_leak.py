import os
import tracemalloc  # For detailed memory tracking

import psutil
from datasets import load_dataset
from memory_profiler import profile  # For profiling functions
from tqdm import tqdm

from autoformalizer.clients.lean4_client import Lean4Client, batch_verify_proof


# Function to check current memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert bytes to MB


# Start tracemalloc to track memory allocations
tracemalloc.start()


@profile
def example_batch(client):
    dataset = load_dataset("AI-MO/math-test-inference-results", split="train")
    working_dir = None
    # Select the first 500 uuids
    dataset = dataset.select(range(100 * 64))
    print(dataset)

    # Initial memory usage
    print(f"Initial memory usage: {get_memory_usage()} MB")

    new_samples = []
    for sample in tqdm(dataset):
        new_samples.append(
            {
                "proof_id": sample["uuid"] + "_" + str(sample["proof_id"]),
                "uuid": sample["uuid"],
                "proof": sample["proof"],
            }
        )

    # Memory usage after preparing new_samples
    print(f"Memory usage after preparing new_samples: {get_memory_usage()} MB")

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

    # Memory usage after calling batch_verify_proof
    print(f"Memory usage after batch_verify_proof: {get_memory_usage()} MB")

    print(len(results))
    print(results[0])

    # Final memory usage
    print(f"Final memory usage: {get_memory_usage()} MB")

    # Snapshot the memory usage
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    print("[Top 10 memory consuming lines]")
    for stat in top_stats[:10]:
        print(stat)


# Call the example function to profile it
client = Lean4Client(
    "http://lean4-evaluator.app.msh.team/",
)
for _ in range(100):
    example_batch(client)
