from copy import copy
import datasets
import pandas as pd
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor

def process_and_save_chunk(data_chunk, shard_dir, chunk_idx):
    """
    Process a chunk of data and save it to a Parquet file.
    """
    print(f"Processing chunk {chunk_idx} with {len(data_chunk)} samples")
    result_data = []
    
    # Process the chunk
    for sample in tqdm(data_chunk):
        if sample["n_correct_proofs"] == 0:
            continue
        if sample["is_negation"] == True:
            continue
        formal_statement = sample["formal_statement"]
        for proof in sample["correct_proof_samples"]:
            the_sample = copy(sample)
            the_sample["proof"] = proof
            del the_sample["correct_proof_samples"]
            result_data.append(the_sample)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(result_data)
    
    # Create shard file name based on chunk index
    shard_file = os.path.join(shard_dir, f"shard_{chunk_idx}.parquet")
    
    # Write the shard to disk
    df.to_parquet(shard_file, engine='pyarrow', index=False)
    print(f"Saved {len(result_data)} samples to {shard_file}")

def process_dataset_in_parallel(data, chunk_size, shard_dir):
    """
    Process the dataset in parallel and save it to multiple Parquet files (shards).
    """
    # Split the dataset into chunks for parallel processing
    num_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size > 0 else 0)
    print(f"Number of chunks: {num_chunks}")
    
    # Create a ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=None) as executor:
        chunk_idx = 0
        futures = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(data))
            data_chunk = data.select(range(start_idx, end_idx))  # Slice data
            
            # Submit each chunk for processing
            futures.append(executor.submit(process_and_save_chunk, data_chunk, shard_dir, chunk_idx))
            chunk_idx += 1
        
        # Wait for all futures to complete
        for future in futures:
            future.result()  # Block until each future is done

if __name__ == "__main__":
    # Load the original dataset
    data = datasets.load_dataset("AI-MO/auto-statements-moon-santa-prover-v1", split="train", cache_dir="/mnt/moonfs/kimina-m2/.cache")

    # Initialize variables
    chunk_size = 10000  # Set chunk size for saving
    shard_dir = "/mnt/moonfs/kimina-m2/.cache/temp_expanded_proof_shards_no_negation/"
    os.makedirs(shard_dir, exist_ok=True)  # Ensure the shard directory exists

    # Process the dataset in parallel and save it to Parquet
    process_dataset_in_parallel(data, chunk_size, shard_dir)