import pathlib
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.synchronize import Lock as LockBase
from typing import Any, Dict, List, Tuple

import pandas as pd
from datasets import load_dataset
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

from autoformalizer.clients import lean4_client
from autoformalizer.jobs.input_sharder import InputSharder
from autoformalizer.jobs.statement_formalizer import ShardProcessor, query_with_retry
from autoformalizer.model_utils.infer_hf_dataset import negate_theorem


def remove_sorry(statement: str) -> str:
    """
    Removes occurrences of 'sorry' in the statement or proof.
    This helps produce final code that compiles without placeholders.
    """
    return statement.split("sorry")[0] + "\n"


def fliter_statement_by_proof_record(
    df: pd.DataFrame,
    proof_record_dir: str,
    filter_threshold: int,
    proof_record_lock: LockBase,
) -> pd.DataFrame:
    """
    Filter out statement that already been proof more than `filter_threshold` times.
    """
    # Read proof record jsonl
    if not pathlib.Path(proof_record_dir).exists():
        logger.warning(f"Proof record file {proof_record_dir} does not exist.")
        return df
    with proof_record_lock:
        proof_record_df = pd.read_json(proof_record_dir, lines=True)

    if len(proof_record_df) == 0:
        logger.warning("Empty proof record file.")
        return df
    # Step 2: Aggregate the proof counts (if necessary) by summing over 'statement_id'
    proof_counts = (
        proof_record_df.groupby("statement_id")["proof_count"].sum().to_dict()
    )

    # Step 3: Filter the input DataFrame based on the proof counts and threshold,
    # and handle special cases with 'neg_' statements.
    negated_count = 0
    filtered_count = 0

    def should_keep(statement_id: str) -> bool:
        # get the negated count and filtered count variables
        nonlocal negated_count, filtered_count

        # Check the regular version of the statement_id
        regular_count = proof_counts.get(statement_id, 0)

        if statement_id.startswith("neg_"):
            neg_count = proof_counts.get(statement_id.replace("neg_", ""), 0)
        else:
            neg_count = proof_counts.get(f"neg_{statement_id}", 0)

        # If either the regular or the 'neg_' version has a proof_count > threshold, we should filter out
        if neg_count > 0:
            negated_count += 1
        if regular_count > filter_threshold:
            filtered_count += 1
        return regular_count <= filter_threshold and neg_count == 0

    # Apply the filtering logic to each statement_id
    filtered_df = df[df["statement_id"].map(should_keep)]

    logger.info(
        f"Filtered {filtered_count} statements, {negated_count} negated statements."
        + f"Total statements: {len(df)}, Remaining statements: {len(filtered_df)}"
    )

    return filtered_df


def write_proof_record(
    df: pd.DataFrame,
    proof_record_dir: str,
    proof_record_lock: LockBase,
):
    """
    Write proof record to jsonl.
    """
    # Group the DataFrame by 'statement_id' and count 'is_valid_no_sorry' per group
    result = (
        df[df["is_valid_no_sorry"]]
        .groupby("statement_id")
        .size()
        .reset_index(name="proof_count")
    )
    new_proof_count = len(result)

    with proof_record_lock:
        # Read the existing proof record file
        if pathlib.Path(proof_record_dir).exists():
            proof_record_df = pd.read_json(proof_record_dir, lines=True)
            # Merge the new results with the existing proof record
            result = pd.concat([proof_record_df, result], ignore_index=True)

            # Aggregate the proof counts by 'statement_id' and sum them
            result = result.groupby("statement_id", as_index=False)["proof_count"].sum()

        # Append the result to the JSON Lines file
        result.to_json(
            proof_record_dir, orient="records", lines=True, mode="w", index=False
        )

    logger.info(
        f"Proof record updated with {new_proof_count}/{len(result)} new proofs."
    )


class ProofGenerator(ShardProcessor):
    """
    Generates proofs for already autoformalized statements.
    Uses the OpenAI completion endpoint (or vLLM-compatible) in a multi-threaded manner.
    """

    def __init__(
        self,
        openai_api_bases: str,
        models: str,
        openai_api_key: str = "EMPTY",
        enable_proof_filtering: bool = False,
        proof_record_dir: str = None,
        proof_record_lock: Any = None,
        filter_threshold: int = 0,
        is_mathlib: bool = False,
    ):
        """
        Args:
            openai_api_base (str): Base URL of the inference server (e.g. 'http://localhost:8081/v1').
            model (str): The model name to use (e.g. 'deepseek-ai/DeepSeek-Prover-V1.5-RL').
            openai_api_key (str, optional): Your OpenAI API key or 'EMPTY' if none is required by local server.
        """
        super().__init__()
        self.clients = {}
        for idx, model in enumerate(models):
            self.clients[model] = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_bases[idx],
            )
        self.models = models
        self.openai_api_bases = openai_api_bases
        self.enable_proof_filtering = enable_proof_filtering
        self.proof_record_dir = proof_record_dir
        self.proof_record_lock = proof_record_lock
        self.filter_threshold = filter_threshold
        self.is_mathlib = is_mathlib

    def _run_single_sample(
        self,
        uuid: str,
        problem_id: str,
        formal_statement: str,
        max_tokens: int,
        temperature: float,
        n: int,
        prompt_template: str,
        cot_prompt_template: str,
    ) -> Dict[str, Any]:
        """
        Generates multiple proof completions for one formal statement using the completions API.
        Returns a dictionary of results.
        """
        # Clean up the statement to remove "sorry" placeholders
        fs_clean = remove_sorry(formal_statement)
        selected_model = random.choice(self.models)

        if selected_model == "deepseek-ai-DeepSeek-Prover-V1_5-RL-cot":
            prompt = cot_prompt_template.format(formal_statement=fs_clean)
        else:
            prompt = prompt_template.format(formal_statement=fs_clean)

        # Special handling for mathlib
        if self.is_mathlib:
            prompt = prompt.rstrip() + "\n"
            prompt = prompt[-4000:]

        try:
            completion = query_with_retry(
                client=self.clients[selected_model],
                type="text_completion",
                model=selected_model.replace("-cot", ""),
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
            )
            # Convert to a dictionary we can process more easily
            completion_dict = completion.to_dict()

            return {
                "uuid": uuid,
                "problem_id": problem_id,
                "formal_statement": formal_statement,
                "cleaned_statement": fs_clean,
                "completions": completion_dict,
                "selected_model": selected_model,
                "error": "",
            }

        except Exception as e:
            logger.error(
                f"Error generating proof for {problem_id}: {e}, {selected_model}"
            )
            return {
                "uuid": uuid,
                "problem_id": problem_id,
                "formal_statement": formal_statement,
                "cleaned_statement": fs_clean,
                "completions": None,
                "selected_model": selected_model,
                "error": str(e),
            }

    def run_batch(
        self,
        samples: List[Dict[str, Any]],
        prompt_template: str,
        cot_prompt_template: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        n: int = 8,
        num_threads: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Generates proofs in parallel for a batch of formal statements.

        Args:
            samples (list): Each dict should contain:
                - 'uuid'
                - 'statement_id' (or any unique ID key)
                - 'formal_statement'
            prompt_template (str): Prompt template for text completion.
            max_tokens (int): Max tokens for proof generation.
            temperature (float): Sampling temperature for diversity.
            n (int): Number of completions to generate per statement.
            num_threads (int): Number of parallel threads to use.

        Returns:
            A list of dictionaries with:
                - 'problem_id'
                - 'formal_statement'
                - 'cleaned_statement'
                - 'completions'
                - 'error' (optional)
        """
        results = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_sample = {}
            for sample in samples:
                uuid = sample["uuid"]
                problem_id = sample["statement_id"]
                formal_statement = sample["formal_statement"]
                future = executor.submit(
                    self._run_single_sample,
                    uuid,
                    problem_id,
                    formal_statement,
                    max_tokens,
                    temperature,
                    n,
                    prompt_template,
                    cot_prompt_template,
                )
                future_to_sample[future] = problem_id

            for future in tqdm(
                as_completed(future_to_sample),
                total=len(future_to_sample),
                desc="Generating proofs",
            ):
                result = future.result()
                results.append(result)

        return results

    def process_shard(
        self,
        shard_id: int,
        input_dir: str,
        output_dir: str,
        prompt_template: str,
        cot_prompt_template: str,
        max_tokens: int = 1000,
        temperature: float = 1.0,
        n: int = 8,
        num_threads: int = 4,
    ) -> Tuple[bool, int]:
        """
        Processes a single shard of formal statements and writes the generated proofs to a Parquet file.

        Args:
            shard_id (int): Shard ID.
            input_dir (str): Directory containing the shard.
            output_dir (str): Directory to save the proofs.
            prompt_template (str): Prompt template for proof completion.
            max_tokens (int): Max tokens for each proof generation.
            temperature (float): Sampling temperature.
            n (int): Number of completions to generate per statement.
            num_threads (int): Number of parallel threads.

        Returns:
            (status: bool, shard_id: int)
        """
        input_dir = pathlib.Path(input_dir)
        output_dir = pathlib.Path(output_dir)
        input_shard_path = input_dir / f"{shard_id}.parquet"
        output_shard_path = output_dir / f"{shard_id}.parquet"

        try:
            if not input_shard_path.exists():
                logger.error(f"Input shard {input_shard_path} does not exist.")
                return False, shard_id

            df = pd.read_parquet(input_shard_path)
            if self.enable_proof_filtering is True:
                df = fliter_statement_by_proof_record(
                    df,
                    proof_record_dir=self.proof_record_dir,
                    filter_threshold=self.filter_threshold,
                    proof_record_lock=self.proof_record_lock,
                )
            samples = df.to_dict(orient="records")

            # Generate proofs
            batch_results = self.run_batch(
                samples=samples,
                prompt_template=prompt_template,
                cot_prompt_template=cot_prompt_template,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
                num_threads=num_threads,
            )

            # Flatten results: each sample has possibly multiple completions
            proof_rows = []
            for br in batch_results:
                completions = br["completions"]
                if completions is None or "choices" not in completions:
                    # Error or no completions
                    proof_rows.append(
                        {
                            "statement_id": br["problem_id"],
                            "uuid": br["uuid"],
                            "formal_statement": br["formal_statement"],
                            "proof_id": str(uuid.uuid4()),
                            "formal_proof": None,
                            "selected_model": br["selected_model"],
                            "error": br.get("error", "no_completions"),
                        }
                    )
                    continue

                # Each choice has text
                for idx, choice in enumerate(completions["choices"]):
                    # We'll append the original statement + the generated text
                    gen_text = choice["text"]
                    if self.is_mathlib and "```" not in gen_text:
                        gen_text = gen_text.split("\n")
                        while gen_text[0].strip() == "":
                            gen_text = gen_text[1:]
                        gen_text = "\n".join(gen_text)
                        gen_text = gen_text.split("\n\n")[0]

                    final_proof = br["cleaned_statement"] + gen_text

                    # Simple parse or cleaning if needed
                    final_proof = final_proof.split("```")[0].strip()

                    proof_rows.append(
                        {
                            "statement_id": br["problem_id"],
                            "uuid": br["uuid"],
                            "formal_statement": br["formal_statement"],
                            "proof_id": f"{br['problem_id']}_{idx}",
                            "formal_proof": final_proof,
                            "selected_model": br["selected_model"],
                            "error": None,
                        }
                    )

            out_df = pd.DataFrame(proof_rows)
            output_shard_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_parquet(output_shard_path, index=False)
            logger.info(f"Shard {shard_id} proofs saved to {output_shard_path}")

            return True, shard_id
        except Exception as e:
            logger.error(f"Error processing shard {shard_id}: {e}")
            return False, shard_id


class ProofVerifier(ShardProcessor):
    """
    Verifies the generated proofs in batch, similar to statement verification.
    Assumes you have a suitable Lean4 or external verification endpoint that accepts proofs.
    """

    def __init__(
        self,
        client_url: str,
        client_api_key: str = None,
        timeout: int = 60,
        num_threads: int = 5,
        batch_size: int = 100,
        enable_proof_filtering: bool = False,
        proof_record_dir: str = None,
        proof_record_lock=None,
    ):
        """
        Args:
            client_url (str): URL of the Lean4 or proof verification service.
            client_api_key (str, optional): API key for the service if required.
            timeout (int, optional): Timeout in seconds for the batch verification call.
            num_threads (int, optional): Number of threads used within the verification client.
        """
        super().__init__()
        self.client = lean4_client.Lean4Client(url=client_url, api_key=client_api_key)
        self.timeout = timeout
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.enable_proof_filtering = enable_proof_filtering
        self.proof_record_dir = proof_record_dir
        self.proof_record_lock = proof_record_lock

    def process_shard(
        self,
        shard_id: int,
        input_dir: str,
        output_dir: str,
    ) -> Tuple[bool, int]:
        """
        Read the generated proofs from a shard, verify them, and write out the results.

        1. Reads a Parquet shard containing 'statement_id', 'formal_proof', etc.
        2. Performs batch verification.
        3. Merges the results, writes back to the same or a new Parquet file.

        Returns:
            (status: bool, shard_id: int)
        """
        input_dir = pathlib.Path(input_dir)
        output_dir = pathlib.Path(output_dir)
        input_shard_path = input_dir / f"{shard_id}.parquet"
        output_shard_path = output_dir / f"{shard_id}.parquet"

        try:
            if not input_shard_path.exists():
                logger.error(f"Input shard {input_shard_path} does not exist.")
                return False, shard_id

            df = pd.read_parquet(input_shard_path)
            proofs = df.to_dict(orient="records")

            # Prepare for verification
            # Each item: { "proof_id", "uuid", "proof", ... }
            verification_samples = []
            for row in proofs:
                # Some data might be missing or named differently
                # We'll assume "formal_proof" is the code to verify
                # and "proof_id" is a unique ID
                verification_samples.append(
                    {
                        "proof_id": row["proof_id"],
                        "uuid": row["uuid"],
                        "proof": row["formal_proof"],
                    }
                )

            # Batch verification call
            verification_results = lean4_client.batch_verify_proof(
                client=self.client,
                samples=verification_samples,
                timeout=self.timeout,
                num_threads=self.num_threads,
                batch_size=self.batch_size,
                working_dir=None,
            )

            # Convert verification results to DataFrame
            results_df = pd.DataFrame(verification_results)
            # Typically returns columns like: proof_id, formal_proof, is_valid_with_sorry, etc.

            # Merge results back
            merged_df = pd.merge(
                df, results_df, on="proof_id", how="left", suffixes=("", "_verified")
            )

            if self.enable_proof_filtering is True:
                write_proof_record(
                    merged_df,
                    proof_record_dir=self.proof_record_dir,
                    proof_record_lock=self.proof_record_lock,
                )

            # Save the verified data
            output_dir.mkdir(parents=True, exist_ok=True)
            merged_df.to_parquet(output_shard_path, index=False)
            logger.info(f"Shard {shard_id} verified, results in {output_shard_path}")

            return True, shard_id
        except Exception as e:
            logger.error(f"Error verifying shard {shard_id}: {e}")
            return False, shard_id


def main_example():
    """
    Example usage of ProofGenerator and ProofVerifier.
    (Not an official pipeline, just a demonstration.)
    """
    start_time = time.time()
    # 1. Generate proofs for shard 0
    generator = ProofGenerator(
        openai_api_base="http://localhost:8082/v1",
        model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
        openai_api_key="EMPTY",
    )

    # This prompt template is just an example; adapt it to your needs
    prompt_template = (
        "Complete the following Lean 4 code:\n\n```lean4\n{formal_statement}"
    )

    # Suppose we have input shards in /path/to/statements,
    # and we want to write the proofs to /path/to/proofs
    ds = load_dataset("AI-MO/AutoformalizationV2B0", split="train")
    ds = ds.select(range(300))  # Just for testing

    working_dir = pathlib.Path("/tmp/prover_example")
    problem_dir = working_dir / "problems"
    proof_dir = working_dir / "proofs"
    verified_proof_dir = working_dir / "verified_proofs"

    ds = ds.map(lambda x: {"formal_statement": negate_theorem(x["lean_code"])})
    ds = ds.filter(lambda x: x["formal_statement"])
    ds = ds.map(
        lambda x: {"statement_id": str(uuid.uuid4()), "uuid": str(uuid.uuid4())}
    )
    sharder = InputSharder(
        ds,
        problem_dir,
        shard_size=300,
        columns_to_read=["uuid", "formal_statement", "statement_id"],
    )

    shards = []
    for shard_id, shard_path in sharder:
        shards.append((shard_id, shard_path))
        logger.info(f"Shard {shard_id} saved to {shard_path}")

    generator(
        shard_id=0,
        input_dir=problem_dir,
        output_dir=proof_dir,
        prompt_template=prompt_template,
        max_tokens=1024,
        temperature=1.0,
        n=4,
        num_threads=20,
    )

    # 2. Verify the generated proofs for shard 0
    verifier = ProofVerifier(
        client_url="https://kimina.saas.moonshot.cn/lean4-evaluator",
        timeout=60,
        num_threads=5,
    )

    verifier(
        shard_id=0,
        input_dir=proof_dir,
        output_dir=verified_proof_dir,
    )
    # compute time
    end_time = time.time()
    logger.info(f"Total time: {end_time - start_time} seconds")

    df = pd.read_parquet(verified_proof_dir / "00000.parquet")
    valid_rate = df["is_valid_no_sorry"].mean()
    logger.info(f"Valid rate: {valid_rate}")
    # valid uuid
    uuid_df = df.groupby("uuid").sum()
    uuid_valid_rate = (uuid_df["is_valid_no_sorry"] > 0).mean()
    logger.info(f"UUID valid rate: {uuid_valid_rate}")


if __name__ == "__main__":
    # Simple test usage (not the same as your existing run() function).
    # You can adapt this pattern or ignore it.
    main_example()
