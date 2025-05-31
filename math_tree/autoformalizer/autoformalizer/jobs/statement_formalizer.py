import pathlib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import fire
import pandas as pd
import tenacity
from datasets import load_dataset
from loguru import logger
from numinamath.dataset_utils import make_natural_language_prompt
from openai import OpenAI
from tqdm import tqdm

from autoformalizer.clients import lean4_client
from autoformalizer.data_utils import constants, process_statement
from autoformalizer.data_utils.user_prompt import get_user_prompt
from autoformalizer.model_utils.infer_hf_dataset import negate_theorem


def retry_if_service_error(exception):
    if hasattr(exception, "args") and len(exception.args) > 0:
        error_message = exception.args[0]
        # Check if the error message contains the specific text you want to retry on
        return "service not found" in error_message
    return False


@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    stop=tenacity.stop_after_attempt(1000),
    retry=retry_if_service_error,
)
def query_with_retry(client, type="chat", **kargs):
    if type == "chat":
        return client.chat.completions.create(**kargs)
    elif type == "text_completion":
        return client.completions.create(**kargs)
    else:
        raise ValueError(f"Unknown type: {type}")


class ShardProcessor:
    def __init__(self):
        """
        Base class for processing shards, with built-in logic to skip already processed shards.
        """
        pass

    def _check_output_exists(self, shard_id: int, output_dir: pathlib.Path) -> bool:
        """
        Check if the output file for the given shard_id already exists.

        Args:
            shard_id (int): The ID of the shard.
            output_dir (pathlib.Path): The directory to save output shards.

        Returns:
            bool: True if output file exists, False otherwise.
        """
        output_shard_path = output_dir / f"{shard_id}.parquet"
        return output_shard_path.exists()

    def __call__(
        self, shard_id: str, input_dir: pathlib.Path, output_dir: pathlib.Path, **kwargs
    ):
        """
        Process a shard, but only if the output doesn't already exist.

        Args:
            shard_id (int): The ID of the shard to process.
            input_dir (pathlib.Path): Directory containing input shards.
            output_dir (pathlib.Path): Directory to save output shards.

        Returns:
            tuple(bool, int): (status, shard_id)
                status = True if processing succeeded, False otherwise
                shard_id = the same shard_id
        """
        if isinstance(shard_id, int):
            raise ValueError("shard_id as int is not supported anymore")
        # Check if the output already exists
        if self._check_output_exists(shard_id, output_dir):
            logger.info(f"Output for shard {shard_id} already exists, skipping...")
            output_shard_path = output_dir / f"{shard_id}.parquet"
            return True, (shard_id, output_shard_path)

        # If the output doesn't exist, proceed with processing
        return self.process_shard(shard_id, input_dir, output_dir, **kwargs)

    def process_shard(
        self, shard_id: int, input_dir: pathlib.Path, output_dir: pathlib.Path
    ):
        """
        This method should be overridden by subclasses to implement the actual processing.

        Args:
            shard_id (int): The ID of the shard.
            input_dir (pathlib.Path): Directory containing input shards.
            output_dir (pathlib.Path): Directory to save output shards.

        Returns:
            tuple(bool, int): (status, shard_id)
                status = True if processing succeeded, False otherwise
                shard_id = the same shard_id
        """
        raise NotImplementedError("Subclasses should implement this method.")


class StatementFormalizer(ShardProcessor):

    def __init__(self, model, openai_api_key, openai_api_base):
        super().__init__()
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.model = model

    def run_single_sample(self, sample, n_samples, temperature, max_tokens):
        """
        Handle the autoformalization of a single sample.
        Returns a dictionary with the formalizations for that sample.
        """
        try:
            # Create the chat messages
            prompt = get_user_prompt(
                sample["natural_language"],
                has_header=True,
                theorem_names=sample.get("theorem_names"),
                source=None,
                include_source=False,
            )
            messages = [
                {"role": "system", "content": constants.system_prompt},
                {"role": "user", "content": prompt},
            ]

            # Query OpenAI’s Chat Completion API
            response = query_with_retry(
                client=self.client,
                type="chat",
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n_samples,
            )

            # Extract all generated completions
            response_dict = response.to_dict()
            completions = [
                choice["message"]["content"] for choice in response_dict["choices"]
            ]

            return {
                "uuid": sample["uuid"],
                "problem": sample["problem"],
                "natural_language_statement": sample["natural_language"],
                "formalizations": completions,
            }
        except Exception as e:
            logger.error(f"Error processing sample {sample['uuid']}: {e}")
            return {
                "uuid": sample["uuid"],
                "problem": sample["problem"],
                "natural_language_statement": sample["natural_language"],
                "formalizations": [],
                "error": str(e),
            }

    def run(self, samples, n_samples, temperature, max_tokens, num_threads=4):
        """
        Run the formalization of the natural language samples in parallel.

        Args:
            samples (list): A list of dictionaries. Each dictionary should contain:
                - 'uuid': The statement ID.
                - 'natural_language_statement': The statement to formalize.
                - 'theorem_names': A list of theorem names, if any.
            n_samples (int): The number of autoformalizations (completions) to generate.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum tokens for output.
            num_threads (int): Number of worker threads to use.

        Returns:
            A list of dictionaries, each containing:
                - 'uuid'
                - 'natural_language'
                - 'formalizations': a list of generated formal statements (one per completion).
        """
        results = []

        # Create a pool of worker threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Schedule each sample’s formalization in a separate thread
            futures = [
                executor.submit(
                    self.run_single_sample, sample, n_samples, temperature, max_tokens
                )
                for sample in samples
            ]

            # Use tqdm to track progress as threads complete
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Formalizing statements"
            ):
                result = future.result()
                results.append(result)

        return results

    def process_shard(
        self,
        shard_id,
        input_dir,
        output_dir,
        n_samples,
        temperature,
        max_tokens,
        num_threads,
        add_negation=False,
        add_informal=False,
    ):
        """
        Process a single shard: read, formalize, and write the output.

        Args:
            shard_id (int): The ID of the shard.
            input_dir (pathlib.Path): Directory containing input shards.
            output_dir (pathlib.Path): Directory to save output shards.
            n_samples (int): Number of formalizations per sample.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum tokens for output.
            num_threads (int): Number of threads to use.
            add_negation (bool): Whether to add negations of the formalizations.
                If True, each formalization will be negated and saved as a separate statement.
                The number of statements will be doubled.
            add_informal (bool): Whether to add the informal problem as a comment before the formal statement.

        Returns:
            tuple: (status: bool, shard_id: int)
        """
        try:
            input_shard_path = input_dir / f"{shard_id}.parquet"
            output_shard_path = output_dir / f"{shard_id}.parquet"

            if not input_shard_path.exists():
                logger.error(f"Input shard {input_shard_path} does not exist.")
                return False, shard_id

            # Read the input shard
            df = pd.read_parquet(input_shard_path)
            # apply make_natural_language_prompt
            samples = df.to_dict(orient="records")
            for sample in samples:
                sample.update(make_natural_language_prompt(sample))

            # Run formalization
            formalized_results = self.run(
                samples=samples,
                n_samples=n_samples,
                temperature=temperature,
                max_tokens=max_tokens,
                num_threads=num_threads,
            )

            # Prepare DataFrame for output
            statements = []
            for result in formalized_results:
                for formalization in result.get("formalizations", []):
                    statement_id = str(uuid.uuid4())
                    if add_informal:
                        updated_formal = process_statement.insert_informal(
                            formalization, result["problem"]
                        )
                    else:
                        updated_formal = formalization
                    statements.append(
                        {
                            "uuid": result["uuid"],
                            "statement_id": statement_id,
                            "natural_language": result["natural_language_statement"],
                            "formal_statement": updated_formal,
                            "is_negation": False,
                        }
                    )
                    if add_negation:
                        negation = negate_theorem(formalization)
                        statements.append(
                            {
                                "uuid": result["uuid"],
                                "statement_id": "neg_" + statement_id,
                                "natural_language": result[
                                    "natural_language_statement"
                                ],
                                "formal_statement": negation,
                                "is_negation": True,
                            }
                        )

            if not statements:
                logger.warning(f"No formalizations generated for shard {shard_id}.")
                return False, (shard_id, None)

            statements_df = pd.DataFrame(statements)

            # Write to output shard
            statements_df.to_parquet(output_shard_path, index=False)
            logger.info(f"Shard {shard_id} formalized and saved to {output_shard_path}")
            return True, (shard_id, output_shard_path)
        except Exception as e:
            logger.error(f"Error processing shard {shard_id}: {e}")
            return False, (shard_id, None)


class StatementVerifier(ShardProcessor):
    def __init__(
        self,
        client_url: str,
        client_api_key: str = None,
        timeout: int = 60,
        num_threads: int = 5,
        batch_size: int = 100,
    ):
        """
        Args:
            client_url (str): URL of the Lean4 verification service.
            client_api_key (str, optional): API key for the verification service.
            output_path (str, optional): Directory path to write verification results.
            timeout (int, optional): Timeout in seconds for the entire verification call.
            num_threads (int, optional): Number of threads used within lean4_client.batch_verify_proof
                                         (if it supports parallelization).
        """
        super().__init__()
        self.client = lean4_client.Lean4Client(url=client_url, api_key=client_api_key)
        self.timeout = timeout
        self.num_threads = num_threads
        self.batch_size = batch_size

    def process_shard(self, shard_id, input_dir, output_dir):
        """
        Process a single shard: read, verify, and write the output.

        Args:
            shard_id (int): The ID of the shard.
            input_dir (str or pathlib.Path): Directory containing input shards.
            output_dir (str or pathlib.Path): Directory to save output shards.

        Returns:
            tuple(bool, int): (status, shard_id)
                status = True if verification succeeded, False otherwise
                shard_id = the same shard_id
        """
        try:
            input_dir = pathlib.Path(input_dir)
            output_dir = pathlib.Path(output_dir)

            input_shard_path = input_dir / f"{shard_id}.parquet"
            output_shard_path = output_dir / f"{shard_id}.parquet"

            if not input_shard_path.exists():
                logger.error(f"Input shard {input_shard_path} does not exist.")
                return False, shard_id

            # Read the input shard
            df = pd.read_parquet(input_shard_path)
            statements = df.to_dict(orient="records")

            # Prepare statements for batch verification
            # Expecting each record to have at least statement_id, uuid, formal_statement
            verification_samples = []
            for row in statements:
                verification_samples.append(
                    {
                        "proof_id": row["statement_id"],
                        "uuid": row["uuid"],
                        "proof": row["formal_statement"],
                    }
                )

            # Run verification in a single batch
            verification_results = lean4_client.batch_verify_proof(
                client=self.client,
                samples=verification_samples,
                timeout=self.timeout,
                num_threads=self.num_threads,
                batch_size=self.batch_size,
                working_dir=None,
            )

            # Convert results to DataFrame for saving
            verification_df = pd.DataFrame(verification_results)

            # rename columns to match the expected output
            verification_df = verification_df.rename(
                columns={"proof_id": "statement_id", "formal_proof": "formal_statement"}
            )
            # remove uuid
            verification_df = verification_df.drop(columns=["uuid"])

            # join the verification results with the original statements on statement_id
            verification_df = pd.merge(
                df, verification_df, on="statement_id", how="left"
            )

            # update the input shard with the verification results
            verification_df.to_parquet(input_shard_path, index=False)
            logger.info(
                f"Shard {shard_id} verified, results updated and saved to {input_shard_path}"
            )

            # filter the correct statements and save them to the output shard
            correct_statements = verification_df[verification_df["is_valid_with_sorry"]]
            correct_statements.to_parquet(output_shard_path, index=False)
            logger.info(
                f"Shard {shard_id} verified statements saved to {output_shard_path}"
            )

            return True, shard_id

        except Exception as e:
            logger.error(f"Error processing shard {shard_id}: {e}")
            return False, shard_id


def example_statement_formalizer(
    model: str,
    openai_api_key: str,
    openai_api_base: str,
    client_url: str,
    client_api_key: str = None,
):
    # load dataset
    aops_wiki_base = load_dataset("AI-MO/aops-wiki-base", split="train")
    aops_wiki_base = aops_wiki_base.map(make_natural_language_prompt)
    aops_wiki_base = aops_wiki_base.select(range(100))

    assert "natural_language" in aops_wiki_base.column_names
    assert "theorem_names" in aops_wiki_base.column_names
    assert "uuid" in aops_wiki_base.column_names

    woker = StatementFormalizer(
        model="AI-MO/Qwen7BCoder_Autoformalizer",
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
    )

    results = woker.run(
        samples=aops_wiki_base,
        n_samples=8,
        temperature=0.8,
        max_tokens=1024,
        num_threads=20,
    )

    statements = []
    for result in results:
        for formalization in result["formalizations"]:
            statements.append(
                {
                    "uuid": result["uuid"],
                    "statement_id": uuid.uuid4(),
                    "natural_language": result["natural_language_statement"],
                    "formal_statement": formalization,
                }
            )
    statements_df = pd.DataFrame(statements)
    print(statements_df.head())

    client = lean4_client.Lean4Client(
        url=client_url,
        api_key=client_api_key,
    )

    # make sure we have proof_id, uuid, and proof
    verification_samples = []
    for sample in tqdm(statements):
        verification_samples.append(
            {
                # for this dataset, proof_id is not unique, unfortunately
                "proof_id": sample["statement_id"],
                "uuid": sample["uuid"],
                "proof": sample["formal_statement"],
            }
        )
    batch_size = 100
    verification_results = lean4_client.batch_verify_proof(
        client=client,
        samples=verification_samples,
        timeout=60,
        num_threads=5,
        batch_size=batch_size,
        working_dir=None,
    )
    verification_df = pd.DataFrame(verification_results)
    verification_df = verification_df.rename(
        columns={"proof_id": "statement_id", "formal_proof": "formal_statement"}
    )
    logger.info(verification_df.head())

    # log the valid rate
    valid_rate = verification_df["is_valid_with_sorry"].mean()
    logger.info(f"Valid rate: {valid_rate:.3f}")


if __name__ == "__main__":
    """
    Usage:
    python -m autoformalizer.inference.workers \
        --model="AI-MO/Qwen7BCoder_Autoformalizer" \
        --openai_api_key="EMPTY" \
        --openai_api_base="http://localhost:8081/v1" \
        --client_url="https://kimina.saas.moonshot.cn/lean4-evaluator"

    You should something like
    2024-12-24 01:38:02.218 | INFO     | __main__:example_statement_formalizer:181 - Valid rate: 0.890
    """
    fire.Fire(example_statement_formalizer)
