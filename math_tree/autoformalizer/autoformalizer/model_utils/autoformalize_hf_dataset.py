import os
import pathlib
import sys

import fire
from datasets import Dataset, load_dataset
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from autoformalizer.clients import lean4_client
from autoformalizer.data_utils import constants
from autoformalizer.data_utils.user_prompt import get_user_prompt
from autoformalizer.model_utils.autoformalize import autoformalize_dataframe_batched
from autoformalizer.model_utils.infer_hf_dataset import infer_hf_dataset_by_batch


def autoformalize_hf_dataset(
    model_path: str,
    dataset_id: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    max_tokens: int = 1024,
    batch_size: int = 1024,
):
    """
    Autoformalizes a HF dataset using an LLM model using greedy sampling.

    Requirements:
    The input dataset must contain the following columns:
    - 'natural_language': The natural language statement to formalize.
    - 'theorem_names': A list of theorem names to be used in autoformalization.
    - 'include_source': A boolean value indicating whether to include the source in the prompt.
    - 'has_header': A boolean value indicating whether to include a header in the autoformalization.
    - 'source': The source of the natural language statement. (optional)

    The results will be uploaded to the Hugging Face hub under the same dataset ID in column 'autoformalization'.

    """
    ds = load_dataset(dataset_id, split="train")

    df = ds.to_pandas()

    llm = LLM(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
    )

    df = autoformalize_dataframe_batched(
        llm, tokenizer, sampling_params, df, batch_size=batch_size
    )

    autoformalizations = df["autoformalization_1"]

    if "autoformalization" in ds.column_names:
        ds = ds.remove_columns("autoformalization")
    # add this to ds
    ds = ds.add_column("autoformalization", autoformalizations)

    ds.push_to_hub(
        dataset_id,
        private=True,
        commit_message=f"Autoformalized using LLM model {model_path}",
    )


def sample_autoformalisation(
    model_path: str,
    dataset_id: str,
    n_samples: int = 8,
    temperature: float = 0.8,
    max_tokens: int = 1024,
    batch_size: int = 1024,
    push_to_hub: bool = False,
    output_dataset_id: str = None,
    dry_run: bool = False,
):
    """
    Sample autoformalisation given a HF dataset.
    The dataset should contain a field 'natural_language' with the natural language statement to formalize.

    Args:
        - model_path: The path to the LLM model.
        - dataset_id: The HF dataset ID.
        - n_samples: The number of autoformalizations to sample.
        - temperature: The sampling temperature.
        - max_tokens: The maximum number of tokens to generate.
        - batch_size: The batch size for inference.
        - push_to_hub: Whether to push the dataset to the Hugging Face hub.
        - output_dataset_id: The output dataset ID on the Hugging Face hub.
        - dry_run: Whether to run a dry run with a smaller dataset.

    Returns:
        The HF dataset with the sampled autoformalizations.
        The sampled autoformalizations will be added to the dataset under the column 'autoformalization_samples'.
        autoformalization_samples is a list of dict with:
            - 'code': the autoformalization code
            - 'lean4_feedback_status': a boolean indicating whether the autoformalization compiled successfully
        The best autoformalization will be added to the dataset under the column 'autoformalization'.
    """
    # allow to pass the dataset
    if type(dataset_id) is str:
        ds = load_dataset(dataset_id)
    else:
        ds = dataset_id

    print(ds)
    if dry_run:
        if type(ds) is Dataset:
            ds = ds.select(range(50))
        else:
            for key in ds.keys():
                ds[key] = ds[key].select(range(50))

    print(f"\nSampling {n_samples} autoformalizations from {dataset_id}\n")

    llm = LLM(
        model_path,
        download_dir=f"{os.getenv('HOME')}/.cache/vllm/",
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_params = SamplingParams(
        n=n_samples, temperature=temperature, max_tokens=max_tokens
    )

    # create the chat messages
    def create_messages(sample):
        prompt = get_user_prompt(
            sample["natural_language"],
            has_header=True,
            theorem_names=sample["theorem_names"],
            source=None,
            include_source=False,
        )
        return {
            "messages": [
                {"role": "system", "content": constants.system_prompt},
                {"role": "user", "content": prompt},
            ]
        }

    ds = ds.map(create_messages)

    ds = infer_hf_dataset_by_batch(
        llm,
        tokenizer,
        sampling_params,
        ds,
        batch_size=batch_size,
        output_column_name="autoformalizations",
    )

    def _process(sample):
        autoformalization_samples = []
        for code in sample["autoformalizations"]:
            autoformalization_samples.append({"code": code})
        return {"auto_statements_candidates": autoformalization_samples}

    ds = ds.map(_process, num_proc=32)
    ds = ds.remove_columns("autoformalizations")

    if push_to_hub:
        ds.push_to_hub(
            output_dataset_id,
            private=True,
        )
    return ds


def mini_batch_statement_feedback(
    dataset_or_id: str,
    working_dir: str,
    num_proc: int = 100,
):
    """
    Verify the formal statements in a dataset using Lean4 Client.
    See parallel_verify_proof in autoformalizer.clients.lean4_client for more details
    about how the verification is done.

    Args:
        - dataset_or_id: The dataset or dataset ID to verify.
            The dataset must contain the following columns:
            - 'uuid': The UUID of the problem.
            - 'formal_statement': The formal statement to verify, it is supposed to contain sorry
            - 'statement_id': unique ID for the formal statement. It is simply in the form of
                f"{uuid}_{index}" where index is the index of the formal statement during the genration.
        - working_dir: The working directory to store the Lean4 files.
        - num_proc: The number of processes to use for parallel verification.

    Returns:
        The dataset with the following columns added, see parallel_verify_proof:
            - 'lean_feedback': The Lean feedback for the formal statement.
            - 'has_error': A boolean indicating whether the formal statement has an error.
    """
    client = lean4_client.Lean4Client(
        "https://kimina.saas.moonshot.cn/lean4-evaluator",
        api_key=os.environ.get("MOONSHOT_LEAN4_API_KEY"),
    )
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    dataset = load_dataset(dataset_or_id, split="train")
    batch_size = 5000
    working_dir = pathlib.Path(working_dir)

    statement_id_dict = {}
    for offset in range(0, len(dataset), batch_size):
        samples = []
        logger.info(f"Processing problems {offset} to {offset + batch_size}")
        end_batch = min(offset + batch_size, len(dataset))
        for sample in tqdm(dataset.select(range(offset, end_batch))):
            # convert the column name to the one used in the lean4_client
            samples.append(
                {
                    "uuid": sample["uuid"],
                    "proof": sample["formal_statement"],
                    "proof_id": sample["statement_id"],
                }
            )
        results = lean4_client.parallel_verify_proof(
            client, samples, timeout=60, num_proc=num_proc, working_dir=working_dir
        )
        for result in results:
            statement_id_dict[result["proof_id"]] = result
        # log the complilation status
        total_proofs = len(samples)
        no_error_proofs = sum(not result["has_error"] for result in results)
        logger.info(f"Total proofs to verify: {total_proofs}")
        logger.info(f"No error proofs: {no_error_proofs}")

    def _update_dataset(sample):
        statement_id = sample["statement_id"]
        feedback = statement_id_dict[statement_id]["lean_feedback"]
        has_error = statement_id_dict[statement_id]["has_error"]
        return {
            "lean_feedback": feedback,
            "has_error": has_error,
        }

    dataset = dataset.map(_update_dataset)
    return dataset


if __name__ == "__main__":
    """
    Example usage:
    python -m autoformalizer.model_utils.autoformalize_hf_dataset autoformalize_hf_dataset \
        --model_path="AI-MO/Qwen7BCoder_AutoformalizerV1B3" \
        --dataset_id="AI-MO/cnk_12_algebra_500"

    python -m autoformalizer.model_utils.autoformalize_hf_dataset sample_autoformalisation \
        --model_path="AI-MO/Qwen7BCoder_AutoformalizerV1B3" \
        --dataset_id="AI-MO/math-autoformlization-v0.1" \
        --batch_size=512 \
        --push_to_hub=True \
        --output_dataset_id="AI-MO/_math-autoformalization" \
        --dry_run=True
    """
    fire.Fire(
        {
            "autoformalize_hf_dataset": autoformalize_hf_dataset,
            "sample_autoformalisation": sample_autoformalisation,
        }
    )
