import concurrent.futures
from typing import List

import fire
import pandas as pd
from datasets import load_dataset
from numinamath.openai_utilities.batch_manager import run_completion_with_batch_api
from openai import OpenAI

from autoformalizer.eval_utils.constants import gpt_verification_system_prompt

client = OpenAI()


def parse_formalization_status(response: str) -> str:
    # Split the response into lines
    lines = response.splitlines()

    formalization_str = "Formalization Status:"

    # Iterate over lines to find the "Formalization Status:" part
    for i in range(1, len(lines) + 1):
        line = lines[len(lines) - i]
        if formalization_str in line:
            # Extract the status after "Formalization Status:"
            answer = line.split("Formalization Status:")[1].strip()
            if "Correct" in answer:
                return "Correct"
            elif "Incorrect" in answer:
                return "Incorrect"

    # If "Formalization Status:" is not found, return a default value
    return "Status not found"


def gpt_feedback(lean4_code: str, natural_language: str, model_name: str):

    # Prepare the user prompt combining the natural language problem and its formalization
    prompt = f"Natural Language Statement:\n{natural_language}\n\nLean 4 Statement:\n{lean4_code}\n"

    # Create messages with system prompt and user prompt
    messages = [
        {"role": "system", "content": gpt_verification_system_prompt},
        {"role": "user", "content": prompt},
    ]

    # Call the OpenAI API for completion
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
    )

    # Extract the response from the model
    response = completion.choices[0].message.content

    return response


def parallel_gpt_feedback(
    lean4_codes: List[str],
    natural_language_ls: List[str],
    model_name: str = "gpt-4o",
    num_workers: int = 60,
):

    def gpt_feedback_helper(args):
        lean4_code, natural_language, model_name = args
        return gpt_feedback(lean4_code, natural_language, model_name)

    # print(f"Running GPT-4o feedback for {len(lean4_codes)} formalizations...")

    inputs = zip(lean4_codes, natural_language_ls, [model_name] * len(lean4_codes))

    # Use ThreadPoolExecutor to parallelize the calls to gpt_feedback, with optional control over num_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks and get results
        results = list(executor.map(gpt_feedback_helper, inputs))

    return results


def batch_gpt_feedback(
    lean4_codes: List[str],
    natural_language_ls: List[str],
    model_name: str = "gpt-4o",
    cache_dir: str = None,
    max_tokens: int = 1024,
):
    """
    Batch version of the GPT feedback function.
    """

    # Prepare the user prompts combining the natural language problems and their formalizations
    prompts = []
    for lean4_code, natural_language in zip(lean4_codes, natural_language_ls):
        prompt = f"Natural Language Statement:\n{natural_language}\n\nLean 4 Statement:\n{lean4_code}\n"
        prompts.append(prompt)

    # Create messages with system prompt and user prompts
    messages_list = []
    for prompt in prompts:
        messages_list.append(
            [
                {"role": "system", "content": gpt_verification_system_prompt},
                {"role": "user", "content": prompt},
            ]
        )

    # Call run_completion_with_batch_api
    completions = run_completion_with_batch_api(
        messages_list,
        model_name=model_name,
        cache_dir=cache_dir,
        max_tokens=max_tokens,
    )
    assert len(completions) == len(lean4_codes)
    results = [
        x["response"]["body"]["choices"][0]["message"]["content"] for x in completions
    ]
    return results


def dataframe_gpt_feedback(
    input_csv: str, output_csv: str, model_name: str = "gpt-4o", verbose: bool = True
):
    """
    Runs verification with GPT-4o for each formalization in a given DataFrame.

    Args:
    - input_csv: Path to the input CSV file.
    - output_csv: Path to the output CSV file.
    - model_name: The name of the GPT model to use.
    - verbose: Whether to print results.

    Requirements:
    The input CSV file must contain the following columns:
    - 'natural_language': The natural language statement to formalize.
    - 'autoformalization_i': The i-th autoformalization to get feedback for.

    The output CSV file will contain the following columns:
    - 'gpt_feedback_i': The feedback for the i-th autoformalization.
    - 'gpt_feedback_i_status': The status of the i-th autoformalization. (True/False)
    """

    df = pd.read_csv(input_csv)
    natural_language_ls = df["natural_language"].tolist()

    for column in df.columns:
        if str(column).startswith("autoformalization_"):
            idx = column.split("_")[-1]
            lean4_codes = df[column].tolist()
            feedbacks = parallel_gpt_feedback(
                lean4_codes, natural_language_ls, model_name=model_name
            )

            df[f"gpt_feedback_{idx}"] = [str(feedback) for feedback in feedbacks]
            df[f"gpt_feedback_{idx}_bool"] = [
                True if parse_formalization_status(feedback) == "Correct" else False
                for feedback in feedbacks
            ]

            if verbose:
                val_counts = df[f"gpt_feedback_{idx}_bool"].value_counts()
                print(f"Feedback for autoformalization_{idx}:")
                print(val_counts)
                print(val_counts / val_counts.sum())

    df.to_csv(output_csv, index=False)

    if verbose:
        N = len(df)
        df = df[df["compiler_feedback_1_bool"]]
        df = df[df["gpt_feedback_1_bool"]]
        print(
            f"Percentage of (True, True) autoformalizations: {len(df)}/{N} = {len(df)/N:.2f}"
        )

    return df


def hf_gpt_feedback(
    input_dataset_id: str,
    input_dataset_branch: str,
    output_dataset_id: str,
    output_dataset_branch: str,
    filter_compiled: bool = False,
    model_name: str = "gpt-4o",
    verbose: bool = True,
):
    """
    Runs verification with GPT-4o for each formalization in a given HuggingFace Dataset.

    Args:
    - input_dataset_id: The ID of the input HuggingFace Dataset.
    - input_dataset_branch: The branch of the input HuggingFace Dataset.
    - output_dataset_id: The ID of the output HuggingFace Dataset.
    - output_dataset_branch: The branch of the output HuggingFace Dataset.
    - filter_compiled: Whether to filter for formalizations that have positive compiler feedback.
    - model_name: The name of the GPT model to use.
    - verbose: Whether to print results.

    Requirements:
    The input dataset must contain the following columns:
    - 'natural_language': The natural language statement to formalize.
    - 'autoformalization_i': The i-th autoformalization to get feedback for.
    - If filter_compiled is True, the dataset must contain the following columns:
        - 'compiler_feedback_i_bool': The boolean value indicating whether the i-th autoformalization compiled.

    The output dataset file will contain the following columns:
    - 'gpt_feedback_i': The feedback for the i-th autoformalization.
    - 'gpt_feedback_i_bool': The status of the i-th autoformalization. (True/False)

    If filter_compiled is True, the output dataset will only contain formalizations
    that have positive compiler feedback.

    The output dataset will be pushed to the HuggingFace Hub.
    """

    ds = load_dataset(input_dataset_id, split="train", revision=input_dataset_branch)

    natural_language_ls = ds["natural_language"]

    autof_columns = [
        column
        for column in ds.column_names
        if str(column).startswith("autoformalization_")
    ]

    for column in autof_columns:

        idx = column.split("_")[-1]

        # Filter for formalizations that have positive compiler feedback
        if filter_compiled:
            compiler_feedback_column = f"compiler_feedback_{idx}_bool"
            # hold a list of indices of the rows that have positive compiler feedback
            compiled_ids = [i for i, x in enumerate(ds[compiler_feedback_column]) if x]
        else:
            compiled_ids = range(len(ds))

        lean4_codes = [ds[column][i] for i in compiled_ids]
        filtered_natural_language_ls = [natural_language_ls[i] for i in compiled_ids]

        feedbacks = parallel_gpt_feedback(
            lean4_codes, filtered_natural_language_ls, model_name=model_name
        )
        feedbacks = [str(feedback) for feedback in feedbacks]
        feedback_bools = [
            parse_formalization_status(feedback) == "Correct" for feedback in feedbacks
        ]

        if verbose:
            # Print the percentage of correct formalizations
            true_count = sum(feedback_bools)
            if filter_compiled:
                print(f"Filtered for {len(feedbacks)} autoformalizations that compile.")

            print(
                f"{model_name} sucess rate for autoformalization_{idx}: {true_count/len(ds):.3%}."
            )

        # Add the feedback columns to the dataset
        full_feedbacks = [""] * len(ds)
        full_feedback_bools = [None] * len(ds)
        for j in range(len(compiled_ids)):
            full_feedbacks[compiled_ids[j]] = feedbacks[j]
            full_feedback_bools[compiled_ids[j]] = feedback_bools[j]

        if f"{model_name}_feedback_{idx}" in ds.column_names:
            ds = ds.remove_columns(f"{model_name}_feedback_{idx}")

        if f"{model_name}_feedback_{idx}_bool" in ds.column_names:
            ds = ds.remove_columns(f"{model_name}_feedback_{idx}_bool")

        ds = ds.add_column(f"{model_name}_feedback_{idx}", full_feedbacks)
        ds = ds.add_column(f"{model_name}_feedback_{idx}_bool", full_feedback_bools)

    ds.push_to_hub(
        output_dataset_id,
        revision=output_dataset_branch,
        private=True,
        commit_message=f"Feedback for autoformalizations using {model_name} model.",
    )


if __name__ == "__main__":
    """
    Example usage:
    python -m autoformalizer.eval_utils.gpt_feedback \
    dataframe_gpt_feedback \
    --input_csv="/home/mert/autoformalizer/scripts/mert/Autof181024_feedback.csv" \
    --output_csv="/home/mert/autoformalizer/scripts/mert/Autof181024_gpt.csv"

    python -m autoformalizer.eval_utils.gpt_feedback \
    hf_gpt_feedback \
    --input_dataset_id="AI-MO/AutoformalizationEvalV2" \
    --input_dataset_branch="Qwen7BCoder_AutoformalizerV1B3" \
    --output_dataset_id="AI-MO/AutoformalizationEvalV2" \
    --output_dataset_branch="Qwen7BCoder_AutoformalizerV1B3" \
    --filter_compiled=True

    """

    fire.Fire(
        {
            "dataframe_gpt_feedback": dataframe_gpt_feedback,
            "hf_gpt_feedback": hf_gpt_feedback,
        }
    )
