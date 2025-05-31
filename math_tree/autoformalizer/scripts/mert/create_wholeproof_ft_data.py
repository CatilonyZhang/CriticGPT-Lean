from typing import List

from datasets import load_dataset


def create_wholeproof_ft_data(
    dataset_id: str,
    columns: List[str],
    output_dataset_id: str,
    dataset_split: str = "train",
    mode: str = "pt",
    prompt_template: str = "",
    input_prompt_template: str = "",
    output_prompt_template: str = "",
    system_prompt_template: str = "",
):
    """
    Generates wholeproof finetuning data from a HF dataset with formal proofs using prompt templates.

    Args:
        dataset_id (str): The ID of the dataset to be used.
        columns (str): Columns to use to format the prompt templates.
        output_dataset_id (str): The ID of the output dataset.
        dataset_split (str): The split of the dataset to be used.
        mode (str): The mode of dataset generation.
        prompt_template (str): The template for the prompt in the pt data.
        input_prompt_template (str): The template for the input prompt in the ft data.
        output_prompt_template (str): The template for the output prompt in the ft data.
        system_prompt_template (str): The template for the system prompt in the ft data.

    There are 2 modes of dataset generation:
    - "ft":
    Generates question-answer pairs in Alpaca format. The input and output are formatted according to
    `input_prompt_template` and `output_prompt_template`.
    - "pt":
    Generates one field "text" formatted according to `prompt_template` used to be trained in "pretraining" format.
    """

    ds = load_dataset(dataset_id, split=dataset_split)

    # filter for relevant columns
    ds = ds.remove_columns([c for c in ds.column_names if c not in columns])

    if mode not in ["ft", "pt"]:
        raise ValueError("Invalid mode.")

    def map_to_text(example):

        if mode == "ft":
            system = system_prompt_template.format(**example)
            instruction = input_prompt_template.format(**example)
            output = output_prompt_template.format(**example)

            example["system"] = system
            example["instruction"] = instruction
            example["output"] = output
            example["input"] = ""

        elif mode == "pt":

            text = prompt_template.format(**example)
            example["text"] = text

        return example

    ds = ds.map(map_to_text, num_proc=42)

    # only keep relevant column
    if mode == "ft":
        cols = ["system", "instruction", "output", "input"]
    elif mode == "pt":
        cols = ["text"]

    remove_cols = [c for c in ds.column_names if c not in cols]
    ds = ds.remove_columns(remove_cols)

    # Save the wholeproof data
    ds.push_to_hub(output_dataset_id, private=True)


if __name__ == "__main__":
    # example of creating PT style data
    create_wholeproof_ft_data(
        dataset_id="AI-MO/wholeproof-data-beta1pick10-250107",
        columns=["proof"],
        prompt_template="Complete the following Lean 4 code:\n\n```lean4\n{proof}\n```",
        output_dataset_id="AI-MO/wholeproof-data-pt-beta1pick10-250107",
        dataset_split="train",
        mode="pt",
    )

    # example of creating SFT style data in Alpaca format
    """
    create_wholeproof_ft_data(
        dataset_id="AI-MO/wholeproof-data-241231",
        columns=["proof_input", "proof"],
        input_prompt_template="Complete the following Lean 4 code:\n\n```lean4\n{proof_input}\n```",
        output_prompt_template="Here is the completed Lean 4 code:\n\n```lean4\n{proof}\n```",
        system_prompt_template="You are an expert theorem prover in Lean 4.",
        output_dataset_id="AI-MO/wholeproof-data-ft-241231",
        dataset_split="train",
        mode="ft",
    )
    """
