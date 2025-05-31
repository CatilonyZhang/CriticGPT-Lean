from datasets import load_dataset

from autoformalizer.eval_utils.lean_feedback import parallel_lean4_feedback


def build_lean4_steps(steps):
    """
    Constructs Lean 4 code from a list of steps including comment, position and tactic.
    Returns:
        str: The generated Lean 4 code as a string.
    """

    # Keep the last tactic in case of closing > 1 goals with a single tactic
    last_tactic = ""
    # Keeps a list of lines of code
    curr_code = []
    last_line = -1

    for i, step in enumerate(steps):
        line_start = step["position"]["start"]["line"]
        col_start = step["position"]["start"]["column"]

        if step["comment"] is not None:
            # Comments induce a new line
            curr_code += step["comment"].split("\n")

        # Add separator between tactics
        if line_start > last_line:
            curr_code.append("")
            last_line = step["position"]["finish"]["line"]
        elif last_tactic != "\u00B7" and last_tactic != ".":
            curr_code[-1] += ";"

        # Handle multiple tactics in a single step
        tactic_lines = step["tactic"].strip().split("\n")

        for i, tactic in enumerate(tactic_lines):
            if col_start >= len(curr_code[-1]):
                # Correcting number of leading spaces:
                curr_code[-1] += " " * (col_start - len(curr_code[-1]))
            elif step["tactic"] == last_tactic:
                # Special case for closing > 1 goals with a single tactic
                curr_code.append((" " * (col_start)))
            else:
                # If conflicting with previous tactic due to different parsing,
                # add white space and concatenate after
                curr_code[-1] += " "

            curr_code[-1] += tactic
            if i != len(tactic_lines) - 1:
                curr_code.append("")

        last_tactic = step["tactic"]

    return "\n".join(curr_code)


def build_lean4_wholeproof(steps):
    """
    Constructs Lean 4 code from a list of steps for wholeproof
    Returns:
        str: The generated Lean 4 code as a string.
    """
    curr_code = ""
    for step in steps:
        if step["comment"] is not None:
            curr_code += step["comment"]
            if not step["comment"].strip().endswith("\n"):
                curr_node += "\n"
        curr_code += step["tactic"]
    return curr_code


def get_sft_wholeproof(example, ignore_context=False):
    if ignore_context:
        header = example["header"] + example["statement"]
    else:
        header = example["header"] + example["context"] + example["statement"]
    # remove the sorry from the end
    header = header.split("sorry")[0] + "\n"
    formal_statement = header + build_lean4_wholeproof(example["steps"])
    return formal_statement


def get_sft_steps(example, ignore_context=False):
    sft_datas = []
    if ignore_context:
        header = example["header"] + example["statement"]
    else:
        header = example["header"] + example["context"] + example["statement"]
    # remove the sorry from the end
    header = header.split("sorry")[0] + "\n"

    for i, tactic_snap in enumerate(example["steps"]):
        state = "\n".join(tactic_snap["goals_before"])

        proof_before = header + build_lean4_steps(example["steps"][:i])
        proof_after = build_lean4_steps(example["steps"][i:])

        sft_data = f"STATE:\n{state.strip()}\n"
        sft_data += f"\nPROOF_BEFORE:\n{proof_before}\nPROOF_AFTER:\n{proof_after}"
        sft_datas.append(sft_data)

    return sft_datas


def generate_prover_data(
    dataset_id: str,
    dataset_branch: str,
    output_dataset_id: str,
    output_dataset_branch: str,
    sft_data_column: str = "whole_proof",
    ignore_context: bool = False,
    dataset_split: str = "train",
):
    """
    Generates fine-tuning data for the prover from step-by-step lean4 proofs.

    Args:
        dataset_id (str): The ID of the dataset.
        dataset_branch (str): The branch of the dataset.
        output_dataset_id (str): The ID of the output dataset
        output_dataset_branch (str): The branch of the output dataset.
        sft_data_column (str): The column name to add the obtained sft data. Default is 'whole_proof'
        dataset_split (str): The split of the dataset. Default is 'train'.

    Returns:
        None

    Results are saved to the dataset in the Hub.

    Format of the prover data:

    STATE:
    {current state}
    PROOF_BEFORE:
    {current proof}
    PROOF_AFTER:
    {proof after the current step, completion until the end of the proof}
    """

    # Load the dataset
    ds = load_dataset(dataset_id, revision=dataset_branch, split=dataset_split)

    if sft_data_column not in ds.column_names:
        ds = ds.add_column(sft_data_column, [[]] * len(ds))

    def add_prover_data(example):
        """
        Add prover data to the example.
        """
        sft_datas = get_sft_wholeproof(example, ignore_context)
        example[sft_data_column] = sft_datas
        example["num_" + sft_data_column] = len(sft_datas)
        return example

    # Add prover data to the dataset
    ds = ds.map(add_prover_data, num_proc=42)

    # Save the dataset
    ds.push_to_hub(output_dataset_id, revision=output_dataset_branch)


if __name__ == "__main__":
    generate_prover_data(
        "AI-MO/sft-mathlib4-whole",
        "main",
        "AI-MO/sft-mathlib4-whole",
        "main",
        ignore_context=True,
    )
