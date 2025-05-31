from copy import copy

import datasets
from datasets import load_dataset


def format_goals_with_indention(goals, indention):
    goals = goals.split("\n")
    goals = [indention + goal for goal in goals]
    return "\n".join(goals)


def construct_prompts(tactics, proof_input):
    # if proof_input.strip().endswith("by"):
    #     start = proof_input.strip()[:-len("by")]
    # else:
    #     start = proof_input
    start = proof_input.rstrip()

    if "by" in tactics[0]["tactic"]:
        tactic_text = tactics[0]["tactic"]
        tactic_texts = tactic_text.split("\n")
        while tactic_texts[0].strip() == "" or tactic_texts[0].strip() == "by":
            tactic_texts = tactic_texts[1:]
        tactics[0]["tactic"] = "\n" + "\n".join(tactic_texts)

    results = []
    for j in range(len(tactics)):
        current_proof = start + "".join([t["tactic"] for t in tactics[:j]])
        input = "Complete the following Lean 4 code:\n\n```lean4\n" + current_proof
        indention = " " * 2
        input += f"\n{indention}/- tactic state:"
        for goal in tactics[j]["goalsBefore"]:
            format_goals = format_goals_with_indention(goal, indention + " " * 2)
            input += f"\n{format_goals}"
        input += f"\n{indention}-/\n"
        output = "".join([t["tactic"] for t in tactics[j:]])
        output += "\n```"

        output = output.split("\n")
        while output[0].strip() == "":
            output = output[1:]
        output = "\n".join(output)
        # code_prefix = input[
        #     len(
        #         "Complete the following Lean 4 code with explanatory"
        #         + " comments preceding each line of code:\n\n```lean4\n"
        #     ) :
        # ]
        results.append([input, output])

    return results


if __name__ == "__main__":

    dataset = load_dataset(
        "AI-MO/step-proof-stage1-pt-250119-v3_minhash-n3",
        split="train",
        cache_dir="/mnt/moonfs/wanghaiming-m2/.cache/ttmmpp",
    )

    final_results = []
    for idx, sample in enumerate(dataset):
        res = construct_prompts(sample["tactics"], sample["proof_input"])
        for input, output in res:
            new_sample = copy(sample)
            new_sample["proof_input"] = input
            new_sample["proof_output"] = output
            new_sample["text"] = input + output
            final_results.append(new_sample)

        # if idx == 10:
        #     break
    print("Number of sample", len(dataset))
    print("Number of step sample", len(final_results))

    new_dataset = datasets.Dataset.from_list(final_results)
    new_dataset.push_to_hub(
        "AI-MO/step-proof-stage2-pt-250119-v3_minhash-n3-step-only", private=True
    )
