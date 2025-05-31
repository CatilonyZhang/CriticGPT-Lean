import json
from collections import defaultdict

import fire
from datasets import load_dataset
from tqdm import tqdm


def build_auto_statements_v3():
    formal_problems = load_dataset("AI-MO/FormalProblemsV0", split="train")
    filtered_dataset = load_dataset(
        "AI-MO/autoformalization-olympiads-v0.1-filtered", split="train"
    )
    formal_natural = set(formal_problems["natural_language"])
    filtered_natural = set(filtered_dataset["natural_language"])

    # test if subset
    is_subset = formal_natural.issubset(filtered_natural)
    assert (
        is_subset
    ), "FormalProblemsV0 is not a subset of autoformalization-olympiads-v0.1-filtered"

    natural_to_problem = {
        n: i for i, n in enumerate(filtered_dataset["natural_language"])
    }

    # complete formal_problem with
    # uuid, problem, answer
    def _process(sample):
        natural_language = sample["natural_language"]
        index = natural_to_problem[natural_language]
        original_sample = filtered_dataset[index]
        uuid = original_sample["uuid"]
        assert natural_language == original_sample["natural_language"]
        problem = original_sample["problem"]
        answer = original_sample["answer"]
        return {
            "uuid": uuid,
            "problem": problem,
            "answer": answer,
            "source": "olympiads",
            "natural_language_statement": natural_language,
            "formal_statement": sample["lean_code"],
        }

    formal_problems = formal_problems.map(_process)
    formal_problems.push_to_hub("AI-MO/auto-statements-v3", private=True)

    formal_proofs = load_dataset("AI-MO/formal-problems-v0-proofs", split="train")

    def _update_uuid(sample):
        problem_index = int(sample["problem_id"])
        uuid = formal_problems[problem_index]["uuid"]
        return {
            "uuid": uuid,
            "problem": formal_problems[problem_index]["problem"],
        }

    formal_proofs = formal_proofs.map(_update_uuid, num_proc=10)
    formal_proofs.push_to_hub("AI-MO/auto-proofs-v3", private=True)


def add_proofs_to_auto_statements_v3():
    auto_statements = load_dataset("AI-MO/auto-statements-v3", split="train")
    auto_proofs = load_dataset("AI-MO/auto-proofs-v3", split="train")

    uuid_proofs = defaultdict(list)
    for sample in tqdm(auto_proofs):
        uuid_proofs[sample["uuid"]].append(
            {
                "formal_proof": sample["proof"],
                "is_valid": sample["is_valid_server"],
            }
        )

    def add_proofs(sample):
        proofs = uuid_proofs[sample["uuid"]]
        n_correct_proofs = sum(p["is_valid"] for p in proofs)
        n_proofs = len(proofs)
        # get the first correct proof or ""
        correct_proof = next((p["formal_proof"] for p in proofs if p["is_valid"]), "")
        return {
            "n_proofs": n_proofs,
            "n_correct_proofs": n_correct_proofs,
            "formal_proof": correct_proof,
            "correct_proof_samples": json.dumps(
                [p["formal_proof"] for p in proofs if p["is_valid"]]
            ),
        }

    auto_statements = auto_statements.map(add_proofs)
    auto_statements = auto_statements.select_columns(
        [
            "uuid",
            "problem",
            "answer",
            "source",
            "natural_language_statement",
            "formal_statement",
            "n_proofs",
            "n_correct_proofs",
            "formal_proof",
            "correct_proof_samples",
        ]
    )
    auto_statements.push_to_hub("AI-MO/auto-statements-v3", private=True)


if __name__ == "__main__":
    """
    python numinamath/dataset_manager/build_auto_v3.py build_auto_statements_v3

    python numinamath/dataset_manager/build_auto_v3.py add_proofs_to_auto_statements_v3
    """
    fire.Fire(
        {
            "build_auto_statements_v3": build_auto_statements_v3,
            "add_proofs_to_auto_statements_v3": add_proofs_to_auto_statements_v3,
        }
    )
