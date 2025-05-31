import json
from collections import defaultdict

import fire
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm


def add_uuid_to_aops_formalization():
    base_ds = load_dataset("AI-MO/aops-base-v2.0", split="train")
    auto_ds = load_dataset("AI-MO/aops-autoformalization-v0.1", split="train")

    def add_uuid(sample):
        index = sample["index"]
        assert base_ds[index]["problem"] == sample["problem"]
        return {"uuid": base_ds[index]["uuid"]}

    auto_ds = auto_ds.map(add_uuid, num_proc=8)
    auto_ds.push_to_hub("AI-MO/aops-autoformalization-v0.1", private=True)


def add_uuid_to_inference_results():
    # list of datasets
    math_train = load_dataset("AI-MO/math-base-v1.0", split="train")
    math_test = load_dataset("AI-MO/math-base-v1.0", split="test")
    aops = load_dataset("AI-MO/aops-base-v2.0", split="train")

    # list of results
    math_train_results = load_dataset(
        "AI-MO/math-train-inference-results", split="train"
    )
    math_test_results = load_dataset("AI-MO/math-test-inference-results", split="train")
    aops_results = load_dataset("AI-MO/aops-inference-results", split="train")

    # build map
    ds_map = {
        "math-train": (math_train, math_train_results),
        "math-test": (math_test, math_test_results),
        "aops": (aops, aops_results),
    }

    def add_uuid(sample, ds):
        index = sample["problem_id"]
        assert ds[index]["problem"] == sample["problem"]
        return {"uuid": ds[index]["uuid"]}

    for key, (ds, results) in ds_map.items():
        results = results.map(lambda x: add_uuid(x, ds), num_proc=8)
        results.push_to_hub(f"AI-MO/{key}-inference-results", private=True)
        print(f"Pushed {key} inference results to hub")
        print(results)


def build_auto_proof_dataset():
    result_datasets = {
        "math-train": load_dataset("AI-MO/math-train-inference-results", split="train"),
        "math-test": load_dataset("AI-MO/math-test-inference-results", split="train"),
        "aops": load_dataset("AI-MO/aops-inference-results", split="train"),
    }
    print(result_datasets)

    # keep uuid, proof_id, feedback, is_valid
    # then concatenate all datasets
    proof_ds = []
    for key, ds in result_datasets.items():
        ds = ds.select_columns(["uuid", "proof", "proof_id", "feedback", "is_valid"])
        ds = ds.map(lambda x: {"source": key}, num_proc=8)
        proof_ds.append(ds)

    proof_ds = concatenate_datasets(proof_ds)
    print(proof_ds)
    proof_ds.push_to_hub("AI-MO/auto-proofs-v1", private=True)


def update_auto_proof_with_server_feedback():
    auto_proof_server = load_dataset("AI-MO/auto-proof-v1-server", split="train")
    valid_ds = auto_proof_server.filter(lambda x: x["is_valid_server"])
    print(f"Valid samples: {len(valid_ds)}")

    def _fix_valid(sample):
        is_valid = sample["is_valid_server"] & ("sorry" not in sample["proof"])
        return {
            "is_valid": is_valid,
            "feedback": sample["server_feedback"],
        }

    auto_proof_server = auto_proof_server.map(_fix_valid, num_proc=8)
    valid_ds = auto_proof_server.filter(lambda x: x["is_valid"])
    print(f"Valid samples after fix: {len(valid_ds)}")
    auto_proof_server.push_to_hub("AI-MO/auto-proofs-v1", private=True)


def build_auto_statement_dataset():
    auto_datasets = {
        "math-train": load_dataset("AI-MO/math-autoformalization-v0.1", split="train"),
        "math-test": load_dataset("AI-MO/math-autoformalization-v0.1", split="test"),
        "aops": load_dataset("AI-MO/aops-autoformalization-v0.1", split="train"),
    }
    print(auto_datasets)

    # keep column uuid, problem, answer, problem_type, question_type, theorem_names, natural_language, autoformalization
    # then concatenate all datasets
    auto_ds = []
    for key, ds in auto_datasets.items():
        ds = ds.map(lambda x: {"source": key})
        ds = ds.select_columns(
            [
                "uuid",
                "source",
                "problem",
                "answer",
                "problem_type",
                "question_type",
                "theorem_names",
                "natural_language",
                "autoformalization",
            ]
        )
        ds = ds.rename_column("natural_language", "natural_language_statement")
        ds = ds.rename_column("autoformalization", "formal_statement")
        # add key as source
        auto_ds.append(ds)

    auto_ds = concatenate_datasets(auto_ds)
    print(auto_ds)

    # print number of unique uuid
    print(len(auto_ds.unique("uuid")))

    auto_proof = load_dataset("AI-MO/auto-proofs-v1", split="train")
    uuid_proofs = defaultdict(list)
    for sample in tqdm(auto_proof):
        uuid_proofs[sample["uuid"]].append(
            {
                "formal_proof": sample["proof"],
                "is_valid": sample["is_valid"],
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

    auto_ds = auto_ds.map(add_proofs)
    auto_ds.push_to_hub("AI-MO/auto-statements-v1", private=True)


if __name__ == "__main__":
    fire.Fire(
        {
            "add_uuid_to_aops_formalization": add_uuid_to_aops_formalization,
            "add_uuid_to_inference_results": add_uuid_to_inference_results,
            "build_auto_proof_dataset": build_auto_proof_dataset,
            "update_auto_proof_with_server_feedback": update_auto_proof_with_server_feedback,
            "build_auto_statement_dataset": build_auto_statement_dataset,
        }
    )
