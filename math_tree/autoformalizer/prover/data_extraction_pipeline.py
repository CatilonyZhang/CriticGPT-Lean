import json
import re
import uuid

import datasets
from tqdm import tqdm

from autoformalizer.clients.infotree.process_infotree import extract_data
from autoformalizer.clients.lean4_client import Lean4Client, batch_verify_proof
from autoformalizer.repl_lean_feedback.repl_utils import split_proof_header


def extract_indentation(s):
    match = re.search(r"by\n(\s*)", s)
    return match.group(1) if match else ""


def split_on_first_by(s):
    i = s.find("by")
    if i == -1:
        return (s, "")
    return (s[:i], s[i:])


def removes_skip(s):
    if s.endswith("\n  skip"):
        return s[: -len("\n  skip")]
    return s


def extraction_is_correct(steps, original_theorem):
    # original_theorem doesn't contain the imports!
    statement, proof = split_on_first_by(original_theorem)
    proof = removes_skip(proof)
    reconstructed_proof = ""
    for s in steps:
        reconstructed_proof += s["tactic"]
    if proof.strip() != reconstructed_proof.strip():
        print("Mismatch between proof and reconstructed proof.")
        print(proof)
        print("----------------")
        print(reconstructed_proof)
        print("\n\n\n")
        return False
    else:
        return True


def remove_header_and_tail(full_text):
    full_text = full_text[full_text.index("```lean4") + len("```lean4") :]
    full_text = full_text[: full_text.rindex("```")]
    return full_text


if __name__ == "__main__":
    # 1. Define the Lean4 client
    lean4_client = Lean4Client(
        url="http://lean4-evaluator-internal.app.msh.team/",
    )

    # 2. Load the dataset
    dataset = datasets.load_dataset(
        "AI-MO/wholeproof-pt-250119-v3_minhash-n3",
        split="train",
        cache_dir="/mnt/moonfs/wanghaiming-m2/.cache/ttmmpp",
    )

    # 3. Get the Lean files, add the skip and get the infotrees
    codes = []

    for idx, sample in enumerate(dataset):
        full_code = sample["text"]  # contains imports, statement, proof
        full_code = remove_header_and_tail(full_code)
        # indentation = extract_indentation(full_code)
        full_code += "\n  skip"
        header, theorem = split_proof_header(
            full_code
        )  # header: imports, theorem: statement+proof
        sample["proof"] = full_code
        sample["proof_id"] = str(uuid.uuid4())
        sample["theorem"] = theorem
        codes.append(sample)
        # if idx == 100:
        #     break

    results = batch_verify_proof(
        client=lean4_client,
        samples=codes,
        timeout=60,
        num_threads=220,
        batch_size=1,
        infotree_type="original",
    )

    # 4. Get the data from the infotrees, remove the skip step, check for alignment
    data = []
    for result in tqdm(results):
        try:
            infotree = (
                json.loads(result["lean_feedback"])
                .get("response", {})
                .get("infotree", [])
            )
        except Exception as e:
            print("Error while processing infotree:", e)
            continue
        tactics = extract_data(infotree, result["theorem"])
        if len(tactics) == 0:
            continue
        if "skip" in tactics[-1]["tactic"]:
            tactics = tactics[:-1]  # remove the skip
        if extraction_is_correct(tactics, result["theorem"]):
            result["tactics"] = tactics
            data.append(result)

    print("Number of samples:", len(results))
    print("Number of samples with correct extraction:", len(data))

    # 5. Save the data
    dataset = datasets.Dataset.from_list(data)

    # keep only the relevant columns
    dataset = dataset.remove_columns(
        [
            "proof",
            "theorem",
            "proof_id",
            "lean_feedback",
            "has_error",
            "is_valid_no_sorry",
            "is_valid_with_sorry",
        ]
    )

    dataset.push_to_hub("AI-MO/step-proof-stage1-pt-250119-v3_minhash-n3", private=True)
