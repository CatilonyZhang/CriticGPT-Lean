import uuid
from time import time

from autoformalizer.clients.infotree.process_infotree import (
    extract_data,
    extract_nodes_and_edges,
)
from autoformalizer.clients.infotree.visualize_infotree import visualize_infotree
from autoformalizer.clients.lean4_client import Lean4Client
from autoformalizer.repl_lean_feedback.leanrepl import LeanREPL
from autoformalizer.repl_lean_feedback.repl_utils import split_proof_header

platform = "online"


def append_skip(proof_attempt: str, indent: int):
    return proof_attempt + "\n" + [" "] * indent + "skip"


if __name__ == "__main__":
    lean_file = "scripts/marco/test.lean"

    start = time()

    # Extract the infotree from the Lean file
    if platform == "offline":
        repl = LeanREPL()
        output = repl.extract_infotree_from_file(lean_file)
        print(output["messages"])
        infotree = output["infotree"]
        repl.close()
    else:
        lean4_client = Lean4Client(
            url="http://lean4-evaluator.app.msh.team/",  # switch to your url
        )
        with open(lean_file, "r") as f:
            lean_code = f.read()
        codes = [{"code": lean_code, "custom_id": str(uuid.uuid4())}]
        response = lean4_client.one_pass_verify_batch(
            codes=codes, timeout=10, infotree_type="original"
        )
        infotree = response["results"][0]["response"]["infotree"]

    file_time = time() - start
    print("Time taken to extract the infotree from the Lean file:", file_time)

    # Visualize the infotree
    nodes, edges, _ = extract_nodes_and_edges(
        infotree, include_failed_pp=False, deduplicate=True
    )
    visualize_infotree(nodes, edges, "infotree.html")

    # Extract data from the infotree
    with open(lean_file, "r") as f:
        lean_code = f.read()
    header, proof = split_proof_header(lean_code)

    if platform == "offline":
        # offline infotree takes into account header in line count
        data = extract_data(infotree, lean_code)
    else:
        data = extract_data(infotree, proof)

    data = extract_data(infotree, proof)
    infotree_time = time() - start - file_time
    print("Time taken to process the infotree:", infotree_time)

    for d in data:
        print(d)
        print("-" * 10)
