import os

from autoformalizer.clients.infotree.process_infotree import extract_data
from autoformalizer.clients.lean4_client import Lean4Client
from autoformalizer.repl_lean_feedback.leanrepl import LeanREPL
from prover.bfs_prover import BestFirstSearchProver
from prover.tactic_generator import APITacticGenerator

platform = "online"


def append_skip(proof_attempt: str, indent: int):
    return proof_attempt + "\n" + " " * indent + "skip"


if __name__ == "__main__":

    if platform == "offline":
        lean_folder = "/home/mantas/autoformalizer/scripts/mantas/edge_cases"
        for filename in os.listdir(lean_folder):
            file_path = os.path.join(lean_folder, filename)
            if os.path.isfile(file_path):
                with open(file_path, "r") as file:
                    content = file.read()
                    # Append the last line of the proof with the skip command after the default indentation
                    # TODO: implement an algorithm to determine the indentation level
                    skip_content = append_skip(content, 2)
                    # store the content in a new file with a name that indicates the skip command was appended
                    new_filename = filename.replace(".lean", "_skip.lean")
                    new_file_path = os.path.join(lean_folder, new_filename)
                    with open(new_file_path, "w") as file:
                        file.write(skip_content)
                    repl = LeanREPL()
                    output_old = repl.extract_infotree_from_file(file_path)
                    output_new = repl.extract_infotree_from_file(new_file_path)
                    infotree_old = output_old["infotree"]
                    data_old = extract_data(infotree_old, content)
                    infotree_new = output_new["infotree"]
                    data_new = extract_data(infotree_new, skip_content)
                    repl.close()
                    # Process the content as needed
                    print(f"Processing file: {filename}")
                    print("Old step data:")
                    for d in data_old:
                        print(d)
                        print("-" * 10)
                    print("New step data:")
                    for d in data_new:
                        print(d)
                        print("-" * 10)
    if platform == "online":
        tactic_generator = APITacticGenerator()

        lean4_client = Lean4Client(
            url="https://kimina.saas.moonshot.cn/lean4-evaluator",
        )

        search = BestFirstSearchProver(
            tactic_generator,
            lean4_client,
            timeout=300,
            num_sampled_tactics=5,
            max_expansions=20,
        )

        code_prefix = """import Mathlib
import Aesop


set_option maxHeartbeats 0


open BigOperators Real Nat Topology Rat


theorem mathd_algebra_208 : Real.sqrt 1000000 - 1000000 ^ ((1 : ℝ) / 3) = 900 := by
  rw [sub_eq_add_neg]"""
        suggestions = [
            (
                "\n  norm_num [Real.sqrt_eq_rpow, Real.rpow_def_of_pos, show (0 : ℝ) < 3 by norm_num]",
                1.0,
            )
        ]
        indent = 3
        output = search._run_seq_tactic(None, code_prefix, suggestions, indent)
        print(output)
