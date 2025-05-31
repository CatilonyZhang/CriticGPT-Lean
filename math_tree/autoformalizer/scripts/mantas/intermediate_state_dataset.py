import json
import os
import subprocess

from tqdm import tqdm

from autoformalizer.eval_utils.constants import base
from autoformalizer.repl_lean_feedback.create_dataset import create_dataset_from_folder

if __name__ == "__main__":
    # Read the .jsonl file
    with open("scripts/mantas/reservoir/scraped_reservoir.jsonl", "r") as f:
        repos = [json.loads(line) for line in f]

    # Clone each repo at the specified commit hash
    with tqdm(
        total=len(repos), desc="Building and extracting state data from repositories"
    ) as pbar:
        for repo in repos:
            repo_url = repo["repository_link"]
            commit_hash = repo["commit_hash"]

            # Clone the repo
            subprocess.run(["git", "clone", f"{repo_url}.git"], cwd=base)

            # Change directory to the repo folder and checkout the commit
            repo_name = repo_url.split("/")[-1]
            path_to_repo = os.path.join(base, repo_name)
            subprocess.run(["git", "-C", repo_name, "checkout", commit_hash], cwd=base)
            # Get project cache if one is available, this might error but is not fatal
            subprocess.run("lake exe cache get", shell=True, cwd=path_to_repo)
            # Build the Lean project
            subprocess.run("lake build", shell=True, cwd=path_to_repo)
            # Match the REPL version to the project version
            subprocess.run(
                f"cp lean-toolchain {base}repl/lean-toolchain",
                shell=True,
                cwd=path_to_repo,
            )
            subprocess.run("lake build", shell=True, cwd=os.path.join(base, "repl"))
            create_dataset_from_folder(
                folder_path=".",
                path_to_repo=path_to_repo,
                output_file=f"scripts/mantas/dataset/{repo_name}.jsonl",
                recurse=True,
                num_workers=32,
                timeout_threshold=360,
            )
            try:
                subprocess.run(f"rm -rf {path_to_repo}", shell=True, cwd=base)
                print(f"Successfully removed {repo_name}")
            except subprocess.CalledProcessError as e:
                print(f"Error while removing directory: {e}")

            pbar.update(1)
