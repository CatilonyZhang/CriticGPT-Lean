# -----------------------------------------------------------------------------
# Author: Jia
# Date: 2024-12-11
# -----------------------------------------------------------------------------
import fire
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from numinamath.dataset_utils import make_natural_language_prompt
from tqdm import tqdm

from autoformalizer.eval_utils import gpt_feedback
from autoformalizer.model_utils import autoformalize_hf_dataset as auto_hf


def generate_aops_wiki_candidates():
    """
    Take the aops base dataset, generate natural language prompts for each problem
    and sample autoformalizations from the LLM model.
    """

    aops_wiki_base = load_dataset("AI-MO/aops-wiki-base", split="train")
    aops_wiki_base = aops_wiki_base.map(make_natural_language_prompt)

    # filter the problems
    def _filter_problems(sample):
        if sample["question_type"] == "proof":
            return True
        else:
            if sample["answer"] == "notfound":
                return False
            if sample["problem_is_valid"] == "Incomplete":
                return False
            if sample["solution_is_valid"] == "Invalid":
                return False

        return True

    aops_wiki_base = aops_wiki_base.filter(_filter_problems)
    aops_wiki_base = auto_hf.sample_autoformalisation(
        model_path="AI-MO/Qwen7BCoder_AutoformalizerV1B7",
        dataset_id=aops_wiki_base,
        batch_size=1024,
        push_to_hub=False,
        n_samples=8,
    )
    print(aops_wiki_base)
    auto_statements_candidates = []
    for sample in aops_wiki_base:
        for i, candidate in enumerate(sample["auto_statements_candidates"]):
            auto_statements_candidates.append(
                {
                    "uuid": sample["uuid"],
                    "natural_language_statement": sample["natural_language"],
                    "formal_statement": candidate["code"],
                    "statement_id": f"{sample['uuid']}_{i}",
                    "source": "aops_wiki",
                }
            )
    # push to hub
    auto_statements_candidates = Dataset.from_list(auto_statements_candidates)

    # push to V2 and main
    auto_statements_candidates.push_to_hub(
        "AI-MO/auto-statements-candidates",
        private=True,
        revision="V2",
    )
    auto_statements_candidates.push_to_hub(
        "AI-MO/auto-statements-candidates",
        private=True,
    )


def compute_feedback_aops_wiki():
    ds = auto_hf.mini_batch_statement_feedback(
        "AI-MO/auto-statements-candidates",
        working_dir="./autoformalizer_cache/auto_statements_candidates_aops_wiki",
        num_proc=100,
    )
    ds.push_to_hub(
        "AI-MO/auto-statements-candidates",
        private=True,
        revision="V2",
    )


def collect_auto_statements_from_other_datasets():
    auto_statements = load_dataset("AI-MO/auto-statements-v1", split="train")
    print(auto_statements)
    # remove problem with high proof rate
    problem_to_remove = set(
        auto_statements.filter(lambda x: x["n_correct_proofs"] > 10)["uuid"]
    )
    print(f"Remove {len(problem_to_remove)} problems")

    auto_datasets = DatasetDict(
        {
            "math-train": load_dataset(
                "AI-MO/math-autoformalization-v0.1", split="train"
            ),
            "math-test": load_dataset(
                "AI-MO/math-autoformalization-v0.1", split="test"
            ),
            "aops": load_dataset("AI-MO/aops-autoformalization-v0.1", split="train"),
        }
    )

    auto_datasets = auto_datasets.filter(lambda x: x["uuid"] not in problem_to_remove)

    statement_candidates = []
    for key, ds in auto_datasets.items():
        for sample in tqdm(ds):
            for i, candidate in enumerate(sample["autoformalization_samples"]):
                statement_candidates.append(
                    {
                        "uuid": sample["uuid"],
                        "natural_language_statement": sample["natural_language"],
                        "formal_statement": candidate["code"],
                        "statement_id": f"{sample['uuid']}_{i}",
                        "source": key,
                        "lean_feedback": "",
                        "has_error": not candidate["lean4_feedback_status"],
                    }
                )
    print(f"Number of statement candidates: {len(statement_candidates)}")

    statement_candidates = Dataset.from_list(statement_candidates)

    # print no error candidates number
    no_error_candidates = statement_candidates.filter(lambda x: not x["has_error"])
    print(f"No error candidates: {len(no_error_candidates)}")

    # concatenate with v2
    auto_statements_v2 = load_dataset(
        "AI-MO/auto-statements-candidates", split="train", revision="V2"
    )
    print(auto_statements_v2)

    statement_candidates = concatenate_datasets(
        [auto_statements_v2, statement_candidates]
    )
    print(statement_candidates)

    # push to hub
    statement_candidates.push_to_hub(
        "AI-MO/auto-statements-candidates",
        private=True,
        revision="V2",
    )

    # push also to main to visualize
    statement_candidates.push_to_hub(
        "AI-MO/auto-statements-candidates",
        private=True,
    )


def remove_exact_duplicates():
    ds = load_dataset("AI-MO/auto-statements-candidates", split="train", revision="V2")
    total_samples = len(ds)

    # add index
    ds = ds.map(lambda x, i: {**x, "index": i}, with_indices=True)

    # remove exact duplicates of columns formal_statement
    unique_statement = set()
    unique_index = set()
    for sample in ds:
        if sample["formal_statement"] not in unique_statement:
            unique_statement.add(sample["formal_statement"])
            unique_index.add(sample["index"])

    ds = ds.filter(lambda x: x["index"] in unique_index)

    # print the number of unique statements and the number of duplicates
    print(f"Number of unique statements: {len(unique_statement)}")
    print(f"Number of duplicates: {total_samples - len(unique_statement)}")
    print(
        f"Total number of unique uuids after removing duplicates: {len(set(ds['uuid']))}"
    )

    ds = ds.remove_columns("index")
    print(ds)

    ds.push_to_hub(
        "AI-MO/auto-statements-candidates",
        private=True,
        revision="V2",
    )

    ds.push_to_hub(
        "AI-MO/auto-statements-candidates",
        private=True,
        revision="main",
    )


def compute_gpt_feedback(dry_run=True):
    candidates = load_dataset(
        "AI-MO/auto-statements-candidates", split="train", revision="V2"
    )
    print(candidates)

    if dry_run:
        # pick the first 1600 samples for dry run, around 200 uuids so to compute pass@8
        candidates = candidates.select(range(1600))

    lean4_codes = [sample["formal_statement"] for sample in candidates]
    natural_language_ls = [
        sample["natural_language_statement"] for sample in candidates
    ]

    # check the number of unique lean4 codes
    print(f"Number of unique lean4 codes: {len(set(lean4_codes))}")
    print(f"Total number of lean4 codes: {len(lean4_codes)}")

    print("Computing GPT feedback")
    print(f"Number of candidates: {len(lean4_codes)}")
    cache_dir = "./autoformalizer_cache/gpt_feedback"

    generations = gpt_feedback.batch_gpt_feedback(
        lean4_codes,
        natural_language_ls,
        cache_dir=cache_dir,
    )

    assert len(generations) == len(lean4_codes)

    def _update_dataset(sample, i):
        is_correct_by_gpt = gpt_feedback.parse_formalization_status(generations[i])
        return {
            "gpt_feedback": generations[i],
            "is_correct_by_gpt": is_correct_by_gpt,
        }

    candidates = candidates.map(_update_dataset, with_indices=True)

    if not dry_run:
        candidates.push_to_hub(
            "AI-MO/auto-statements-candidates",
            private=True,
            revision="V2",
        )
    else:
        candidates.push_to_hub(
            "AI-MO/auto-statements-candidates-dry-run",
            private=True,
        )


def calculate_statistics(dry_run=True):
    """compute some statistics on the error rate and gpt feedback correct rate"""
    if dry_run:
        ds_dry_run = load_dataset(
            "AI-MO/auto-statements-candidates-dry-run", split="train"
        )
        df = ds_dry_run.to_pandas()
    else:
        ds = load_dataset(
            "AI-MO/auto-statements-candidates", split="train", revision="V2"
        )
        df = ds.to_pandas()

    # print the total number and number of unique uuids
    print(f"Total number of statements: {len(df)}")
    print(f"Total number of unique uuids: {len(df['uuid'].unique())}")

    # calculate the no error pass rate for all statements
    df["is_correct_by_compiler"] = ~df["has_error"]
    compiler_pass_rate = df["is_correct_by_compiler"].mean()
    print(f"Compiler pass rate: {compiler_pass_rate}")

    # calculate compiler pass@8 rate for uuid
    # this means that if there is at least one correct statement, then the compiler pass for the uuid
    df["compiler_pass@8"] = df.groupby("uuid")["is_correct_by_compiler"].transform(
        "max"
    )
    compiler_pass_8_rate = df["compiler_pass@8"].mean()
    print(f"Compiler pass@8 rate: {compiler_pass_8_rate}")

    # calculate the compiler pass rate and gpt pass rate
    df["is_correct_by_compiler_gpt"] = df["is_correct_by_compiler"] & (
        df["is_correct_by_gpt"] == "Correct"
    )
    compiler_gpt_pass_rate = df["is_correct_by_compiler_gpt"].mean()
    print(f"Compiler GPT pass rate: {compiler_gpt_pass_rate}")

    # calculate compiler gpt pass@8 rate for uuid
    df["compiler_gpt_pass@8"] = df.groupby("uuid")[
        "is_correct_by_compiler_gpt"
    ].transform("max")
    compiler_gpt_pass_8_rate = df["compiler_gpt_pass@8"].mean()
    print(f"Compiler GPT pass@8 rate: {compiler_gpt_pass_8_rate}")


def create_auto_statements_v2():
    ds = load_dataset("AI-MO/auto-statements-candidates", split="train", revision="V2")
    # keep only compiler and gpt correct statements
    ds = ds.filter(
        lambda x: (x["is_correct_by_gpt"] == "Correct") and (not x["has_error"])
    )

    print(f"Number of correct statements: {len(ds)}")
    print(f"Number of unique uuids: {len(set(ds['uuid']))}")

    # show the break down of unique uuid by source
    df = ds.to_pandas()
    print(df.groupby("source")["uuid"].nunique())

    ds.push_to_hub(
        "AI-MO/auto-statements-v2",
        private=True,
    )


if __name__ == "__main__":
    """Process steps:

    1. Generate natural language prompts for each problem in the aops-wiki-base dataset.

    >>> python -m numinamath.dataset_manager.build_auto_v2 generate_aops_wiki_candidates

    2. Compute feedback using lean4 client
    >>> python -m numinamath.dataset_manager.build_auto_v2 compute_feedback_aops_wiki

    3. Collect auto statements from other datasets
    >>> python -m numinamath.dataset_manager.build_auto_v2 collect_auto_statements_from_other_datasets

    4. Remove exact duplicates
    We realize there are exact duplicates in the sampled auto statements. We remove them.
    >>> python -m numinamath.dataset_manager.build_auto_v2 remove_exact_duplicates

    5. Compute feedback using GPT
    >>> python -m numinamath.dataset_manager.build_auto_v2 compute_gpt_feedback \
        --dry_run=False

    6. Calculate statistics
    >>> python -m numinamath.dataset_manager.build_auto_v2 calculate_statistics \
        --dry_run=False

    7. Create auto statements v2, by keeping only compiler and gpt correct statements
    >>> python -m numinamath.dataset_manager.build_auto_v2 create_auto_statements_v2
    """
    fire.Fire(
        {
            "generate_aops_wiki_candidates": generate_aops_wiki_candidates,
            "compute_feedback_aops_wiki": compute_feedback_aops_wiki,
            "collect_auto_statements_from_other_datasets": collect_auto_statements_from_other_datasets,
            "remove_exact_duplicates": remove_exact_duplicates,
            "compute_gpt_feedback": compute_gpt_feedback,
            "calculate_statistics": calculate_statistics,
            "create_auto_statements_v2": create_auto_statements_v2,
        }
    )
