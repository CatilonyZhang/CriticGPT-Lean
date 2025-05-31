import datasets

from autoformalizer.data_utils import process_statement

data = datasets.load_from_disk(
    "/mnt/moonfs/kimina-m2/.cache/auto-statements-moon-santa-prover-v1-cleanup-hard"
)
# data = data.select(range(1000))


def _insert_informal(sample):
    natural_language = sample["natural_language"]
    # remove final answer, discussable
    informal = natural_language.split("The final answer is")[0].strip()
    sample["formal_statement"] = process_statement.insert_informal(
        sample["formal_statement"], informal
    )
    return sample


data = data.map(
    _insert_informal,
    num_proc=128,
)

data.push_to_hub("AI-MO/auto-statements-moon-flying-ant-v1-20250110", private=True)

# data.save_to_disk("/mnt/moonfs/kimina-m2/.cache/auto-statements-moon-santa-prover-v1-cleanup-hard-informal_added")
