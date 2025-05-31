from datasets import load_dataset
from loguru import logger

from autoformalizer.sft_data.consolidate import consolidate_dataset


class HumanProofsPipeline:

    NAME = "AI-MO/numina-math-lean4"
    EVAL_NAME = "AI-MO/inhouse-evaluation-v1-20250102"

    def __init__(self):
        self.dataset = load_dataset(self.NAME, split="train")

    def process(self):

        self.dataset = self.dataset.filter(lambda x: x["is_valid_no_sorry"])
        # remove MATH test
        self.dataset = self.dataset.filter(
            lambda x: "source:MATH-test" not in x["tags"]
        )
        logger.info(f"Total valid proofs: {len(self.dataset)}")
        final_ds = consolidate_dataset(self.dataset, proof_col="lean4_solution")

        # load the eval dataset
        eval_ds = load_dataset(self.EVAL_NAME)
        splits = [
            "inhouse_evaluation_math_test",
            "inhouse_evaluation_others_10_percent",
        ]
        uuids = set()
        for split in splits:
            subset = eval_ds[split]
            uuids.update(subset["uuid"])
        logger.info(f"Total eval proofs: {len(uuids)}")
        final_ds = final_ds.filter(lambda x: x["uuid"] not in uuids)
        print(final_ds)
        final_ds.push_to_hub("AI-MO/human-proofs-sft", private=True)


if __name__ == "__main__":
    """Usage:
    python -m autoformalizer.sft_data.pipelines.human_proofs.base
    """
    pipeline = HumanProofsPipeline()
    pipeline.process()
