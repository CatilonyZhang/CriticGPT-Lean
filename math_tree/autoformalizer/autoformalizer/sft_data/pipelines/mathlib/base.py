from datasets import load_dataset


class MathlibPipeline:

    NAME = "AI-MO/human-statements-dataset-v1-mathlib-20250106"

    def __init__(self):
        self.dataset = load_dataset(self.NAME, split="train")

    def process(self):

        self.dataset = self.dataset.rename_column("formal_statement", "proof_input")
        self.dataset = self.dataset.rename_column("ground_truth", "proof_output")
        self.dataset = self.dataset.map(
            lambda x: {"formal_proof": x["proof_input"] + x["proof_output"]}
        )
        self.dataset.push_to_hub("AI-MO/mathlib-sft", private=True)


if __name__ == "__main__":
    """Usage:
    python -m autoformalizer.sft_data.pipelines.mathlib.base
    """
    pipeline = MathlibPipeline()
    pipeline.process()
