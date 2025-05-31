from datasets import concatenate_datasets, load_dataset
from loguru import logger

WHOLE_PROOF_PT_DATASETS = {
    "AI-MO/wholeproof-pt-250115-v3": [
        "AI-MO/auto-sft-moon-santa-v3.1beta1",
        "AI-MO/human-statements-sft-v1pick10",
        "AI-MO/human-proofs-sft",
    ],
    "AI-MO/wholeproof-pt-250115-v2": [
        "AI-MO/auto-sft-moon-santa-v3beta1",
        "AI-MO/human-statements-sft-v1pick10",
        "AI-MO/human-proofs-sft",
    ],
    "AI-MO/wholeproof-pt-250115-v1": [
        "AI-MO/auto-sft-moon-santa-v2beta1",
        "AI-MO/human-statements-sft-v1pick10",
        "AI-MO/human-proofs-sft",
    ],
    "AI-MO/wholeproof-pt-250119-v1": [
        "AI-MO/auto-sft-moon-santa-v3beta1",
        "AI-MO/auto-sft-moon-flying-ant-prover-v1-base",
        "AI-MO/human-statements-sft-v1pick10",
        "AI-MO/human-proofs-sft",
    ],
    "AI-MO/wholeproof-pt-250119-v2": [
        "AI-MO/auto-sft-moon-santa-v3beta1",
        "AI-MO/auto-sft-moon-flying-ant-prover-v1-base",
        # add it twice to increase the probability of being selected
        "AI-MO/auto-sft-moon-flying-ant-prover-v1-base",
        "AI-MO/human-statements-sft-v1pick10",
        "AI-MO/human-proofs-sft",
    ],
    "AI-MO/wholeproof-pt-250119-v3": [
        "AI-MO/auto-sft-moon-santa-v3beta1",
        "AI-MO/auto-sft-moon-flying-ant-prover-v1-base",
        ("AI-MO/human-statements-sft-v1pick10", 5),
        ("AI-MO/human-proofs-sft", 10),
    ],
    "AI-MO/wholeproof-pt2-250119-v4": [
        ("AI-MO/auto-sft-moon-santa-v3beta1", 0.05),
        ("AI-MO/auto-sft-moon-flying-ant-prover-v1-base", 0.1),
        ("AI-MO/human-statements-sft-v1pick10", 5),
        ("AI-MO/human-proofs-sft", 20),
    ],
}


def generating_text(sample):
    proof = sample["formal_proof"]
    return {
        "text": f"Complete the following Lean 4 code:\n\n```lean4\n{proof}\n```",
    }


def make_whole_proof_pt_dataset(sft_list):
    # load all datasets

    ds_list = []
    for sft in sft_list:
        if isinstance(sft, tuple):
            sft_name, coef = sft
        elif isinstance(sft, str):
            sft_name = sft
            coef = 1
        else:
            raise ValueError(f"Invalid sft: {sft}")
        ds = load_dataset(sft_name, split="train", num_proc=8)
        if isinstance(coef, int):
            ds = concatenate_datasets([ds] * coef)
        elif isinstance(coef, float):
            ds = ds.shuffle(seed=142)
            ds = ds.select(range(int(len(ds) * coef)))
        else:
            raise ValueError(f"Invalid coef: {coef}")
        logger.info(f"Loaded {sft_name} using coef {coef}, total samples: {len(ds)}")

        ds = ds.select_columns(
            [
                "uuid",
                "proof_input",
                "proof_output",
                "formal_proof",
            ]
        )
        ds_list.append(ds)

    ds = concatenate_datasets(ds_list)
    ds = ds.map(generating_text, num_proc=8)
    ds = ds.select_columns(
        [
            "uuid",
            "proof_input",
            "proof_output",
            "text",
        ]
    )
    return ds


def check_dataset_exists(dataset_name: str) -> bool:
    """
    # Example usage
    dataset_name = "AI-MO/wholeproof-pt-250119"
    check_dataset_exists(dataset_name)
    """
    try:
        # Attempt to load the dataset
        load_dataset(dataset_name)
        return True
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    for dataset_name, sft_list in WHOLE_PROOF_PT_DATASETS.items():
        # if check_dataset_exists(dataset_name):
        #     logger.info(f"Dataset {dataset_name} already exists")
        #     continue
        logger.info(f"Generating dataset {dataset_name}")
        ds = make_whole_proof_pt_dataset(sft_list)
        # shuffle the dataset
        ds = ds.shuffle(seed=142)
        print(ds)
        ds.push_to_hub(dataset_name, private=True)
