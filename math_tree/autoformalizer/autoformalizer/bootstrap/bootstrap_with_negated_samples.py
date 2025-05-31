import fire
from datasets import load_dataset, concatenate_datasets
import random
from autoformalizer.model_utils.infer_hf_dataset import (
    infer_hf_dataset,
    infer_hf_dataset_with_negated_sample,
)
from autoformalizer.eval_utils.all_feedback import all_feedback
from accelerate import Accelerator
from accelerate.utils import gather_object
import torch.distributed as dist


def bootstrap(
    model_path: str,
    ref_ds: str,
    bs_ds: str,
    bs_ds_branch: str,
    save_ds: str,
    save_ds_branch: str,
    n_samples: int,
    bootstrap_id: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    seed: int = 40,
    model_name: str = "gpt-4o",
):
    """
    Bootstraps a dataset for a given model.

    TODO: NEED TO EDIT

    Parameters:
    model_path (str): The path to the model.
    ref_ds (str): The reference dataset ID.
    bs_ds (str): The bootstrap dataset ID.
    bs_ds_branch (str): The bootstrap dataset branch.
    save_ds (str): The save dataset ID.
    save_ds_branch (str): The save dataset branch.
    n_samples (int): The number of samples to bootstrap from
    bootstrap_id (str): The bootstrap ID.
    """

    ref_ds_name = ref_ds

    bs_ds = load_dataset(bs_ds, split="train", revision=bs_ds_branch)
    ref_ds = load_dataset(ref_ds, split="train")

    bs_ds_ids = bs_ds["id"]
    # Filter the reference dataset for ids that don't appear in the bootstrap dataset
    print(f"Bootstrap dataset size: {len(bs_ds)}")
    print(f"Reference dataset size before filtering: {len(ref_ds)}")
    #  convert to int for align the type
    ref_ds = ref_ds.add_column("id", [int(id_) for id_ in ref_ds["aops_id"]])
    ref_ds = ref_ds.filter(lambda x: x["id"] not in bs_ds_ids)

    print(f"Reference dataset size after filtering: {len(ref_ds)}")

    # Sample n_samples from the reference dataset and push to hub
    n_samples = min(n_samples, len(ref_ds))
    random_rows = [random.randint(0, len(ref_ds) - 1) for _ in range(n_samples)]
    ref_ds = ref_ds.select(random_rows)
    print(f"Sampled dataset size: {len(ref_ds)}")

    ref_ds.push_to_hub(
        save_ds,
        revision=f"bootstrap_{bootstrap_id}_with_negated_sample",
        commit_message=f"Bootstrap {bootstrap_id} from {ref_ds_name} with_negated_sample",
    )

    # # Autoformalize the sampled dataset
    infer_hf_dataset_with_negated_sample(
        model_path=model_path,
        dataset_id=save_ds,
        output_dataset_id=save_ds,
        dataset_branch=f"bootstrap_{bootstrap_id}_with_negated_sample",
        output_dataset_branch=f"bootstrap_{bootstrap_id}_with_negated_sample",
        n_samples=1,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    ### jq, todo:
    # Request the proof model to handle the negated statements and perform proof checks for filtering.
    # Next steps:
    # I will filter the statements where the negated versions can be proven. The remaining statements will then be processed to receive feedback from Lean and GPT for further refinement."

    # Get Lean and GPT feedback

    # all_feedback(input_dataset_id=save_ds,input_dataset_branch=f'bootstrap_{bootstrap_id}', output_dataset_id=save_ds,output_dataset_branch=f'bootstrap_{bootstrap_id}', filter_compiled=True, model_name=model_name)

    # # Load the dataset and filter for positive Lean and GPT feedback
    # addition_ds = load_dataset(save_ds, split='train', revision=f'bootstrap_{bootstrap_id}')
    # addition_ds = addition_ds.filter(lambda x: x['compiler_feedback_1_bool'] and x[f'{model_name}_feedback_1_bool'])

    # # Filter for columns in bs_ds
    # addition_ds = addition_ds.rename_column('autoformalization_1', 'lean_code')
    # addition_ds = addition_ds.add_column('source', [f"bootstrap_{bootstrap_id}"] * len(addition_ds))
    # addition_ds = addition_ds.add_column('has_header', [True] * len(addition_ds))
    # addition_ds = addition_ds.select_columns(bs_ds.column_names)
    # addition_ds = addition_ds.map(lambda x: {'id': int(x['id']) if isinstance(x['id'], str) else x['id']})

    # # add the addition to bootstrap dataset
    # bs_ds = concatenate_datasets([bs_ds, addition_ds])
    # print(f"Total items after concatenation: {len(bs_ds)}")

    # # Push the updated bootstrap dataset to hub
    # bs_ds.push_to_hub(save_ds, revision=save_ds_branch, commit_message=f"Addition from bootstrap {bootstrap_id}")


if __name__ == "__main__":
    """
    Example usage:
    python -m autoformalizer.bootstrap.bootstrap_with_negatie_samples \
    --model_path="$MODEL_DIR" \
    --ref_ds="$REF_DS" \
    --bs_ds="$BS_DS" \
    --bs_ds_branch="$BS_DS_BRANCH" \
    --save_ds="$SAVE_DS" \
    --save_ds_branch="$SAVE_DS_BRANCH" \
    --n_samples="$N_SAMPLES" \
    --bootstrap_id="$BOOTSTRAP_ID" \
    """

    fire.Fire(bootstrap)
