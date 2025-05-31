import numpy as np
from datasets import Dataset
from loguru import logger

from autoformalizer.data_utils import process_proof, process_statement


def consolidate_dataset(
    ds: Dataset,
    num_proc: int = None,
    informal_prefix: bool = False,
    proof_col: str = "formal_proof",
) -> Dataset:
    """Split the dataset into input and output for the proof, and filter out samples with unsplitable proofs.

    Args:
        ds (Dataset): The dataset to be processed.
            ds should have the following columns:
            - uuid
            - is_negation, needed if informal_prefix
            - natural_language, needed if informal_prefix
            - formal_proof
        num_proc (int, optional): Number of processes to use. Defaults to None.
        informal_prefix (bool, optional):
            Whether to insert informal problem as comment before the formal statement. Defaults to False.
        proof_col (str, optional): The column name of the proof. Defaults to "formal_proof".
    Output columns:
        - uuid
        - statement_id
        - formal_statement
        - proof_input
        - proof_output
        - formal_proof
    """
    ds_len = len(ds)
    ds = ds.filter(
        lambda x: len(process_proof.get_statement_split_indexes(x[proof_col])) > 0,
        num_proc=num_proc,
    )
    logger.info(f"Filtered {ds_len - len(ds)} samples with unsplitable proofs.")

    # insert informal problem as comment before the formal statement
    if informal_prefix:

        def _insert_informal(sample):
            natural_language = sample["natural_language"]
            # remove final answer, discussable
            informal = natural_language.split("The final answer is")[0].strip()
            if not sample["is_negation"]:
                shortest_proof = process_statement.insert_informal(
                    sample[proof_col], informal
                )
            else:
                shortest_proof = sample[proof_col]
            return {
                proof_col: shortest_proof,
            }

        ds = ds.map(
            _insert_informal,
            num_proc=num_proc,
        )

    def _split_proof(sample):
        proof_input, proof_out = process_proof.split_proof(sample[proof_col])
        return {
            "proof_input": proof_input,
            "proof_output": proof_out,
            "formal_proof": sample[proof_col],
        }

    ds = ds.map(_split_proof, num_proc=num_proc)
    if "is_negation" in ds.column_names:
        negations = ds["is_negation"]
        # log negation rate
        logger.info(f"Negation rate: {np.mean(negations)}")

    ds = ds.select_columns(
        [
            "uuid",
            # need to include statement ?
            # "statement_id",
            # "formal_statement",
            "proof_input",
            "proof_output",
            "formal_proof",
        ]
    )
    return ds
