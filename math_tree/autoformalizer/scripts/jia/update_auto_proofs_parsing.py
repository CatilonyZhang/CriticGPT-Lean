import json

from datasets import Dataset, load_dataset
from loguru import logger

from autoformalizer.data_utils import metrics
from autoformalizer.eval_utils import lean_feedback


class IndexDataset:
    """
    A class to efficiently index and retrieve rows from a dataset based on a primary
    key or secondary indices.

    This class allows fast lookups of rows by a unique primary key or
    non-unique secondary indices.

    Attributes:
        dataset (Dataset): The dataset to be indexed. Must support item access
            (e.g., dataset[key] or dataset[index]).
        primary_key (str): The column name or key representing the primary key
            of the dataset. Must be unique if defined.
        indices (list[str]): A list of column names or keys to build secondary
            indices. These indices may have duplicate values.

    Methods:
        get_row_by_key(key):
            Retrieves a single row from the dataset using the primary key.
        get_rows_by_index(index, idx):
            Retrieves all rows matching the specified index and value.

    Raises:
        TypeError: If the dataset is not an instance of the `Dataset` class.
        ValueError: If the primary key is not unique or if an invalid key/index is accessed.
    """

    def __init__(self, dataset, primary_key=None, indices=None):
        """
        Initializes the IndexDataset class.

        Args:
            dataset (Dataset): The dataset to be indexed.
            primary_key (str, optional): The primary key column. Must be unique if provided. Defaults to None.
            indices (list[str], optional): A list of secondary index columns. Defaults to None.

        Raises:
            TypeError: If the dataset is not an instance of `Dataset`.
            ValueError: If the primary key is not unique.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be an instance of Dataset")

        self.dataset = dataset
        self.primary_key = primary_key
        self.indices = indices
        self._key_to_row = {}
        self._index_to_row = {}

        if self.primary_key:
            self._build_primary_key_index()
        if self.indices:
            self._build_secondary_indices()

    def _build_primary_key_index(self):
        ds_key = self.dataset[self.primary_key]
        if len(ds_key) != len(set(ds_key)):
            raise ValueError("Primary key values must be unique")
        self._key_to_row = {key: i for i, key in enumerate(ds_key)}
        logger.info(f"Primary key index built for {self.primary_key}")

    def _build_secondary_indices(self):
        for index in self.indices:
            self._index_to_row[index] = {}
            ds_index = self.dataset[index]
            for i, idx in enumerate(ds_index):
                self._index_to_row[index].setdefault(idx, []).append(i)
        logger.info(f"Secondary indices built for {self.indices}")

    def get_row_by_key(self, key):
        if not self.primary_key:
            raise ValueError("Primary key is not defined")
        return self.dataset[self._key_to_row.get(key, -1)]

    def get_rows_by_index(self, index, idx):
        if index not in self.indices:
            raise ValueError(f"Index {index} is not defined")
        return [self.dataset[i] for i in self._index_to_row.get(index, {}).get(idx, [])]


def run_parsing(dataset):
    # dataset = dataset.select(range(0, 1000))
    def _parse(sample):
        server_response = json.loads(sample["lean_server_feedback"])
        is_valid_no_sorry = lean_feedback.parse_client_response(server_response)[
            "is_valid_no_sorry"
        ]

        return {
            "is_valid_no_sorry": is_valid_no_sorry,
        }

    dataset = dataset.map(_parse, num_proc=15)

    df_metrics = metrics.compute_auto_proofs_metrics(dataset)
    return dataset, df_metrics


def add_statement_id():
    for ds_name in ["AI-MO/auto-statements-v1", "AI-MO/auto-statements-v3"]:
        dataset = load_dataset(ds_name, split="train")
        if "statement_id" in dataset.column_names:
            logger.info(f"statement_id already exists in {ds_name}")
            continue
        dataset = dataset.map(
            lambda x, index: {"statement_id": x["uuid"] + "_" + str(index)},
            with_indices=True,
        )
        dataset.push_to_hub(ds_name, private=True)


def find_statement_id_for_auto_proofs(
    proof_ds_name="AI-MO/auto-proofs-v1", statement_ds_name="AI-MO/auto-statements-v1"
):
    auto_proofs = load_dataset(proof_ds_name, split="train")
    auto_statements = load_dataset(statement_ds_name, split="train")
    if "statement_id" in auto_proofs.column_names:
        raise ValueError(f"statement_id already exists in {proof_ds_name}")
    auto_index_statements = IndexDataset(
        auto_statements, primary_key="statement_id", indices=["uuid"]
    )

    def _find_statement_id(sample, index):
        uuid = sample["uuid"]
        rows = auto_index_statements.get_rows_by_index("uuid", uuid)
        statement_id = None
        proof_id = None
        for row in rows:
            formal_statement = row["formal_statement"]
            no_sorry = formal_statement.split("sorry")[0]
            # no_sorry = formal_statement.replace(" by sorry\n", "").replace(" by sorry", "")
            # no_sorry = no_sorry.replace("\nsorry", "").replace(" sorry", "")
            if len(rows) == 1 or no_sorry in sample["proof"]:
                statement_id = row["statement_id"]
                proof_id = f"{row['statement_id']}_{index}"
            # else:

        if statement_id is None:
            logger.warning(f"miss match for {uuid}")
            logger.info(formal_statement)
            logger.info(repr(no_sorry))
            logger.info(sample["proof"])
            logger.error(f"row length: {len(rows)}")
            raise ValueError(f"Could not find statement_id for {uuid}")
        return {"statement_id": statement_id, "proof_id": proof_id}

    auto_proofs = auto_proofs.map(_find_statement_id, with_indices=True, num_proc=1)
    auto_proofs.push_to_hub(proof_ds_name, private=True)


def update_auto_proofs_parsing():
    v1 = load_dataset("AI-MO/auto-proofs-v1", split="train")
    v1 = v1.rename_column("server_feedback", "lean_server_feedback")
    v1, df_metrics = run_parsing(v1)
    # find the difference between is_valid_server and is_valid_no_sorry
    diff = v1.filter(lambda x: x["is_valid_server"] != x["is_valid_no_sorry"])
    diff.push_to_hub("AI-MO/auto-proofs-v1-parser-diff", private=True)
    v1.push_to_hub("AI-MO/auto-proofs-v1", private=True)
    logger.info(df_metrics)

    v3 = load_dataset("AI-MO/auto-proofs-v3", split="train")
    v3 = v3.rename_column("server_feedback", "lean_server_feedback")
    v3, df_metrics = run_parsing(v3)
    # find the difference between is_valid_server and is_valid_no_sorry
    diff = v3.filter(lambda x: x["is_valid_server"] != x["is_valid_no_sorry"])
    diff.push_to_hub("AI-MO/auto-proofs-v3-parser-diff", private=True)
    v3.push_to_hub("AI-MO/auto-proofs-v3", private=True)
    logger.info(df_metrics)


if __name__ == "__main__":
    add_statement_id()
    find_statement_id_for_auto_proofs(
        "AI-MO/auto-proofs-v1", "AI-MO/auto-statements-v1"
    )
    find_statement_id_for_auto_proofs(
        "AI-MO/auto-proofs-v3", "AI-MO/auto-statements-v3"
    )
    update_auto_proofs_parsing()
