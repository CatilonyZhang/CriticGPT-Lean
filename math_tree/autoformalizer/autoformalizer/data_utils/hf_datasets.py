from datasets import Dataset
from loguru import logger


class IndexDataset:
    """
    A class to efficiently index and retrieve rows from a dataset based on a primary
    key or secondary indices.

    This class allows fast lookups of rows by a unique primary key or
    non-unique secondary indices.

    Attributes:
        dataset (Dataset): The dataset to be indexed.
        primary_key (str): The column name or key representing the primary key
            of the dataset. Must be unique if defined.
        indices (list[str]): A list of column names or keys to build secondary
            indices. These indices may have duplicate values.

    Methods:
        get_row_by_key(key):
            Retrieves a single row from the dataset using the primary key.
        get_rows_by_index(index, idx):
            Retrieves all rows matching the specified index and value.
        get_all_unique_indices(index):
            Retrieves all unique values for a specified index.
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
        self.indices = indices or []
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
        logger.info(f"Primary key index built for '{self.primary_key}'")

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
        row_idx = self._key_to_row.get(key, -1)
        if row_idx == -1:
            raise KeyError(f"Key '{key}' not found in primary index")
        return self.dataset[row_idx]

    def get_rows_by_index(self, index, idx):
        if index not in self.indices:
            raise ValueError(f"Index {index} is not defined")
        return [self.dataset[i] for i in self._index_to_row.get(index, {}).get(idx, [])]
