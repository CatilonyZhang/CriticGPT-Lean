import concurrent
import pathlib

from datasets import Dataset, load_dataset
from tqdm import tqdm


class InputSharder:
    """
    A utility class to split large Parquet files into smaller shards for efficient processing.

    Attributes:
        input_paths (list): List of input parquet file paths to be sharded.
        output_path (str): Directory where the shard files will be saved.
        shard_size (int): Number of rows per shard. Default is 10,000.
        columns_to_read (list): Columns to read from the input Parquet files.
            Default includes 'uuid', 'problem', 'answer', and 'source'.
        start_shard_id (int): Starting ID for the shard numbering. Default is 0.
    """

    def __init__(
        self,
        dataset_path,
        output_path,
        shard_size=10_000,
        columns_to_read=None,
        repeat=1,
    ):
        """
        Initialize the InputSharder class.

        Args:
            input_paths (list): List of input Parquet file paths.
            output_path (str): Directory where the shard files will be saved.
            shard_size (int): Number of rows per shard. Default is 10,000.
            columns_to_read (list): Columns to read from the Parquet files.
            repeat (int): Number of times to repeat the input dataset. Default is 1.
        """
        self.shard_size = shard_size
        if isinstance(dataset_path, str):
            dataset_path = load_dataset(dataset_path, split="train")
        elif not isinstance(dataset_path, Dataset):
            raise ValueError(
                "dataset_path must be either a string or a Dataset object."
            )
        self.input_ds = dataset_path
        n_shards = len(self.input_ds) // self.shard_size + 1
        self.digits = len(str(n_shards))
        self.output_path = pathlib.Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.columns_to_read = columns_to_read
        self.repeat = repeat
        self.shards = []
        self._parallel_shard_data()

    def _create_one_shard(self, shard_info):
        if not shard_info["shard_output_path"].exists():
            shard_ds = self.input_ds.select(
                range(shard_info["start_index"], shard_info["end_index"])
            )
            if self.columns_to_read:
                shard_ds = shard_ds.select_columns(self.columns_to_read)
            shard_ds.to_parquet(shard_info["shard_output_path"])

    def _parallel_shard_data(self):
        shard_id = 0
        shards_info = []

        for _ in range(self.repeat):
            for i in range(0, len(self.input_ds), self.shard_size):
                end_idx = min(i + self.shard_size, len(self.input_ds))
                shard_id_str = f"{shard_id:0{self.digits}d}"
                shard_filename = f"{shard_id_str}.parquet"
                shard_output_path = self.output_path / shard_filename

                shards_info.append(
                    {
                        "start_index": i,
                        "end_index": end_idx,
                        "shard_output_path": shard_output_path,
                    }
                )
                shard_id += 1
                self.shards.append([shard_id_str, shard_output_path])

        # Using ProcessPoolExecutor with tqdm to show progress
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit tasks to executor
            futures = [
                executor.submit(self._create_one_shard, shard_info)
                for shard_info in shards_info
            ]
            # Wrap the futures with tqdm to show progress
            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Sharding data",
            ):
                pass

    def __iter__(self):
        # read dataset by shards
        for shard_id, shard_output_path in self.shards:
            yield shard_id, shard_output_path
