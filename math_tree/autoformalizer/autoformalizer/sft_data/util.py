import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import click
import numpy as np
import openai
from datasets import Dataset, load_dataset, load_from_disk
from loguru import logger
from pandas import DataFrame
from tqdm import tqdm


def _proof_using_0(sample):
    proofs = sample["correct_proof_samples"]
    return any("use 0" in proof for proof in proofs)


def longest_common_sublist(list1, list2):
    m = len(list1)
    n = len(list2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    end_index = -1

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_index = i - 1
            else:
                dp[i][j] = 0

    if max_length == 0:
        return []
    start_index = end_index - max_length + 1
    return list1[start_index : end_index + 1]


def find_different_proof(input_list, n, col="shortest_proof"):
    proof_list = input_list[col].values.tolist()
    overlaps = {}
    for s1, s2 in combinations(proof_list, 2):
        overlap = longest_common_sublist(s1.split(), s2.split())
        overlaps[(s1, s2)] = len(overlap)
    least_similar_proof_list = []
    index_list = []
    while len(least_similar_proof_list) < n and proof_list:
        min_distance = 999
        best_string = None
        i = 0
        for string in proof_list:
            if i in index_list:
                i += 1
                continue
            total_overlap = sum(
                overlaps.get((string, s), 0) for s in least_similar_proof_list
            )
            if total_overlap < min_distance:
                min_distance = total_overlap
                best_string = string
                index_list.append(i)
            i += 1
        least_similar_proof_list.append(best_string)
    output_list = input_list.iloc[index_list]
    return output_list


def filter_by_str_length(df: DataFrame, min_length: int = 3, max_length: int = 3000):
    return df[df["proof_output"].str.len().between(min_length, max_length)]


def filter_by_line_length(df: DataFrame, min_length: int = 2, max_length: int = 150):
    return df[
        df["proof_output"].str.split("\n").apply(len).between(min_length, max_length)
    ]


def filter_by_token_length(
    df: DataFrame,
    min_length: int = 3,
    max_length: int = 1500,
    tokenizer_path: str = "/mnt/moonfs/kimina-m2/models/qwen/Qwen2.5-Coder-7B-Instruct",
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    return df[
        df["proof_output"]
        .apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))
        .between(min_length, max_length)
    ]


def filter_by_near_dedup(
    dataset: Dataset,
    source_name: str,
    output_path: str,
    threshold: float = 0.7,
    dedup_column: str = "proof_output",
    num_perm: int = 250,
    min_length: int = 0,
    ngram: int = 5,
):
    """
    Refer to the following for more details:
    https://huggingface.co/blog/dedup
    https://github.com/ChenghaoMou/text-dedup

    """

    try:
        from text_dedup.minhash import main as minhash_main
        from text_dedup.utils import IOArgs, MetaArgs, MinHashArgs
        from text_dedup.utils.timer import Timer
    except ImportError:
        raise ImportError("Run ``pip install text-dedup``.")

    cache_dir = os.environ.get("HF_HOME", ".cache")
    cache_dir = os.path.join(cache_dir, source_name, "train")
    dataset.save_to_disk(cache_dir)

    t = Timer()

    io_args = IOArgs(
        path=cache_dir,
        local=True,
        num_proc=128,
        cache_dir=cache_dir,
        output=output_path,
        debug=False,
        clean_cache=True,
    )
    meta_args = MetaArgs(column=dedup_column, batch_size=10000)

    with t("MinHash"):
        ctx = click.Context(minhash_main)
        minhash_args = MinHashArgs(
            num_perm=num_perm,
            ngram=ngram,
            min_length=min_length,
            threshold=threshold,
        )
        io_args.output = output_path
        ctx.invoke(
            minhash_main,
            io_args=io_args,
            meta_args=meta_args,
            minhash_args=minhash_args,
        )

    logger.info(f"Processed dataset written to {output_path}.")

    # clear cache
    if os.path.exists(cache_dir):
        os.system(f"rm -rf {cache_dir}")

    output_dataset = load_from_disk(str(output_path))
    return output_dataset


def get_embedding(text, model):
    response = openai.embeddings.create(
        input=[text] if isinstance(text, str) else text,
        model=model,
    )
    embedding = response.data[0].embedding
    return embedding


def filter_by_gt_embedding(
    df: DataFrame,
    target_column: str,
    gt_dataset: str,
    similar_threshold: float = 0.6,
    model: str = "text-embedding-ada-002",
    base_url: str = "http://localhost:8000/v1/",
    api_key: str = "EMPTY",
    num_parallel: int = 4,
    batch_size: int = 100,
) -> DataFrame:

    openai.api_key = api_key
    openai.base_url = base_url

    gt_dataset = load_dataset(gt_dataset, split="train", num_proc=num_parallel)
    mask = [len(text) < 8000 for text in gt_dataset[target_column]]
    gt_dataset = gt_dataset.filter(lambda _, idx: mask[idx], with_indices=True)
    gt_texts = gt_dataset[target_column]

    gt_embeddings = []
    for i in range(0, len(gt_texts), batch_size):
        batch_gt = gt_texts[i : i + batch_size]
        if num_parallel > 1:
            with ProcessPoolExecutor(max_workers=num_parallel) as executor:
                batch_embeddings = list(
                    executor.map(get_embedding, batch_gt, [model] * len(batch_gt))
                )
        else:
            batch_embeddings = [get_embedding(text, model) for text in batch_gt]
        gt_embeddings.extend(batch_embeddings)

    gt_embeddings_np = np.array(gt_embeddings)
    gt_norms = np.linalg.norm(gt_embeddings_np, axis=1)
    del gt_embeddings

    texts = df[target_column].tolist()
    cos_sim_list = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]

        if num_parallel > 1:
            with ProcessPoolExecutor(max_workers=num_parallel) as executor:
                batch_embeddings = list(
                    executor.map(get_embedding, batch_texts, [model] * len(batch_texts))
                )
        else:
            batch_embeddings = [
                get_embedding(text, model) for text in tqdm(batch_texts)
            ]

        batch_embeddings_np = np.array(batch_embeddings)
        text_norms = np.linalg.norm(batch_embeddings_np, axis=1)

        dot_products = np.dot(batch_embeddings_np, gt_embeddings_np.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            similarities = dot_products / np.outer(text_norms, gt_norms)

        batch_max_sim = np.nanmax(similarities, axis=1)
        cos_sim_list.extend(batch_max_sim.tolist())

        del batch_embeddings, batch_embeddings_np, dot_products, similarities

    df["cos_sim"] = cos_sim_list
    filtered_df = df[df["cos_sim"] > similar_threshold].reset_index(drop=True)

    return filtered_df
