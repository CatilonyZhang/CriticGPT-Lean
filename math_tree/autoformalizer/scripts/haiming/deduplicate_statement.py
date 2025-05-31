import datasets

data = datasets.load_from_disk(
    "/mnt/moonfs/kimina-m2/.cache/auto-statements-moon-santa-prover-v1-cleanup-hard-informal_added"
)
# data = data.select(range(1000))

data_df = data.to_pandas()
print("Original length:", len(data))


data_df = data_df.drop_duplicates(subset=["formal_statement"], keep="first")
data_df.drop(columns=["__index_level_0__"])

print("Deduplicate length:", len(data_df))
data = datasets.Dataset.from_pandas(data_df, preserve_index=False)
# data.save_to_disk("/mnt/moonfs/kimina-m2/.cache/auto-statements-moon-santa-prover-v1-cleanup-hard-informal_added-dedup")

data.push_to_hub("AI-MO/auto-statements-moon-flying-ant-v1-20250110", private=True)
