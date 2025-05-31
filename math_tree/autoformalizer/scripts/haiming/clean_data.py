import datasets

# data = datasets.load_dataset("AI-MO/auto-statements-moon-santa-prover-v1",
#  split="train", cache_dir="/mnt/moonfs/kimina-m2/.cache/ttmmpp")
data = datasets.load_from_disk(
    "/mnt/moonfs/kimina-m2/.cache/auto-statements-moon-santa-prover-v1-cleanup-hard"
)

# data = data.select(range(1000))
data_df = data.to_pandas()
print("Original length:", len(data_df))

# get uuid that have n_correct_proofs > 30
uuid = data_df.groupby("uuid")["n_correct_proofs"].sum().reset_index()
uuid = uuid[uuid["n_correct_proofs"] > 20]["uuid"].unique()
uuid = set(uuid.tolist())

print("uuid:", len(uuid))

# filter out uuid that have n_correct_proofs < 30
data_df = data_df[~data_df["uuid"].isin(uuid)]
print("Filter uuid:", len(data_df))


print("1")
pos_statements = data_df[
    (data_df["is_negation"] is False) & (data_df["n_correct_proofs"] > 0)
]["statement_id"].unique()
neg_statements = data_df[
    (data_df["is_negation"] is True) & (data_df["n_correct_proofs"] > 0)
]["statement_id"].unique()

# filter out statements with n_correct_proofs > 10
# data_df = data_df[data_df["n_correct_proofs"] < 10]
print("2")
pos_statements = pos_statements.tolist()
neg_statements = neg_statements.tolist()

print("3")
pos_statements_mark = []
for statement in pos_statements:
    pos_statements_mark.append(f"neg_{statement}")
neg_statements_mark = []
for statement in neg_statements:
    neg_statements_mark.append(statement[len("neg_") :])
print("4")
remove_statements = set(pos_statements_mark + neg_statements_mark)
print("nega len", len(remove_statements))

data_df = data_df[~data_df["statement_id"].isin(remove_statements)]
print("5")
print("Remove negate", len(data_df))

# filter out statements with n_correct_proofs > 10
data_df = data_df[data_df["n_correct_proofs"] < 3]
print("6")

print("Remove 10 filter", len(data_df))

data = datasets.Dataset.from_pandas(data_df)
data.save_to_disk(
    "/mnt/moonfs/kimina-m2/.cache/auto-statements-moon-santa-prover-v1-cleanup-harder"
)
