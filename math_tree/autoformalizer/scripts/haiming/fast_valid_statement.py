

import uuid
import datasets
import json

OUTPUT_COLUMNS = set(["statement_id", "natural_language", "is_negation", "uuid", "formal_statement", "tags", "ground_truth"])


def generate_deterministic_uuid(input_string):
    # You can use any namespace, here we are using uuid.NAMESPACE_DNS
    return uuid.uuid5(uuid.NAMESPACE_DNS, input_string)

def miniF2F_formatter(sample):
    prompt = f"{sample["header"]}\n{sample["informal_prefix"]}\n{sample["formal_statement"]}"
    return prompt


def handle_miniF2F(dataset, source):
    def process_sample(sample):
        sample["uuid"] = str(generate_deterministic_uuid(sample["formal_statement"]))
        sample["statement_id"] = sample["uuid"]
        sample["natural_language"] = sample["informal_prefix"]
        sample["is_negation"] = False
        sample["formal_statement"] = miniF2F_formatter(sample)
        sample["tags"] = [f"human-statements-source:{source}"]
        sample["ground_truth"] = ""
        return sample

    dataset = dataset.map(process_sample, num_proc=4)
    return dataset

def handle_annodata(dataset):
    def process_sample(sample):
        sample["is_negation"] = False
        if "tags" in sample:
            if isinstance(sample["tags"], str):
                sample["tags"] = json.loads(sample["tags"])
            sample["tags"].append(f"human-statements-source:inhouse-evaluation-v1-20250102-train")
        else:
            sample["tags"] = [f"human-statements-source:inhouse-evaluation-v1-20250102-train"]
        sample["ground_truth"] = sample["human_proof"]
        return sample

    dataset = dataset.map(process_sample, num_proc=4)
    return dataset

def handle_putman(dataset):
    def process_sample(sample):
        sample["statement_id"] = sample["uuid"]
        sample["natural_language"] = sample["informal_statement"]
        sample["is_negation"] = False
        sample["formal_statement"] = sample["full_text_no_abbrv"]
        sample["ground_truth"] = ""
        if "tags" in sample:
            if isinstance(sample["tags"], str):
                sample["tags"] = json.loads(sample["tags"])
            sample["tags"].append(f"human-statements-source:PutnamBench-lean4-train")
        else:
            sample["tags"] = [f"human-statements-source:PutnamBench-lean4-train"]
        return sample

    dataset = dataset.map(process_sample, num_proc=4)
    return dataset


anno_data = datasets.load_dataset("AI-MO/inhouse-evaluation-v1-20250102", split="train")
miniF2F_valid = datasets.load_dataset("HaimingW/miniF2F-lean4", split="valid")
proofnet_valid = datasets.load_dataset("HaimingW/proofnet-lean4", split="valid")
putnam_train = datasets.load_dataset("HaimingW/PutnamBench-lean4", split="train")

miniF2F_valid = handle_miniF2F(miniF2F_valid, "miniF2F-valid")
proofnet_valid = handle_miniF2F(proofnet_valid, "proofnet-valid")
anno_data = handle_annodata(anno_data)
putnam_train = handle_putman(putnam_train)

# keep only the output columns
anno_data = anno_data.remove_columns([col for col in anno_data.column_names if col not in OUTPUT_COLUMNS])
miniF2F_valid = miniF2F_valid.remove_columns([col for col in miniF2F_valid.column_names if col not in OUTPUT_COLUMNS])
proofnet_valid = proofnet_valid.remove_columns([col for col in proofnet_valid.column_names if col not in OUTPUT_COLUMNS])
putnam_train = putnam_train.remove_columns([col for col in putnam_train.column_names if col not in OUTPUT_COLUMNS])

merged_dataset = datasets.concatenate_datasets([anno_data, miniF2F_valid, proofnet_valid, putnam_train])
print(merged_dataset)

# Save the merged dataset
merged_dataset.push_to_hub("AI-MO/human-statements-dataset-v1-20250103", private=True)


