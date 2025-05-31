from datasets import load_dataset
import re

def detect_lean_version(code):
    lean3_patters = [
        r"^\s*split\b", # Old split syntax in the beginning of a line
        r",\s*\n", # No , to delimit tactic calls
        r"λ[^,]+,[^,]+", # Old lambda syntax
        r"\brw[^,\[]+," #Square brackets are mandatory in rw h,
    ]

    for pattern in lean3_patters:
        if re.search(pattern, code):
            return 3
        
    lean4_patterns = [
        r"\bimport\s+[A-Z]", # Casing is correct on imports
        r"\bfun[^,]+=>[^,]+", # New lambda syntax
        r"λ[^,]+=>[^,]+", # New lambda syntax
        r"\bfun[^,]+↦[^,]+",# New lambda syntax
        r"λ[^,]+↦[^,]+", # New lambda syntax
    ]
    
    for pattern in lean4_patterns:
        if re.search(pattern, code):
            return 4
    return None

def filter_dataset(ds_name, ds_split):
    ds = load_dataset(ds_name, split=ds_split)
    processed_ds = ds.map(lambda x: {"lean_version": detect_lean_version(x["content"])})
    print("Total length:", len(processed_ds))
    print("Lean 3 entries:", len(processed_ds.filter(lambda x: x["lean_version"] == 3)))
    print("Lean 4 entries:", len(processed_ds.filter(lambda x: x["lean_version"] == 4)))
    print("Inconclusive entries:", len(processed_ds.filter(lambda x: x["lean_version"] == None)))
    # Removes Lean 3 entries
    return processed_ds.filter(lambda x: x["lean_version"] in [4, None])

if __name__ == "__main__":
    ds_name = "AI-MO/the-stack-v2-lean"
    ds_split = "train"
    push_ds_name = "AI-MO/the-stack-v2-filtered-lean4"

    # Runs filter through dataset and exports
    # Expected length after filtering: 15086
    # Lean 4: 6467, Inconclusives: 8619
    # Filtered out Lean 3: 58770
    # Percentage of dataset: Lean4 ~ 8.7%, Lean 3 ~ 79.5%, Inconclusive ~ 11.7%
    lean4_ds = filter_dataset(ds_name, ds_split)

    lean4_ds.push_to_hub(push_ds_name, private=True)