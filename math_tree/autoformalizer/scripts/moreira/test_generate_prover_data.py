from datasets import load_dataset
from generate_prover_data import build_lean4_wholeproof
from autoformalizer.eval_utils.lean_feedback import parallel_lean4_feedback

def test_build_lean4(
    dataset_id: str = "AI-MO/sft-numina-math-lean4-241122",
    dataset_branch: str = "main",
    dataset_split: str = "train",
):

    def add_full_proof(example):
        header = example["header"] + example["context"] + example["statement"]
        # remove the sorry from the end
        header = header.split("sorry")[0] + "\n"
        example["whole_proof"] = header + build_lean4_wholeproof(example["steps"])
        return example

    ds = load_dataset(dataset_id, revision=dataset_branch, split=dataset_split)
    ds = ds.map(add_full_proof, num_proc=42)

    feedbacks = parallel_lean4_feedback(ds["whole_proof"])
    feedbacks = [str(feedback) for feedback in feedbacks]
    feedback_bools = [
        (i, "error" not in feedback) for i, feedback in enumerate(feedbacks)
    ]

    mistakes = [fb[0] for fb in feedback_bools if fb[1] == False]
    print(len(mistakes))

    return ds, mistakes 

def check_if_mistakes_have_unsupported_tactics(ds, non_compiled_examples):
    original_ds = load_dataset("AI-MO/numina-math-lean4-241122", split="train")
    ex_unsuported_tact = [] 
    ex_have = []

    for i in non_compiled_examples:
        example = ds[i]
        id = example["file"].split(".")[0]
        filter = original_ds.filter(lambda x: str(x["id"]) == id)
        original_example = ""
        if len(filter) > 0:
            original_example = filter[0]
        else:
            print("Example id: ", id, "not found")
            
        has_let = False
        has_have = False
        for s in example["steps"]:
            has_let = has_let or ("let " in s["tactic"])
            has_have = has_have or ("have " in s["tactic"])
        
        sol = original_example["lean4_solution"]
        has_conv = ("conv " in sol) or ("conv_lhs" in sol) or ("conv_rhs" in sol)
        has_calc = ("calc " in sol)
        if has_let or has_conv or has_calc:
            ex_unsuported_tact.append(i)
        elif has_have:
            ex_have.append(i)

    print(len(ex_unsuported_tact), len(ex_have))
    return ex_unsuported_tact, ex_have

def main():
    ds, mistakes = test_build_lean4()
    check_if_mistakes_have_unsupported_tactics(ds, mistakes)

if __name__ == "__main__":
    main()