from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset
from autoformalizer.model_utils.autoformalize import autoformalize_dataframe_batched
from autoformalizer.model_utils.infer_hf_dataset import infer_hf_dataset
from autoformalizer.eval_utils.lean_feedback import hf_lean4_feedback
import re


def prompt_function(
    natural_language_desc: str,
    has_header: bool,
    theorem_names: list,
    source: str,
    include_source: bool,
):
    user_prompt = """Mathematical Problem in Natural Language:
    {}
    Translate the problem to Lean 4 (only the core declaration, no solution):
    """.format(
        natural_language_desc
    )
    return user_prompt


def extract_lean4(output: str):
    """
    Takes output string, and return all the substrings that start by
    'theorem' and end by ':='
    """
    pttn_code = r"```lean4?.*?(?:```|$)"
    pttn_inline_com = r"--.*?\n"
    pttn_multi_com = r"/-(?:-|-!)?(?:[^-]|-[^/])*-/"
    pttn_newlines = r"\n+"
    output = re.sub(pttn_multi_com, "", output)
    output = re.sub(pttn_inline_com, "", output)
    output = re.sub(pttn_newlines, "\n", output)
    code_blocks = re.findall(pttn_code, output, re.DOTALL)
    code_blocks = [re.sub("```lean4?", "", c) for c in code_blocks]
    code_blocks = [re.sub("```", "", c) for c in code_blocks]
    code_blocks = [re.sub("<|im_end|>user", "", c) for c in code_blocks]
    code_blocks = [re.sub("<|im_end|>", "", c) for c in code_blocks]
    code_blocks = [re.sub("PROOFSTEP", "", c) for c in code_blocks]
    return "\n".join(code_blocks)


def format_autoformalization(dataset_path, revision_name):
    ds = load_dataset(dataset_path, revision=revision_name, split="train")
    autof_columns = [column for column in ds.column_names if str(column).startswith("autoformalization_")]
    for col in autof_columns:
        ds = ds.map(
            lambda x: {col: extract_lean4(x[col])}
        )
    ds.push_to_hub(
        dataset_path,
        revision=revision_name,
        private=True,
        commit_message=f"Extract lean4 code of model output for autoformalization",
    )

def main():
    model_path = "deepseek-ai/DeepSeek-Prover-V1.5-RL"
    dataset_path = "AI-MO/AutoformalizationEvalV2"

    model_name = model_path.split("/")[-1]
    dataset_revision_name = f"{model_name}-EVAL"
    
    infer_hf_dataset(
        model_path=model_path,
        dataset_id=dataset_path,
        output_dataset_id=dataset_path,
        dataset_branch="main",
        output_dataset_branch=dataset_revision_name,
        n_samples=1,
        temperature=1.0,
        max_tokens=2048,
        top_p=0.95,
        repetition_penalty=1.0,
        use_system_prompt=True,
        user_prompt_function=prompt_function,
    )
    format_autoformalization(dataset_path, dataset_revision_name)
    hf_lean4_feedback(dataset_path, dataset_revision_name, dataset_path, dataset_revision_name)
    df = load_dataset(dataset_path, revision=dataset_revision_name, split="train").to_pandas()
    # Check for compilation bool and whether autoformalization is non-empty
    print(
        "Autoformalization compile rate is: ",
        100 * (len( df[ (df.autoformalization_1 != "") & (df.compiler_feedback_1_bool == True) ]) / len(df)),
        "%",
    )


if __name__ == "__main__":
    main()
