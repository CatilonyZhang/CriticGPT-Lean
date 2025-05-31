import os

import fire
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
from autoformalizer.data_utils.negate_theorem import (
    convert_braces_to_parentheses,
    extract_theorem_components,
)
from autoformalizer.data_utils.user_prompt import get_user_prompt
from autoformalizer.model_utils.autoformalize import (
    autoformalize_dataframe_batched,
    autoformalize_dataset_batched,
)
import pdb

def negate_theorem(text):
    """Negates a theorem by applying logical negation according to standard mathematical logic rules.

    Args:
        text (str): Input text containing a theorem in Lean format.
                   Expected format: 'theorem name (params) (hyps) : conclusion := by sorry'
                   where:
                   - name: theorem identifier
                   - params: type parameters like (x y : ℝ)
                   - hyps: hypotheses like (h : x > 0)
                   - conclusion: the theorem's conclusion

    Returns:
        str: The negated theorem in Lean format.
             Returns None if input cannot be parsed as a valid theorem.

    Negation Rules:
        1. For uniqueness theorems (conclusion is an equality):
           - Original: ∀ params, hyps → (x = a)
           - Negated:  ∃ params, hyps ∧ ¬(x = a)

        2. For universal theorems (∀ quantified conclusion):
           - Original: ∀ params, hyps → P(x)
           - Negated:  ∃ params, hyps ∧ ¬P(x)

        3. For existential theorems (∃ quantified conclusion):
           - Original: ∃ params, hyps ∧ P(x)
           - Negated:  ∀ params, hyps → ¬P(x)

    Examples:
        >>> # Negating a uniqueness theorem
        >>> negate_theorem("theorem algebra_5 (x y : ℝ) (h : x^2 - 6*x + y^2 + 2*y = 9) : (x, y) = (3, -1) := by sorry")
        'theorem negated_algebra_5 : ∃ (x y : ℝ) (h : x^2 - 6*x + y^2 + 2*y = 9), ¬((x, y) = (3, -1)) := by sorry'

        >>> # Negating a universal theorem
        >>> negate_theorem("theorem all_positive (n : ℕ) : n > 0 := by sorry")
        'theorem negated_all_positive : ∃ (n : ℕ), ¬(n > 0) := by sorry'
    """
    components = extract_theorem_components(text)
    if components is None:
        return None
    lib_name = components["lib_str"]
    theorem_name = components["theorem_name"]
    variables_and_hypotheses = components["variables_and_hypotheses"]
    conclusion = components["conclusion"]

    # Build the existential quantifiers
    if variables_and_hypotheses:
        quantifiers = (
            "    ∃ " + convert_braces_to_parentheses(variables_and_hypotheses) + ", "
        )
    else:
        quantifiers = ""

    # Add '¬' in front of the conclusion
    negated_conclusion = f"    ¬ ({conclusion})"

    # Build the negated theorem
    negated_theorem_name = "negated_" + theorem_name
    lean_code = f"""{lib_name}
theorem {negated_theorem_name} :
{quantifiers}
{negated_conclusion} := by sorry"""

    return lean_code


def infer_hf_dataset(
    model_path: str,
    dataset_id: str,
    output_dataset_id: str,
    dataset_branch: str = "main",
    output_dataset_branch: str = "main",
    n_samples: str = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    max_tokens: int = 1024,
    batch_size: int = 1024,
    use_system_prompt: bool = True,
    user_prompt_function=get_user_prompt,
):
    """
    Infers formalizations for a given DataFrame using an LLM model.

    Requirements:
    The input CSV file must contain the following columns:
    - 'natural_language': The natural language statement to formalize.
    - 'theorem_names': A list of theorem names to be used in autoformalization.
    - 'include_source': A boolean value indicating whether to include the source in the prompt.
    - 'has_header': A boolean value indicating whether to include a header in the autoformalization.
    - 'source': The source of the natural language statement. (optional)

    The output HuggingFace dataset will contain the columns autoformalization_{i+1} for i in range(n_samples).

    """

    ds = load_dataset(dataset_id, split="train", revision=dataset_branch)

    df = ds.to_pandas()
    llm = LLM(model_path, download_dir="/lustre/fast/fast/txiao/zly/lean/math_tree/ckpt")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
    )

    df = autoformalize_dataframe_batched(
        llm,
        tokenizer,
        sampling_params,
        df,
        batch_size=batch_size,
        use_system_prompt=use_system_prompt,
        user_prompt_function=user_prompt_function,
    )

    for i in range(n_samples):
        col_name = f"autoformalization_{i + 1}"
        autoformalizations = df[col_name]
        ds = ds.add_column(col_name, autoformalizations)
        
    ds.push_to_hub(
        output_dataset_id,
        revision=output_dataset_branch,
        private=True,
        commit_message=f"Inferred autoformalizations using {model_path} model.",
    )


def infer_hf_dataset_with_negated_sample(
    model_path: str,
    dataset_id: str,
    output_dataset_id: str,
    dataset_branch: str = "main",
    output_dataset_branch: str = "main",
    n_samples: str = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    max_tokens: int = 1024,
    batch_size: int = 1024,
    use_system_prompt: bool = True,
    user_prompt_function=get_user_prompt,
):
    """Generates formal mathematical theorems and their negations from natural language statements using an LLM.

    Args:
        model_path (str): Path to the LLM model to use for theorem generation
        dataset_id (str): HuggingFace dataset ID containing the input statements
        output_dataset_id (str): HuggingFace dataset ID where results will be saved
        dataset_branch (str): Branch/revision of input dataset to use
        output_dataset_branch (str): Branch name for saving output dataset
        n_samples (int): Number of theorem variations to generate per statement
        temperature (float): Sampling temperature for LLM generation
        top_p (float): Top-p nucleus sampling parameter
        repetition_penalty (float): Penalty factor for repeated tokens
        max_tokens (int): Maximum number of tokens to generate per theorem
        batch_size (int): Number of statements to process in parallel
        use_system_prompt (bool): Whether to prepend system prompt to input
        user_prompt_function (callable): Function to generate prompts from statements

    Returns:
        None: Results are saved directly to HuggingFace Hub

    Input Dataset Format:
        The input dataset must contain the following columns:
        - 'natural_language': Natural language mathematical statements
        - 'theorem_names': Names to assign to generated theorems
        - 'include_source': Boolean flags for source inclusion
        - 'has_header': Boolean flags for header inclusion
        - 'source': (Optional) Source references for statements

    Output Dataset Format:
        The output dataset contains the original columns plus:
        For each sample i (0 to n_samples-1):
        - autoformalization_{i+1}: Generated formal theorems
        - autoformalization_{i+1}_negated: Negated versions of theorems

    Example:
        >>> infer_hf_dataset_with_negated_sample(
        ...     model_path="EleutherAI/mathematician",
        ...     dataset_id="mathematics/theorems",
        ...     output_dataset_id="mathematics/formal_theorems",
        ...     n_samples=2,
        ...     batch_size=32
        ... )
        # Generates formal theorems and their negations, saves to HF Hub
    """

    ds = load_dataset(dataset_id, split="train", revision=dataset_branch)

    # llm = LLM(model_path, download_dir=f"{os.getenv('math_tree')}/ckpt")
    llm = LLM(model_path, download_dir='/lustre/fast/fast/txiao/zly/lean/math_tree/ckpt')
    # huggingface download model to os.getenv('math_tree')/model_path
    

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
    )

    # Process dataset with autoformalization
    processed_ds = autoformalize_dataset_batched(
        llm,
        tokenizer,
        sampling_params,
        ds,
        batch_size=batch_size,
        use_system_prompt=use_system_prompt,
        user_prompt_function=user_prompt_function,
    )

    # Add negated versions using dataset map
    def add_negations(examples):
        result = {}
        for i in range(n_samples):
            col_name = f"autoformalization_{i + 1}"
            negated_col_name = f"{col_name}_negated"
            result[negated_col_name] = [
                negate_theorem(text) for text in examples[col_name]
            ]
        return result

    final_ds = processed_ds.map(
        add_negations, batched=True, batch_size=batch_size, desc="Generating negations"
    )

    # Merge new columns into original dataset
    new_columns = [col for col in final_ds.column_names if col != "prompt"]
    for col in new_columns:
        ds = ds.add_column(col, final_ds[col])

    # Push to HuggingFace Hub
    ds.push_to_hub(
        output_dataset_id,
        revision=output_dataset_branch,
        private=True,
        commit_message=f"Inferred autoformalizations with negated samples using {model_path} model.",
    )


def infer_hf_dataset_by_batch(
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    dataset: Dataset,
    batch_size: int,
    output_column_name: str,
):
    """
    Infers formalizations for a given HuggingFace dataset using an LLM model by batch.

    Args:
    - llm: The LLM model to use for inference.
    - tokenizer: The tokenizer to use for inference.
    - dataset: The HuggingFace dataset to infer formalizations for.
    - batch_size: The batch size to use for inference.

    Returns:
    - The HuggingFace dataset with the output_column_name added as a new column.
    """

    # infer the autoformalizations by batch
    def infer_autoformalizations(batch):
        messages = batch["messages"]
        texts = [
            tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            for message in messages
        ]

        outputs = llm.generate(texts, sampling_params)
        autoformalizations = {
            output_column_name: [
                [gen.text for gen in output.outputs] for output in outputs
            ]
        }
        return autoformalizations

    # one can use num_proc > 1 which can squeeze even more performance but with OOMs risk
    dataset = dataset.map(infer_autoformalizations, batched=True, batch_size=batch_size)
    return dataset


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Infer formalizations for a HuggingFace dataset")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the LLM model to use for inference",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="HuggingFace dataset ID containing the input statements",
    )
    parser.add_argument(
        "--output_dataset_id",
        type=str,
        required=True,
        help="HuggingFace dataset ID where results will be saved",
    )

    parser.add_argument(
        "--dataset_branch",
        type=str,
        default="main",
        help="Branch/revision of input dataset to use",
    )
    parser.add_argument(
        "--output_dataset_branch",
        type=str,
        default="main",
        help="Branch name for saving output dataset",
    )
    
    infer_hf_dataset(**vars(parser.parse_args()))



if __name__ == "__main__":
    """
    Example usage:
    We store the autoformalizations by the model in the same dataset but in different branch.
    python -m autoformalizer.model_utils.infer_hf_dataset infer_hf_dataset \
    --model_path="AI-MO/Qwen7BCoder_Autoformalizer"  \
    --dataset_id="AI-MO/AutoformalizationEvalV2" \
    --output_dataset_id="AI-MO/AutoformalizationEvalV2" \
    --output_dataset_branch="Qwen7B_AutoformalizerV1B4"
    """
    main()
