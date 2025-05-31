from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pandas as pd
import fire

from autoformalizer.data_utils.user_prompt import get_user_prompt
from autoformalizer.model_utils.autoformalize import autoformalize_dataframe_batched

def infer_dataframe(model_path: str, 
                    input_csv: str, 
                    output_csv: str, 
                    n_samples: str = 1,
                    temperature: float = 0.0,
                    top_p: float = 1.0,
                    repetition_penalty: float = 1.05,
                    max_tokens: int = 512,
                    batch_size: int = 1024,
                    use_system_prompt: bool = True,
                    user_prompt_function = get_user_prompt):
    """
    Infers formalizations for a given DataFrame using an LLM model.

    Requirements:
    The input CSV file must contain the following columns:
    - 'natural_language': The natural language statement to formalize.
    - 'theorem_names': A list of theorem names to be used in autoformalization.
    - 'include_source': A boolean value indicating whether to include the source in the prompt.
    - 'has_header': A boolean value indicating whether to include a header in the autoformalization.
    - 'source': The source of the natural language statement. (optional)

    The output CSV file will contain the columns autoformalization_{i+1} for i in range(n_samples).

    """

    df = pd.read_csv(input_csv)

    llm = LLM(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_params = SamplingParams(n = n_samples,
                                    temperature=temperature,
                                    top_p=top_p,
                                    repetition_penalty=repetition_penalty,
                                    max_tokens=max_tokens)

    df = autoformalize_dataframe_batched(
        llm, 
        tokenizer, 
        sampling_params, 
        df, 
        batch_size=batch_size,
        use_system_prompt=use_system_prompt, 
        user_prompt_function=user_prompt_function,
    )

    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    """
    Example usage: (merge to single line first)
    python -m autoformalizer.model_utils.infer_dataframe --model_path="/workspace/mert/models/Qwen7B_Autoformalizer"  --input_csv="scripts/mert/number_theory_df.csv" --output_csv="scripts/mert/number_theory_autoformalized.csv"
    """
    fire.Fire(infer_dataframe)