from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from autoformalizer.data_utils.constants import system_prompt
from autoformalizer.data_utils.user_prompt import get_user_prompt
import pandas as pd
import numpy as np

from typing import List

from datasets import Dataset, load_dataset

def autoformalize(llm: LLM,
                  tokenizer: AutoTokenizer,
                  sampling_params: SamplingParams, 
                  natural_language: str, 
                  theorem_names: List[str], 
                  source: str, 
                  include_source: bool,
                  use_system_prompt: bool = True,
                  user_prompt_function = get_user_prompt):
    """
    Autoformalizes a given natural language mathematics statement using an LLM model.
    """

    # Prepare the prompt
    prompt = user_prompt_function(natural_language, has_header=True, 
                             theorem_names=theorem_names, 
                             source=source, 
                             include_source=include_source)

    # Create system and user messages
    if use_system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else: 
        messages = [
            {"role": "user", "content": prompt }
        ]

    # Apply the chat template for tokenizer
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate outputs from the model
    outputs = llm.generate([text], sampling_params)

    # Collect and return the generated formalizations
    autoformalizations = [output.outputs[0].text for output in outputs]
    return autoformalizations


def autoformalize_dataframe_batched(llm: LLM,
                                    tokenizer: AutoTokenizer,
                                    sampling_params: SamplingParams, 
                                    df: pd.DataFrame,
                                    batch_size: int = 1024,
                                    use_system_prompt: bool = True,
                                    user_prompt_function = get_user_prompt) -> pd.DataFrame:
    """
    Autoformalizes all data points in a given DataFrame in batches using an LLM model.
    Adds 'autoformalization_i' columns where i is the index of the sample from 1 to n_samples.
    
    Requirements:
    The input DataFrame must contain the following columns:
    - 'natural_language': The natural language statement to formalize.
    - 'theorem_names': A string that contains a list of theorem names for autoformalization separated by comma.
    - 'include_source': A boolean value indicating whether to include the source in the prompt.
    - 'source': A string containing the source of the statement (optional).
    """

    # Get the number of samples from the sampling parameters
    n_samples = sampling_params.n

    # Initialize empty lists for storing autoformalization samples
    autoformalizations = {f'autoformalization_{i+1}': [] for i in range(n_samples)}

    # Initialize list for storing prompts
    all_prompts = []

    # Iterate over the dataframe in batches
    for batch_start in range(0, len(df), batch_size):
        batch_df = df.iloc[batch_start:batch_start + batch_size]

        # Prepare batched prompts
        prompts = []
        for _, row in batch_df.iterrows():
            natural_language = row['natural_language']

            if isinstance(row['theorem_names'], np.ndarray):
                theorem_names = list(row['theorem_names'])
            elif isinstance(row['theorem_names'], list):
                theorem_names = row['theorem_names']
            elif isinstance(row['theorem_names'], str):
                theorem_names = row['theorem_names'].split(',')
            else:
                raise ValueError(f"The 'theorem_names' column must be a np.ndarray, list, or a string. "
                f"However, during runtime, an unexpected type of {type(row['theorem_names'])} was encountered.")
            if 'has_header' in row:
                has_header = row['has_header']
            else:
                has_header = True
            
            # If 'source' column does not exist, set source to empty string and include_source to False
            if 'source' in row:
                source = row['source']
                if 'include_source' in row:
                    include_source = row['include_source']
                else:
                    include_source = False
            else:
                source = ""
                include_source = False

            # Prepare the prompt for each row
            prompt = user_prompt_function(natural_language, has_header=has_header, 
                                     theorem_names=theorem_names, 
                                     source=source, 
                                     include_source=include_source)
            prompts.append(prompt)

        all_prompts.extend(prompts)

        # Prepare the text for the LLM
        messages = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ] if use_system_prompt else 
            [
                {"role": "user", "content": prompt}
            ]
            for prompt in prompts
        ]
        
        texts = [tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True) 
            for message in messages]

        # Generate outputs from the model in batch
        outputs = llm.generate(texts, sampling_params)

        # Collect and store the autoformalizations
        for i, output in enumerate(outputs):
            formalized_outputs = [out.text for out in output.outputs]
            for j in range(n_samples):
                autoformalizations[f'autoformalization_{j+1}'].append(formalized_outputs[j])

        #print(f"Processed batch {batch_start + 1} to {min(batch_start + batch_size, len(df))}", end='\r')
    
    # Add the new columns to the dataframe
    for i in range(n_samples):
        df[f'autoformalization_{i+1}'] = autoformalizations[f'autoformalization_{i+1}']

    df['prompt'] = all_prompts

    return df


def autoformalize_dataset_batched(
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    dataset: Dataset,
    batch_size: int = 1024,
    use_system_prompt: bool = True,
    user_prompt_function = get_user_prompt
) -> Dataset:
    """Processes a HuggingFace dataset to generate formal theorems using an LLM model in batches.

    Args:
        llm (LLM): The LLM model instance for theorem generation
        tokenizer (AutoTokenizer): Tokenizer corresponding to the LLM model
        sampling_params (SamplingParams): Parameters controlling the LLM generation
        dataset (Dataset): HuggingFace dataset containing input statements
        batch_size (int): Number of samples to process in parallel
        use_system_prompt (bool): Whether to include system prompt in generation
        user_prompt_function (callable): Function to generate prompts from input data

    Returns:
        Dataset: Original dataset with added columns:
            - autoformalization_{i}: Generated theorems for each sample i (1 to n_samples)
            - prompt: Generated prompts used for each input

    Input Dataset Requirements:
        Dataset must contain these columns:
        - 'natural_language': Natural language statements to formalize
        - 'theorem_names': Comma-separated theorem names or list/array
        - 'include_source': Boolean flag for source inclusion
        - 'has_header': Boolean flag for header inclusion (optional)
        - 'source': Source reference (optional)

    Example:
        >>> dataset = load_dataset("math/theorems", split="train")
        >>> model = LLM("path/to/model")
        >>> tokenizer = AutoTokenizer.from_pretrained("path/to/model")
        >>> params = SamplingParams(n=2, temperature=0.7)
        >>> processed_dataset = autoformalize_dataset_batched(
        ...     model, tokenizer, params, dataset, batch_size=32
        ... )
    """
    n_samples = sampling_params.n


    def process_batch(examples):
        # Prepare prompts for the batch
        prompts = []
        default_headers = [True] * batch_size
        default_sources = [''] * batch_size
        default_include_sources = [False] * batch_size
        for idx in range(len(examples['natural_language'])):
            natural_language = examples['natural_language'][idx]
            
            # Handle different theorem_names formats
            theorem_names = examples['theorem_names'][idx]
            if isinstance(theorem_names, str):
                theorem_names = theorem_names.split(',')
                
            # Get optional fields with defaults matching batch size
            has_header = examples.get('has_header', default_headers)[idx]
            source = examples.get('source', default_sources)[idx]
            include_source = examples.get('include_source', default_include_sources)[idx]

            prompt = user_prompt_function(
                natural_language,
                has_header=has_header,
                theorem_names=theorem_names,
                source=source,
                include_source=include_source
            )
            prompts.append(prompt)

        # Prepare messages and generate
        messages = [
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": prompt}] if use_system_prompt else
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        texts = [
            tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            ) for message in messages
        ]

        outputs = llm.generate(texts, sampling_params)

        # Format results
        result = {'prompt': prompts}
        for i in range(n_samples):
            result[f'autoformalization_{i+1}'] = [
                output.outputs[i].text for output in outputs
            ]
        
        return result

    # Process dataset in batches
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names
    )

    return processed_dataset