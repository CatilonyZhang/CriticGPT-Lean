import fire
import pandas as pd
from autoformalizer.eval_utils.model_feedback import hf_model_feedback
from autoformalizer.eval_utils.lean_feedback import hf_lean4_feedback



def all_feedback(input_dataset_id: str,
                input_dataset_branch: str,
                output_dataset_id: str,
                output_dataset_branch: str,
                filter_compiled: bool = True,
                model_name: str = "claude-3-5-latest",
                verbose: bool = True):

    # Get the Lean4 feedback
    hf_lean4_feedback(input_dataset_id,
                           input_dataset_branch,
                           output_dataset_id,
                           output_dataset_branch,
                           verbose=verbose)

    # Get the GPT feedback
    # hf_model_feedback(input_dataset_id,
    #                 input_dataset_branch, 
    #                 output_dataset_id, 
    #                 output_dataset_branch, 
    #                 filter_compiled=filter_compiled,
    #                 model_name=model_name,
    #                 verbose=verbose)

if __name__ == "__main__":
    '''
    Example usage:
    python -m autoformalizer.eval_utils.all_feedback \
    --input_dataset_id="AI-MO/AutoformalizationEvalV2" \
    --input_dataset_branch="Qwen7BCoder_AutoformalizerV1B3" \
    --output_dataset_id="AI-MO/AutoformalizationEvalV2" \
    --output_dataset_branch="Qwen7BCoder_AutoformalizerV1B3"

    '''
    fire.Fire(all_feedback)       fire.Fire(all_feedback)   