import pandas as pd
import fire

from autoformalizer.eval_utils.lean_feedback import parallel_lean4_feedback


def dataframe_parallel_lean4_feedback(input_csv: str,
                                      output_csv: str,
                                      verbose: bool = True):
    '''
    Gets Lean 4 feedback for each formalization in a given DataFrame in parallel.

    Args:
    - input_csv: Path to the input CSV file.
    - output_csv: Path to the output CSV file.
    - verbose: Whether to print the results.

    Requirements:
    The input CSV file must contain the following columns:
    - 'autoformalization_i': The i-th autoformalization to get feedback for.

    The output CSV file will contain the following columns:
    - 'compiler_feedback_i': The feedback for the i-th autoformalization.
    - 'compiler_feedback_i_bool': The status of the i-th autoformalization. (True/False)
    '''

    df = pd.read_csv(input_csv)

    for column in df.columns:
        if str(column).startswith("autoformalization_"):
            idx = column.split("_")[-1]
            lean4_codes = df[column].tolist()
            feedbacks = parallel_lean4_feedback(lean4_codes)

            df[f"compiler_feedback_{idx}"] = [str(feedback) for feedback in feedbacks]
            df[f"compiler_feedback_{idx}_bool"] = ~df[f"compiler_feedback_{idx}"].str.contains("error")

            if verbose:
                val_counts = df[f"compiler_feedback_{idx}_bool"].value_counts()
                print(f"Feedback for autoformalization_{idx}:")
                print(val_counts)
                print(val_counts / val_counts.sum())

    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    '''
    Example usage:
    python -m autoformalizer.eval_utils.dataframe_lean_feedback \
    --input_csv="/home/mert/autoformalizer/scripts/mert/Autof181024.csv" \
    --output_csv="/home/mert/autoformalizer/scripts/mert/Autof181024_feedback.csv"
    '''
    fire.Fire(dataframe_parallel_lean4_feedback)