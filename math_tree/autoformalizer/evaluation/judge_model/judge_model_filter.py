import argparse
import logging

from autoformalizer.eval_utils.model_feedback_custom import get_judge_feedback


def main():
    """Main function to run model feedback evaluation.

    Example usage from command line:
    Basic usage:
    python judge_model_eval.py \
        --input_dataset_id="AI-MO/formalized_preview" \

    Advanced usage with all parameters:
    python judge_model_eval.py \
        --input_dataset_id="AI-MO/formalized_preview" \
        --input_branch="test_branch" \
        --output_dataset_id="AI-MO/results" \
        --verbose

    Using the provided shell script:
    bash judge_model_eval.sh "gpt-4o" "main" "AI-MO/formalized_preview_241210"
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run model feedback evaluation")
    parser.add_argument(
        "--input_dataset_id",
        type=str,
        default="AI-MO/formalized_preview_241210",
        help="Input dataset ID",
    )
    parser.add_argument(
        "--input_branch", type=str, default="branch", help="Input dataset branch"
    )

    parser.add_argument(
        "--nl_key", type=str, default="problem", help="natural language key"
    )

    parser.add_argument(
        "--fl_key",
        type=str,
        default="autoformalization_preview",
        help="formal language key",
    )

    parser.add_argument("--output_dataset_id", type=str, help="Output dataset ID")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model sampling (0.0 to 1.0). Lower values make output more deterministic",
    )
    parser.add_argument(
        "--columns_to_use",
        type=str,
        nargs="+",  # This allows one or more arguments
        default=["problem", "autoformalization_preview"],
        help="List of column names needed for prompt template (default: ['problem', 'autoformalization_preview'])",
    )
    parser.add_argument(
        "--num_votes",
        type=int,
        default=1,
        help="Number of votes per input for majority voting (default: 3)",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=-1,
        help="Number of votes per input for majority voting (default: 3)",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Log the configuration
    logging.info("Starting model feedback evaluation with configuration:")
    logging.info(f"Input dataset ID: {args.input_dataset_id}")
    logging.info(f"Input branch: {args.input_branch}")
    logging.info(f"decoding temperature: {args.temperature}")
    logging.info(f"voting number: {args.num_votes}")
    logging.info(f"sample size: {args.sample_size}")
    logging.info(f"column to use: {args.columns_to_use}")

    # Handle default output dataset
    if not args.output_dataset_id:
        args.output_dataset_id = args.input_dataset_id
        logging.info(f"Using input dataset as output: {args.output_dataset_id}")

    # Validate inputs
    if not args.input_dataset_id:
        raise ValueError("Input dataset ID is required parameters")

    # Run the evaluation
    logging.info("Starting evaluation...")

    get_judge_feedback(
        input_dataset_id=args.input_dataset_id,
        input_dataset_branch=args.input_branch,
        output_dataset_id=args.output_dataset_id,
        sample_size=args.sample_size,
        verbose=args.verbose,
        temperature=args.temperature,
        num_votes=args.num_votes,
        columns_to_use=args.columns_to_use,
    )

    logging.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
