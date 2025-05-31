import argparse
import logging

from autoformalizer.eval_utils.model_feedback import evaluate_judge_model


def main():
    """Main function to run model feedback evaluation.

    Example usage from command line:
    Basic usage:
    python judge_model_eval.py \
        --input_dataset_id="AI-MO/formalized_preview" \
        --model_name="gpt-4"

    Advanced usage with all parameters:
    python judge_model_eval.py \
        --input_dataset_id="AI-MO/formalized_preview" \
        --input_branch="test_branch" \
        --output_dataset_id="AI-MO/results" \
        --model_name="claude-3" \
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
    parser.add_argument("--output_dataset_id", type=str, help="Output dataset ID")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

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
    logging.info(f"Model name: {args.model_name}")

    # Handle default output dataset
    if not args.output_dataset_id:
        args.output_dataset_id = args.input_dataset_id
        logging.info(f"Using input dataset as output: {args.output_dataset_id}")

    # Construct output branch name
    output_branch = f"model_{args.model_name}_evaluation_{args.input_branch}"
    logging.info(f"Output branch: {output_branch}")

    # Validate inputs
    if not args.input_dataset_id or not args.model_name:
        raise ValueError("Input dataset ID and model name are required parameters")

    # Run the evaluation
    logging.info("Starting evaluation...")
    evaluate_judge_model(
        input_dataset_id=args.input_dataset_id,
        input_dataset_branch=args.input_branch,
        output_dataset_id=args.output_dataset_id,
        output_dataset_branch=output_branch,
        model_name=args.model_name,
        verbose=args.verbose,
    )

    logging.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
