import argparse
import concurrent.futures
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum, auto
from itertools import cycle
from typing import Dict, List, Optional, Union

import anthropic  # Added import for Anthropic API
import pandas as pd
import yaml
from datasets import load_dataset
from openai import OpenAI
from tqdm.auto import tqdm

from autoformalizer.eval_utils.constants import gpt_verification_system_prompt


def parse_formalization_status(response: str) -> str:
    """
    Parses the formalization status from the model's response.

    Args:
        response (str): The response string containing formalization analysis
            and conclusion.

    Returns:
        str: The formalization status ('Correct', 'Incorrect', or 'Status not found').
        Looks for explicit correct/incorrect markers as well as boxed conclusions.
    """
    # Handle empty or None input
    if not response:
        return "Status not found"

    # Look for boxed conclusion first
    if r"\boxed" in response:
        # Extract text between \boxed{ and }
        import re

        boxed_match = re.search(r"\\boxed{([^}]+)}", response)
        if boxed_match:
            text = boxed_match.group(1)
            # Clean up any TeX formatting and extra whitespace
            text = text.replace(r"\text", "").strip("{ }").strip()
            if "incorrect" in text.lower():
                return "Incorrect"
            elif "correct" in text.lower():
                return "Correct"

    # If no boxed conclusion, look for explicit status line
    lines = response.lower().splitlines()
    formalization_str = "formalization status:"

    for line in reversed(lines):
        if formalization_str in line:
            status = line.split(formalization_str)[1].strip()
            if "incorrect" in status:
                return "Incorrect"
            elif "correct" in status:
                return "Correct"

    # Final fallback: look for any clear indication in the full text
    response_lower = response.lower()
    if "i conclude that the formalization is correct" in response_lower:
        return "Correct"
    elif "i conclude that the formalization is incorrect" in response_lower:
        return "Incorrect"

    return "Status not found"


def calculate_metrics_with_strategies(
    grouped_feedbacks: List[List[str]], human_labels: List[str]
) -> Dict:
    """
    Calculate metrics using both random sampling and majority voting strategies.

    Args:
        grouped_feedbacks: List of lists where each inner list contains multiple model responses
        human_labels: List of ground truth labels

    Returns:
        Dictionary containing metrics for both random sampling and majority voting
    """

    def get_majority_vote(responses: List[str]) -> str:
        """Get majority vote from multiple responses"""
        parsed_responses = [parse_formalization_status(resp) for resp in responses]
        # Filter out "Status not found"
        valid_responses = [r for r in parsed_responses if r != "Status not found"]
        if not valid_responses:
            return "Status not found"

        # Count occurrences
        counter = Counter(valid_responses)
        # Get the most common response and its count
        most_common = counter.most_common(1)[0]
        majority_threshold = len(valid_responses) / 2

        # Return the majority response only if it appears more than 50% of the time
        if most_common[1] > majority_threshold:
            return most_common[0]
        return "Inconclusive"

    def calculate_basic_metrics(
        feedback_bools: List[bool], human_label_bools: List[bool]
    ) -> Dict:
        """Calculate basic classification metrics"""
        matches = sum(1 for m, h in zip(feedback_bools, human_label_bools) if m == h)
        total_samples = len(human_label_bools)
        accuracy = matches / total_samples if total_samples > 0 else 0

        true_positives = sum(
            1 for m, h in zip(feedback_bools, human_label_bools) if m and h
        )
        true_negatives = sum(
            1 for m, h in zip(feedback_bools, human_label_bools) if not m and not h
        )
        false_positives = sum(
            1 for m, h in zip(feedback_bools, human_label_bools) if m and not h
        )
        false_negatives = sum(
            1 for m, h in zip(feedback_bools, human_label_bools) if not m and h
        )

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "total_samples": total_samples,
            "matches": matches,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

    # Random sampling: take first response from each group
    random_sample_responses = [
        group[0] if group else "Status not found" for group in grouped_feedbacks
    ]
    random_sample_bools = [
        parse_formalization_status(resp) == "Correct"
        for resp in random_sample_responses
    ]
    human_label_bools = [label.lower() == "true" for label in human_labels]

    # Majority voting
    majority_responses = [get_majority_vote(group) for group in grouped_feedbacks]
    majority_bools = [resp == "Correct" for resp in majority_responses]

    # Calculate metrics for both approaches
    random_sample_metrics = calculate_basic_metrics(
        random_sample_bools, human_label_bools
    )
    majority_voting_metrics = calculate_basic_metrics(majority_bools, human_label_bools)

    # Add response distribution information for majority voting
    majority_voting_metrics["response_distribution"] = {
        "conclusive": len([r for r in majority_responses if r != "Inconclusive"]),
        "inconclusive": len([r for r in majority_responses if r == "Inconclusive"]),
        "percent_conclusive": len(
            [r for r in majority_responses if r != "Inconclusive"]
        )
        / len(majority_responses),
    }

    return {
        "random_sampling": random_sample_metrics,
        "majority_voting": majority_voting_metrics,
    }


class ModelType(Enum):
    """Enum for supported model types"""

    GEMINI = auto()
    CLAUDE = auto()
    OPENAI = auto()
    O1 = auto()
    QWQ = auto()


@dataclass
class APIEndpoint:
    """Configuration for a single API endpoint"""

    model_path: str
    url: Optional[str] = None
    api_key: Optional[str] = None


class APIConfig:
    """Configuration for multiple API endpoints"""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.endpoints = [APIEndpoint(**endpoint) for endpoint in config["endpoints"]]

    @property
    def model_type(self) -> ModelType:
        """Determines the model type based on the first endpoint's model path"""
        model_path = self.endpoints[0].model_path
        if model_path.startswith("gemini"):
            return ModelType.GEMINI
        elif model_path.startswith("claude"):
            return ModelType.CLAUDE
        elif model_path.startswith("gpt"):
            return ModelType.OPENAI
        return ModelType.QWQ


class APIClientFactory:
    """Factory for creating API clients"""

    @staticmethod
    def create_client(endpoint: APIEndpoint) -> Union[OpenAI, anthropic.Anthropic]:
        """Creates an API client based on endpoint configuration"""
        if endpoint.model_path.startswith("gemini"):
            return OpenAI(
                api_key=endpoint.api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        elif endpoint.model_path.startswith("claude"):
            return anthropic.Anthropic(api_key=endpoint.api_key)
        elif endpoint.model_path.startswith("gpt"):
            return OpenAI(api_key=endpoint.api_key)
        else:
            return OpenAI(api_key="EMPTY", base_url=endpoint.url)


class ModelFeedbackService:
    """Service for handling model feedback operations with multiple endpoints"""

    def __init__(self, config_path: str):
        self.config = APIConfig(config_path)
        self.clients = [
            APIClientFactory.create_client(endpoint)
            for endpoint in self.config.endpoints
        ]
        self.client_cycle = cycle(enumerate(zip(self.clients, self.config.endpoints)))

    def get_model_response(
        self,
        client,
        endpoint: APIEndpoint,
        lean4_code: str,
        natural_language: str,
        temperature: float,
    ) -> str:

        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

        """Gets response from the model based on input"""
        prompt = (
            f"Natural Language Statement:\n{natural_language}\n\n"
            f"Lean 4 Statement:\n{lean4_code}\n"
        )
        if endpoint.model_path.startswith("claude"):
            completion = client.messages.create(
                model=endpoint.model_path,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": f"{gpt_verification_system_prompt}\n\n{prompt}",
                    }
                ],
                temperature=temperature,
            )
            return completion.content[0].text

        elif endpoint.model_path.startswith("gemini"):
            combined_prompt = f"{gpt_verification_system_prompt}\n\n{prompt}"
            messages = [{"role": "user", "content": combined_prompt}]
            completion = client.chat.completions.create(
                model=endpoint.model_path,
                messages=messages,
                temperature=temperature,
            )
            return completion.choices[0].message.content

        elif endpoint.model_path.startswith("o1"):
            combined_prompt = f"{gpt_verification_system_prompt}\n\n{prompt}"
            messages = [{"role": "user", "content": combined_prompt}]
            completion = client.chat.completions.create(
                model=endpoint.model_path,
                messages=messages,
                temperature=temperature,
            )
            return completion.choices[0].message.content

        elif "QwQ" in endpoint.model_path:

            messages = [
                {"role": "system", "content": gpt_verification_system_prompt},
                {"role": "user", "content": prompt},
            ]
            completion = client.chat.completions.create(
                model=endpoint.model_path,  # Use the model path from environment
                messages=messages,
                temperature=temperature,
                max_tokens=4096,  # qwq seems like to output much longer than others,
            )
            return completion.choices[0].message.content

        else:
            messages = [
                {"role": "system", "content": gpt_verification_system_prompt},
                {"role": "user", "content": prompt},
            ]
            completion = client.chat.completions.create(
                model=endpoint.model_path, messages=messages, temperature=temperature
            )
            return completion.choices[0].message.content

    def _get_response_with_retry(
        self,
        client,
        endpoint: APIEndpoint,
        code: str,
        nl: str,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> str:
        """Get model response with retry logic"""
        retry_count = 0
        backoff_time = 1

        while retry_count < max_retries:
            try:
                return self.get_model_response(client, endpoint, code, nl, temperature)
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise e
                time.sleep(backoff_time)
                backoff_time *= 2

    def process_parallel_feedback(
        self,
        lean4_codes: List[str],
        natural_language_ls: List[str],
        num_workers: int = 60,
        num_votes: int = 1,
        temperature: float = 0.0,
    ) -> List[dict]:
        """
        Process feedback in parallel with progress tracking and majority voting.

        Args:
            lean4_codes: List of Lean 4 codes to process
            natural_language_ls: List of natural language statements
            num_workers: Number of parallel workers
            num_votes: Number of votes per input (calls to model)

        Returns:
            List of dicts containing:
                - response: The majority response (or "Inconclusive" if no clear majority)
                - confidence: Ratio of votes for the majority response
                - all_responses: List of all responses received
        """
        total_tasks = len(lean4_codes) * num_votes
        expanded_codes = [code for code in lean4_codes for _ in range(num_votes)]
        expanded_nl = [nl for nl in natural_language_ls for _ in range(num_votes)]

        # Create a single progress bar with dynamic description
        pbar = tqdm(
            total=total_tasks,
            desc="Processing",
            ncols=100,  # Fixed width
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
            ) as executor:
                # Store futures and their metadata separately
                futures_to_metadata = {}

                # Submit all tasks
                for idx, (code, nl) in enumerate(zip(expanded_codes, expanded_nl)):
                    client_idx, (client, endpoint) = next(self.client_cycle)
                    future = executor.submit(
                        self._get_response_with_retry,
                        client,
                        endpoint,
                        code,
                        nl,
                        temperature=temperature,
                    )
                    # Store metadata mapped to future
                    futures_to_metadata[future] = (idx, client_idx)

                # Process completed futures
                grouped_responses = [[] for _ in range(len(lean4_codes))]
                success_count = 0
                error_count = 0

                for future in concurrent.futures.as_completed(
                    futures_to_metadata.keys()
                ):
                    idx, client_idx = futures_to_metadata[future]
                    original_idx = idx // num_votes
                    try:
                        response = future.result()
                        grouped_responses[original_idx].append(response)
                        success_count += 1

                        # Update progress description with success rate
                        success_rate = (
                            (success_count / (success_count + error_count)) * 100
                            if (success_count + error_count) > 0
                            else 100
                        )
                        pbar.set_description(f"Success Rate: {success_rate:.1f}%")

                    except Exception as e:
                        error_count += 1
                        grouped_responses[original_idx].append(
                            f"Error processing item {idx}: {str(e)}"
                        )

                    pbar.update(1)

        finally:
            pbar.close()

            # Print final statistics without progress bars
            total_processed = success_count + error_count
            final_success_rate = (
                (success_count / total_processed) * 100 if total_processed > 0 else 0
            )
            print("\nProcessing Complete:")
            print(f"Total Processed: {total_processed}")
            print(f"Successful: {success_count}")
            print(f"Failed: {error_count}")
            print(f"Success Rate: {final_success_rate:.1f}%")

        return grouped_responses


class DataFrameFeedbackProcessor:
    """Processor for handling DataFrame-based feedback operations"""

    def __init__(self, service: ModelFeedbackService):
        self.service = service

    def process_dataframe(
        self, input_csv: str, output_csv: str, verbose: bool = True
    ) -> pd.DataFrame:
        """Process feedback for a DataFrame"""
        df = pd.read_csv(input_csv)
        natural_language_ls = df["natural_language"].tolist()

        for column in df.columns:
            if column.startswith("autoformalization_"):
                idx = column.split("_")[-1]
                lean4_codes = df[column].tolist()

                feedbacks = self.service.process_parallel_feedback(
                    lean4_codes, natural_language_ls
                )

                df[f"model_feedback_{idx}"] = feedbacks
                df[f"model_feedback_{idx}_bool"] = [
                    parse_formalization_status(feedback) == "Correct"
                    for feedback in feedbacks
                ]

                if verbose:
                    self._print_feedback_stats(df, idx)

        df.to_csv(output_csv, index=False)
        return df

    @staticmethod
    def _print_feedback_stats(df: pd.DataFrame, idx: str):
        """Print statistics about the feedback"""
        val_counts = df[f"model_feedback_{idx}_bool"].value_counts()
        print(f"Feedback for autoformalization_{idx}:")
        print(val_counts)
        print(val_counts / val_counts.sum())


def create_feedback_service(
    config_path: str = "config/api_config.yaml",
) -> ModelFeedbackService:
    """Factory function to create a feedback service"""
    return ModelFeedbackService(config_path=config_path)


# Usage example:
@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace datasets"""

    dataset_id: str
    branch: str
    private: bool = True


class HuggingFaceFeedbackProcessor:
    """Processor for handling HuggingFace dataset feedback operations"""

    def __init__(self, service: ModelFeedbackService):
        self.service = service

    def process_dataset(
        self,
        input_config: HuggingFaceConfig,
        output_config: HuggingFaceConfig,
        filter_compiled: bool = False,
        verbose: bool = True,
    ) -> None:
        """Process feedback for a HuggingFace dataset"""
        dataset = load_dataset(
            input_config.dataset_id, split="train", revision=input_config.branch
        )

        autoformalization_columns = [
            col
            for col in dataset.column_names
            if col.startswith("autoformalization_") and "samples" not in col
        ]

        natural_language_ls = dataset["natural_language"]

        for column in autoformalization_columns:
            idx = column.split("_")[-1]
            processed_data = self._process_column(
                dataset, column, idx, natural_language_ls, filter_compiled, verbose
            )
            dataset = self._update_dataset(dataset, processed_data, idx)

        self._push_to_hub(dataset, output_config)

    def _process_column(
        self,
        dataset,
        column: str,
        idx: str,
        natural_language_ls,
        filter_compiled: bool,
        verbose: bool,
    ) -> tuple:
        """Process a single column of autoformalizations"""
        if filter_compiled:
            compiled_ids = self._get_compiled_ids(dataset, idx)
        else:
            compiled_ids = range(len(dataset))

        lean4_codes = [dataset[column][i] for i in compiled_ids]
        lean4_codes = [item.replace("```", "") for item in lean4_codes]
        filtered_natural_language_ls = [natural_language_ls[i] for i in compiled_ids]

        feedbacks = self.service.process_parallel_feedback(
            lean4_codes, filtered_natural_language_ls
        )

        feedback_bools = [
            parse_formalization_status(feedback) == "Correct" for feedback in feedbacks
        ]

        if verbose:
            self._print_feedback_stats(feedbacks, feedback_bools, filter_compiled)

        return compiled_ids, feedbacks, feedback_bools

    @staticmethod
    def _get_compiled_ids(dataset, idx: str) -> List[int]:
        """Get indices of compiled formalizations"""
        compiler_feedback_col = f"compiler_feedback_{idx}_bool"
        return [i for i, x in enumerate(dataset[compiler_feedback_col]) if x]

    @staticmethod
    def _print_feedback_stats(
        feedbacks: List[str], feedback_bools: List[bool], filter_compiled: bool
    ) -> None:
        """Print feedback statistics"""

        true_count = sum(feedback_bools)
        if filter_compiled:
            print(f"Filtered for {len(feedbacks)} autoformalizations that compile.")
        success_rate = true_count / len(feedback_bools) if feedback_bools else 0
        print(f"Success rate: {success_rate:.3%}")

    def _update_dataset(self, dataset, processed_data: tuple, idx: str):
        """Update dataset with processed feedback"""
        compiled_ids, feedbacks, feedback_bools = processed_data

        full_feedbacks = [""] * len(dataset)
        full_feedback_bools = [None] * len(dataset)

        for j, comp_id in enumerate(compiled_ids):
            full_feedbacks[comp_id] = feedbacks[j]
            full_feedback_bools[comp_id] = feedback_bools[j]

        model_name = self.service.config.model_name
        feedback_col = f"{model_name}_feedback_{idx}"
        feedback_bool_col = f"{model_name}_feedback_{idx}_bool"

        # Remove existing columns if they exist
        for col in [feedback_col, feedback_bool_col]:
            if col in dataset.column_names:
                dataset = dataset.remove_columns(col)

        # Add new columns
        dataset = dataset.add_column(feedback_col, full_feedbacks)
        dataset = dataset.add_column(feedback_bool_col, full_feedback_bools)

        return dataset

    def _push_to_hub(self, dataset, output_config: HuggingFaceConfig) -> None:
        """Push processed dataset to HuggingFace Hub"""
        model_name = self.service.config.model_name
        dataset.push_to_hub(
            output_config.dataset_id,
            revision=output_config.branch,
            private=output_config.private,
            commit_message=f"Feedback for autoformalizations using {model_name} model.",
        )


class ModelJudgeService:
    """Service for evaluating model feedback against human labels"""

    def __init__(self, service: ModelFeedbackService, model_name: str):
        self.service = service
        self.model_name = model_name

    def evaluate_dataset(
        self,
        input_config: HuggingFaceConfig,
        output_config: HuggingFaceConfig,
        verbose: bool = True,
        temperature: float = 0.0,
        num_votes: int = 1,
    ) -> None:
        """Evaluate model feedback against human labels"""
        dataset = load_dataset(
            input_config.dataset_id, split="train", revision=input_config.branch
        )
        filtered_dataset = dataset.filter(lambda x: x["alignment_label"] != "unknown")

        processed_data = self._process_feedback(
            filtered_dataset, temperature, num_votes
        )
        results = self._calculate_metrics(processed_data)

        if verbose:
            self._print_evaluation_results(results)

        self._update_and_push_dataset(
            filtered_dataset,
            processed_data["feedbacks"],
            results,
            output_config,
        )

    def evaluate_dataset_filtered_lean(
        self,
        input_config: HuggingFaceConfig,
        output_config: HuggingFaceConfig,
        filter_compiled: bool = True,
        verbose: bool = True,
        temperature: float = 0.0,
        num_votes: int = 1,
    ) -> None:
        """Evaluate model feedback against human labels"""

        assert filter_compiled

        dataset = load_dataset(
            input_config.dataset_id, split="train", revision=input_config.branch
        )

        filtered_dataset = dataset.filter(lambda x: x["compiler_feedback_1_bool"])

        processed_data = self._process_feedback_autoformalization(
            filtered_dataset, temperature, num_votes
        )
        results = self._calculate_metrics(processed_data)

        if verbose:
            self._print_evaluation_results(results)

        self._update_and_push_dataset(
            filtered_dataset,
            processed_data["feedbacks"],
            results,
            output_config,
        )

    def _process_feedback(self, dataset, temperature, num_votes) -> dict:
        """Process feedback for evaluation"""
        return {
            "natural_language": dataset["problem"],
            "autoformalizations": dataset["autoformalization_preview"],
            "human_labels": dataset["alignment_label"],
            "feedbacks": self.service.process_parallel_feedback(
                dataset["autoformalization_preview"],
                dataset["problem"],
                temperature=temperature,
                num_votes=num_votes,
            ),
        }

    def _process_feedback_autoformalization(
        self, dataset, temperature, num_votes
    ) -> dict:
        """Process feedback for evaluation"""
        return {
            "natural_language": dataset["natural_language"],
            "autoformalizations": dataset["autoformalization_1"],
            "human_labels": dataset["alignment_label"],
            "feedbacks": self.service.process_parallel_feedback(
                dataset["autoformalization_1"],
                dataset["natural_language"],
                temperature=temperature,
                num_votes=num_votes,
            ),
        }

    def _calculate_metrics(self, data: dict) -> dict:
        """Calculate evaluation metrics"""
        return calculate_metrics_with_strategies(
            grouped_feedbacks=data["feedbacks"], human_labels=data["human_labels"]
        )

    @staticmethod
    def _print_evaluation_results(results: dict) -> None:
        """Print evaluation results for both random sampling and majority voting"""

        def print_metrics(metrics: dict, strategy_name: str):
            print(f"\n{strategy_name} Results:")
            print(f"Total samples analyzed: {metrics['total_samples']}")
            print(f"Number of matching predictions: {metrics['matches']}")
            print(f"Accuracy: {metrics['accuracy']:.2%}")
            print(f"Precision: {metrics['precision']:.2%}")
            print(f"Recall : {metrics['recall']:.2%}")
            print(f"F1 Score: {metrics['f1_score']:.2%}")

            print("\nDetailed breakdown:")
            print(f"True Positives: {metrics['true_positives']}")
            print(f"True Negatives: {metrics['true_negatives']}")
            print(f"False Positives: {metrics['false_positives']}")
            print(f"False Negatives: {metrics['false_negatives']}")

            if "response_distribution" in metrics:
                print("\nResponse Distribution:")
                dist = metrics["response_distribution"]
                print(
                    f"Conclusive responses: {dist['conclusive']} ({dist['percent_conclusive']:.2%})"
                )
                print(f"Inconclusive responses: {dist['inconclusive']}")

        print_metrics(results["random_sampling"], "Random Sampling")
        print_metrics(results["majority_voting"], "Majority Voting")

    def _update_and_push_dataset(
        self,
        dataset,
        feedbacks: List[str],
        results: dict,  # Complete results dictionary
        output_config: HuggingFaceConfig,
    ) -> None:
        """Update dataset with results and push to hub"""
        model_name = self.model_name
        feedback_column_name = f"model_{model_name}_feedback"

        if feedback_column_name in dataset.column_names:
            dataset = dataset.remove_columns(feedback_column_name)

        dataset = dataset.add_column(feedback_column_name, feedbacks)

        # Create detailed commit message
        commit_message = (
            f"Added model feedback for {model_name}\n"
            f"Random Sampling Accuracy: {results['random_sampling']['accuracy']:.2%}\n"
            f"Majority Voting Accuracy: {results['majority_voting']['accuracy']:.2%}\n"
            f"Conclusive Responses: {results['majority_voting']['response_distribution']['percent_conclusive']:.2%}"
        )

        dataset.push_to_hub(
            output_config.dataset_id,
            revision=output_config.branch,
            private=output_config.private,
            commit_message=commit_message,
        )


# Example usage functions
def process_feedback(
    input_csv: str, output_csv: str, model_name: str, verbose: bool = True
) -> pd.DataFrame:
    """Process feedback for a given CSV file"""
    service = create_feedback_service(model_name)
    processor = DataFrameFeedbackProcessor(service)
    return processor.process_dataframe(input_csv, output_csv, verbose)


def hf_model_feedback(
    input_dataset_id: str,
    input_dataset_branch: str,
    output_dataset_id: str,
    output_dataset_branch: str,
    model_name: str,
    filter_compiled: bool = False,
    verbose: bool = True,
) -> None:
    """Processes model feedback for autoformalizations in a HuggingFace dataset.

    Args:
        input_dataset_id (str): HuggingFace dataset ID with autoformalizations.
            Format: 'organization/dataset-name'
        input_dataset_branch (str): Branch of input dataset.
        output_dataset_id (str): HuggingFace dataset ID for processed results.
            Format: 'organization/dataset-name'
        output_dataset_branch (str): Branch name for output.
        model_name (str): Model to use for feedback (e.g., 'claude-3').
        filter_compiled (bool, optional): Only process formalizations that compile.
            Defaults to False.
        verbose (bool, optional): Print processing statistics.
            Defaults to True.

    Processing steps:
        1. Loads dataset and identifies autoformalization columns
        2. Optionally filters for successfully compiled formalizations
        3. Gets model feedback for each formalization
        4. Updates dataset with feedback results
        5. Pushes to HuggingFace Hub

    Dataset requirements:
        Input dataset must contain:
        - 'natural_language': Original statements
        - 'autoformalization_i': The i-th formalization attempt
        - If filter_compiled=True: 'compiler_feedback_i_bool'

    Output columns added:
        - '{model_name}_feedback_i': Model's feedback
        - '{model_name}_feedback_i_bool': Whether formalization is correct

    Examples:
        >>> hf_model_feedback(
        ...     input_dataset_id="AI-MO/formalizations",
        ...     input_dataset_branch="main",
        ...     output_dataset_id="AI-MO/feedback_results",
        ...     output_dataset_branch="model_feedback",
        ...     model_name="claude-3",
        ...     filter_compiled=True
        ... )
    """
    service = create_feedback_service(model_name)

    processor = HuggingFaceFeedbackProcessor(service)

    input_config = HuggingFaceConfig(input_dataset_id, input_dataset_branch)
    output_config = HuggingFaceConfig(output_dataset_id, output_dataset_branch)

    processor.process_dataset(input_config, output_config, filter_compiled, verbose)


def evaluate_judge_model(
    input_dataset_id: str,
    input_dataset_branch: str,
    output_dataset_id: str,
    output_dataset_branch: str,
    config_path: str,
    model_name: str,
    verbose: bool = True,
    temperature: float = 0.0,
    num_votes: int = 1,
) -> None:
    """Evaluates model's feedback accuracy against human-labeled dataset.

    Args:
        input_dataset_id (str): HuggingFace dataset ID containing human labels.
            Format: 'organization/dataset-name'
        input_dataset_branch (str): Branch of input dataset to use.
        output_dataset_id (str): HuggingFace dataset ID for results.
            Format: 'organization/dataset-name'
        output_dataset_branch (str): Branch name for output dataset.
        config_path (str): config path of a yaml file indicate model name and port and baseurl if necessary.
        verbose (bool, optional): Whether to print detailed statistics.
            Defaults to True.

    Processing steps:
        1. Loads dataset and filters out 'unknown' labels
        2. Generates model feedback for each formalization
        3. Compares predictions with human labels
        4. Calculates accuracy metrics
        5. Pushes results to HuggingFace Hub

    Dataset requirements:
        Input dataset must contain:
        - 'problem': Natural language statements
        - 'autoformalization_preview': Formalizations
        - 'alignment_label': Human labels ('true'/'false'/'unknown')

    Statistics (when verbose=True):
        - Total samples analyzed
        - Matching predictions count
        - Overall accuracy
        - Confusion matrix metrics

    Examples:
        >>> evaluate_judge_model(
        ...     input_dataset_id="AI-MO/formalized_preview",
        ...     input_dataset_branch="main",
        ...     output_dataset_id="AI-MO/evaluation_results",
        ...     output_dataset_branch="model_eval",
        ... )
    """
    service = create_feedback_service(config_path)

    judge = ModelJudgeService(service, model_name)

    input_config = HuggingFaceConfig(input_dataset_id, input_dataset_branch)
    output_config = HuggingFaceConfig(output_dataset_id, output_dataset_branch)

    judge.evaluate_dataset(input_config, output_config, verbose, temperature, num_votes)


def evaluate_autoformalization_model(
    input_dataset_id: str,
    input_dataset_branch: str,
    output_dataset_id: str,
    output_dataset_branch: str,
    config_path: str,
    model_name: str,
    filter_compiled: bool = True,
    verbose: bool = True,
    temperature: float = 0.0,
    num_votes: int = 1,
) -> None:
    """Evaluates model's feedback accuracy against human-labeled dataset.

    Args:
        input_dataset_id (str): HuggingFace dataset ID containing human labels.
            Format: 'organization/dataset-name'
        input_dataset_branch (str): Branch of input dataset to use.
        output_dataset_id (str): HuggingFace dataset ID for results.
            Format: 'organization/dataset-name'
        output_dataset_branch (str): Branch name for output dataset.
        config_path (str): config path of a yaml file indicate model name and port and baseurl if necessary.
        verbose (bool, optional): Whether to print detailed statistics.
            Defaults to True.

    Processing steps:
        1. Loads dataset and filters out 'unknown' labels
        2. Generates model feedback for each formalization
        3. Compares predictions with human labels
        4. Calculates accuracy metrics
        5. Pushes results to HuggingFace Hub

    Dataset requirements:
        Input dataset must contain:
        - 'problem': Natural language statements
        - 'autoformalization_preview': Formalizations
        - 'alignment_label': Human labels ('true'/'false'/'unknown')

    Statistics (when verbose=True):
        - Total samples analyzed
        - Matching predictions count
        - Overall accuracy
        - Confusion matrix metrics

    Examples:
        >>> evaluate_judge_model(
        ...     input_dataset_id="AI-MO/formalized_preview",
        ...     input_dataset_branch="main",
        ...     output_dataset_id="AI-MO/evaluation_results",
        ...     output_dataset_branch="model_eval",
        ... )
    """

    service = create_feedback_service(config_path)

    judge = ModelJudgeService(service, model_name)

    input_config = HuggingFaceConfig(input_dataset_id, input_dataset_branch)
    output_config = HuggingFaceConfig(output_dataset_id, output_dataset_branch)

    judge.evaluate_dataset_filtered_lean(
        input_config, output_config, filter_compiled, verbose, temperature, num_votes
    )


def get_judge_feedback(
    input_dataset_id: str,
    input_dataset_branch: str,
    output_dataset_id: str,
    output_dataset_branch: str,
    model_name: str,
    nl_key: str,
    fl_key: str,
    sample_size: Optional[int] = None,
    num_votes: int = 1,
    temperature: float = 0.0,
    verbose: bool = True,
    seed: int = 42,
) -> None:
    """Gets judge model feedback and filters dataset based on model responses.

    Args:
        input_dataset_id (str): HuggingFace dataset ID with formalizations
        input_dataset_branch (str): Branch of input dataset
        output_dataset_id (str): HuggingFace dataset ID for results
        output_dataset_branch (str): Branch for output
        config_path (str): Path to YAML config file for judge model
        num_votes (int): Number of votes per formalization
        temperature (float): Sampling temperature
        verbose (bool): Print progress and statistics
        seed (int): Random seed for sampling
    """
    # Create service and load dataset
    # service = create_feedback_service(config_path)
    from autoformalizer.eval_utils.model_feedback_custom import (
        ModelFeedbackService_noyaml,
    )

    service = ModelFeedbackService_noyaml()

    # Load and optionally sample dataset

    full_dataset = load_dataset(
        input_dataset_id, split="train", revision=input_dataset_branch
    )

    if sample_size and sample_size > 0:
        if sample_size > len(full_dataset):
            logging.info(
                f"Warning: Sample size {sample_size} larger than dataset size {len(full_dataset)}."
            )
            sample_size = len(full_dataset)

        # Sample dataset
        indices = list(range(len(full_dataset)))
        random.seed(seed)
        sampled_indices = random.sample(indices, sample_size)
        dataset = full_dataset.select(sampled_indices)

        if verbose:
            logging.info(
                f"Sampled {sample_size} examples from {len(full_dataset)} total examples"
            )
    else:
        dataset = full_dataset
        if verbose:
            logging.info(f"Using full dataset with {len(dataset)} examples")

    if verbose:
        logging.info(f"Processing examples with {num_votes} votes each")

    # Get judge model feedback
    feedbacks = service.process_parallel_feedback(
        dataset[fl_key],
        dataset[nl_key],
        num_votes=num_votes,
        temperature=temperature,
    )

    # Process votes and get statuses
    def process_feedback_group(group):
        parsed_responses = [parse_formalization_status(resp) for resp in group]
        valid_responses = [r for r in parsed_responses if r != "Status not found"]
        if not valid_responses:
            return "Status not found"

        # Get most common response
        counter = Counter(valid_responses)
        most_common = counter.most_common(1)[0]
        return most_common[0]

    # Process all feedbacks
    statuses = [process_feedback_group(group) for group in feedbacks]

    if verbose:
        # Print feedback statistics before filtering
        status_counts = Counter(statuses)
        logging.info("\nInitial feedback distribution:")
        for status, count in status_counts.most_common():
            logging.info(f"{status}: {count} ({count / len(dataset):.1%})")

    # Add feedback columns to full dataset first
    feedback_col = f"{model_name}_feedback"
    status_col = f"{model_name}_status"

    dataset = dataset.add_column(feedback_col, feedbacks)
    dataset = dataset.add_column(status_col, statuses)

    # Filter dataset based on 'Correct' status
    filtered_dataset = dataset.filter(lambda x: x[status_col] == "Correct")

    if verbose:
        logging.info("\nAfter filtering:")
        logging.info(
            f"Kept {len(filtered_dataset)} out of {len(dataset)} examples ({len(filtered_dataset) / len(dataset):.1%})"
        )

    # Set output branch name
    output_dataset_branch = f"filter_with_model_{model_name}"

    # Push to hub
    filtered_dataset.push_to_hub(
        output_dataset_id,
        revision=output_dataset_branch,
        private=True,
        commit_message=(
            f"Added {model_name} judge feedback and filtered for correct examples\n"
            f"Kept {len(filtered_dataset)}/{len(dataset)} examples\n"
            f"Using {num_votes} votes per example"
            + (f"\nSampled {sample_size} examples" if sample_size else "")
        ),
    )

    if verbose:
        logging.info(
            f"\nUploaded filtered dataset with {len(filtered_dataset)} examples to {output_dataset_id}"
        )
        logging.info(f"Branch: {output_dataset_branch}")
        logging.info(f"Added columns: {feedback_col}, {status_col}")


if __name__ == "__main__":

    # Create parser
    parser = argparse.ArgumentParser(description="Model Feedback and Evaluation Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate model performance"
    )
    evaluate_parser.add_argument(
        "--input_dataset_id", required=True, help="Input HuggingFace dataset ID"
    )
    evaluate_parser.add_argument(
        "--input_dataset_branch", required=True, help="Input dataset branch"
    )
    evaluate_parser.add_argument(
        "--output_dataset_id", required=True, help="Output HuggingFace dataset ID"
    )
    evaluate_parser.add_argument(
        "--output_dataset_branch", required=True, help="Output dataset branch"
    )
    evaluate_parser.add_argument(
        "--model_name", required=True, help="Model name to use"
    )
    evaluate_parser.add_argument(
        "--verbose", action="store_true", help="Print detailed statistics"
    )

    # Process command
    process_parser = subparsers.add_parser(
        "process", help="Process dataset with feedback"
    )
    process_parser.add_argument(
        "--input_dataset_id", required=True, help="Input HuggingFace dataset ID"
    )
    process_parser.add_argument(
        "--input_dataset_branch", required=True, help="Input dataset branch"
    )
    process_parser.add_argument(
        "--output_dataset_id", required=True, help="Output HuggingFace dataset ID"
    )
    process_parser.add_argument(
        "--output_dataset_branch", required=True, help="Output dataset branch"
    )
    process_parser.add_argument("--model_name", required=True, help="Model name to use")
    process_parser.add_argument(
        "--filter_compiled",
        action="store_true",
        help="Only process compiled formalizations",
    )
    process_parser.add_argument(
        "--verbose", action="store_true", help="Print detailed statistics"
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command == "evaluate":
        evaluate_judge_model(
            input_dataset_id=args.input_dataset_id,
            input_dataset_branch=args.input_dataset_branch,
            output_dataset_id=args.output_dataset_id,
            output_dataset_branch=args.output_dataset_branch,
            model_name=args.model_name,
            verbose=args.verbose,
        )
    elif args.command == "feedback":
        hf_model_feedback(
            input_dataset_id=args.input_dataset_id,
            input_dataset_branch=args.input_dataset_branch,
            output_dataset_id=args.output_dataset_id,
            output_dataset_branch=args.output_dataset_branch,
            model_name=args.model_name,
            filter_compiled=args.filter_compiled,
            verbose=args.verbose,
        )
    else:
        parser.print_help()
