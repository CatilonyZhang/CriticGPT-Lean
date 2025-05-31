import concurrent.futures
import logging
import os
import random
import time
from collections import Counter
from itertools import cycle
from typing import Callable, Dict, List, Optional

from datasets import load_dataset
from openai import OpenAI
from tqdm.auto import tqdm

from autoformalizer.eval_utils.constants import gpt_verification_system_prompt
from autoformalizer.eval_utils.model_feedback import (
    APIEndpoint,
    ModelFeedbackService,
    parse_formalization_status,
)


class APIConfig:
    """Configuration for multiple API endpoints using environment variables"""

    def __init__(self):
        try:
            self.base_port = int(os.environ.get("BASE_SERVICE_PORT", 5001))
            self.num_workers = int(os.environ.get("NUM_VLLM_WORKERS", 1))
            self.model = os.environ.get("MODEL_PATH", "")
            self.endpoints = self._create_endpoints()
        except ValueError as e:
            raise ValueError(
                "Invalid environment variables: BASE_SERVICE_PORT and NUM_VLLM_WORKERS must be integers"
            ) from e

    def _create_endpoints(self) -> List[APIEndpoint]:
        """Creates endpoints for each VLLM worker with incrementing ports"""
        endpoints = []
        for worker_id in range(self.num_workers):
            port = self.base_port + worker_id
            endpoints.append(
                APIEndpoint(
                    model_path=self.model,  # Default model type
                    url=f"http://localhost:{port}/v1",
                    api_key="EMPTY",
                )
            )
        return endpoints


class APIClientFactory:
    """Factory for creating API clients"""

    @staticmethod
    def create_client(endpoint: APIEndpoint) -> OpenAI:
        """Creates an OpenAI client with the specified base URL"""
        return OpenAI(api_key="EMPTY", base_url=endpoint.url)


class ModelFeedbackService_noyaml(ModelFeedbackService):
    """Service for handling model feedback operations with multiple endpoints using environment variables"""

    def __init__(self):
        # Skip parent class initialization since we don't want YAML config
        # Instead of super().__init__(config_path)
        self.config = APIConfig()  # This uses the env var version of APIConfig
        self.clients = [
            APIClientFactory.create_client(endpoint)
            for endpoint in self.config.endpoints
        ]
        self.client_cycle = cycle(enumerate(zip(self.clients, self.config.endpoints)))

    def get_model_response(
        self,
        client,
        endpoint: APIEndpoint,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Gets response from the model based on prompt and optional system prompt

        Args:
            client: API client instance
            endpoint: APIEndpoint configuration
            prompt: User prompt/query
            system_prompt: Optional system prompt for model context/instruction
            temperature: Sampling temperature

        Returns:
            Model's response text
        """
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

        if endpoint.model_path.startswith("claude"):
            messages = []
            if system_prompt:
                # Claude handles system prompts as user messages with special prefix
                messages.append(
                    {"role": "user", "content": f"{system_prompt}\n\n{prompt}"}
                )
            else:
                messages.append({"role": "user", "content": prompt})

            completion = client.messages.create(
                model=endpoint.model_path,
                max_tokens=1024,
                messages=messages,
                temperature=temperature,
            )
            return completion.content[0].text

        elif endpoint.model_path.startswith("gemini"):
            # Gemini handles system prompts similarly to Claude
            content = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            messages = [{"role": "user", "content": content}]
            completion = client.chat.completions.create(
                model=endpoint.model_path,
                messages=messages,
                temperature=temperature,
            )
            return completion.choices[0].message.content

        elif endpoint.model_path.startswith("o1"):
            # O1 handles system prompts similarly to Claude
            content = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            messages = [{"role": "user", "content": content}]
            completion = client.chat.completions.create(
                model=endpoint.model_path,
                messages=messages,
                temperature=temperature,
            )
            return completion.choices[0].message.content

        elif "QwQ" in endpoint.model_path:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            completion = client.chat.completions.create(
                model=endpoint.model_path,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,  # QwQ tends to output longer responses
            )
            return completion.choices[0].message.content

        else:
            # Default OpenAI-style API handling
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            completion = client.chat.completions.create(
                model=endpoint.model_path, messages=messages, temperature=temperature
            )
            return completion.choices[0].message.content

    def _get_response_with_retry(
        self,
        client,
        endpoint: APIEndpoint,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> str:
        """Get model response with retry logic"""
        retry_count = 0
        backoff_time = 1

        while retry_count < max_retries:
            try:
                return self.get_model_response(
                    client, endpoint, prompt, system_prompt, temperature
                )
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise e
                time.sleep(backoff_time)
                backoff_time *= 2

    def process_parallel_feedback(
        self,
        prompts: List[str],
        system_prompt: str,
        num_workers: int = 60,
        num_votes: int = 1,
        temperature: float = 0.0,
    ) -> List[List[str]]:
        """
        Process feedback in parallel with progress tracking and majority voting.

        Args:
            prompts: List of prompts to process
            num_workers: Number of parallel workers
            num_votes: Number of votes per input (calls to model)
            temperature: Sampling temperature

        Returns:
            List of lists containing responses for each prompt
        """
        total_tasks = len(prompts) * num_votes
        expanded_prompts = [prompt for prompt in prompts for _ in range(num_votes)]

        pbar = tqdm(
            total=total_tasks,
            desc="Processing",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
            ) as executor:
                futures_to_metadata = {}

                for idx, prompt in enumerate(expanded_prompts):
                    client_idx, (client, endpoint) = next(self.client_cycle)
                    future = executor.submit(
                        self._get_response_with_retry,
                        client,
                        endpoint,
                        prompt,
                        system_prompt,
                        temperature=temperature,
                    )
                    futures_to_metadata[future] = (idx, client_idx)

                grouped_responses = [[] for _ in range(len(prompts))]
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


# Example usage:
def default_verification_prompt_func(row: Dict) -> str:
    """Default prompt template for verification"""
    return (
        f"Natural Language Statement:\n{row['problem']}\n\n"
        f"Lean 4 Statement:\n{row['autoformalization_preview']}\n"
    )


def get_judge_feedback(
    input_dataset_id: str,
    input_dataset_branch: str,
    output_dataset_id: str,
    output_dataset_branch: str = "",
    system_prompt: str = gpt_verification_system_prompt,
    prompt_template: Callable[
        [Dict], str
    ] = default_verification_prompt_func,  # Function that takes a row dict and returns a prompt
    columns_to_use: List[str] = [
        "problem",
        "autoformalization_preview",
    ],  # List of column names needed for prompt template
    parse_response: Callable[
        [str], str
    ] = parse_formalization_status,  # Function to parse model responses into status
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
        prompt_template (Callable): Function that takes a row dict and returns a prompt string
        columns_to_use (List[str]): List of column names needed for prompt template
        sample_size (int, optional): Number of examples to sample
        num_votes (int): Number of votes per example
        temperature (float): Sampling temperature
        verbose (bool): Print progress and statistics
        seed (int): Random seed for sampling
    """
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

    # Generate prompts using template with map function
    prompts = list(
        map(
            prompt_template,
            [
                dict(zip(columns_to_use, row))
                for row in zip(*[dataset[col] for col in columns_to_use])
            ],
        )
    )

    # Get judge model feedback
    feedbacks = service.process_parallel_feedback(
        prompts,
        system_prompt=system_prompt,
        num_votes=num_votes,
        temperature=temperature,
    )

    # Process votes and get statuses
    def get_majority_status(
        group: List[str],
        parse_func: Callable[[str], str],
        invalid_marker: str = "Status not found",
    ) -> str:
        """
        Get majority status from a group of responses using custom parse function.

        Args:
            group: List of model responses
            parse_func: Function to parse each response into a status
            invalid_marker: String indicating invalid/not found status

        Returns:
            Most common valid status or invalid_marker if no valid statuses
        """
        parsed_responses = list(map(parse_func, group))
        valid_responses = [r for r in parsed_responses if r != invalid_marker]

        if not valid_responses:
            return invalid_marker

        return Counter(valid_responses).most_common(1)[0][0]

    # Use the provided parse_formalization_status as default
    statuses = [get_majority_status(group, parse_response) for group in feedbacks]

    if verbose:
        status_counts = Counter(statuses)
        logging.info("\nInitial feedback distribution:")
        for status, count in status_counts.most_common():
            logging.info(f"{status}: {count} ({count / len(dataset):.1%})")

    model_name = os.path.splitext(os.path.basename(service.config.model))[0]

    # Add feedback columns
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

    # Push to hub
    if not len(output_dataset_branch):
        output_dataset_branch = f"filter_with_model_{model_name}"

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


# Usage example
if __name__ == "__main__":
    get_judge_feedback(
        input_dataset_id="example/dataset",
        input_dataset_branch="main",
        output_dataset_id="example/results",
        output_dataset_branch="feedback",
        system_prompt=gpt_verification_system_prompt,
        prompt_template=default_verification_prompt_func,
        columns_to_use=["natural_language", "formalization"],
        num_votes=1,
        temperature=0.0,
        verbose=True,
    )
