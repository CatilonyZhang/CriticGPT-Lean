import argparse
import json
import logging
import os
import time
from multiprocessing import Event, Process, Queue, Value
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Tuple

import yaml
from datasets import Dataset, concatenate_datasets, load_dataset
from vllm import LLM, SamplingParams

from autoformalizer.clients import lean4_client
from autoformalizer.eval_utils import get_datetime
from autoformalizer.eval_utils.lean_feedback import has_error, parallel_lean4_feedback
from evaluation.evaluation_constants import (
    LEAN4_DEFAULT_HEADER,
    GenerationResult,
    GenerationTask,
    ProofVerificationResult,
    ProofVerificationTask,
    VLLMConfig,
)


def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {str(e)}")

    return config


class ProverEvaluator:
    """
    Overview: Evaluating any prover model.
    """

    def __init__(self, config_path: str, output_dir: str, **kwargs):
        """
        Initialize the evaluator.

        Args:
            config_path: Path to the configuration YAML file.
            output_dir: Path to the output directory to store the results.
            **kwargs: Keyword arguments to overwrite configuration values.
        """
        self.config = load_config(config_path)

        # Overwrite config values with kwargs
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value  # Overwrite top-level keys directly
            else:
                # Handle nested configurations like `sampling_config`
                subkeys = key.split(".")
                curr = self.config

                for subkey in subkeys[:-1]:
                    if subkey not in curr:
                        raise KeyError(
                            f"Key '{subkey}' does not exist in the configuration."
                        )
                    curr = curr[subkey]  # Navigate to the next level

                if subkeys[-1] not in curr:
                    raise KeyError(
                        f"Key '{subkeys[-1]}' does not exist in the configuration section '{'.'.join(subkeys[:-1])}'."
                    )

                curr[subkeys[-1]] = value  # Set the final value

        if "vllm" not in self.config:
            self.vllm_config = VLLMConfig()
        else:
            self.vllm_config = VLLMConfig(**self.config["vllm"])

        sampling_config = self.config["sampling_config"]

        self.sampling_params = SamplingParams(**sampling_config)

        log_dir = f"{Path(self.config['model_path']).name}_{get_datetime()}"
        self.log_dir = Path(output_dir) / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Write the updated config to the output directory
        with open(os.path.join(self.log_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # Load and prepare the dataset
        self.dataset = load_dataset(
            self.config["data_path"], split=self.config["data_split"]
        )
        self.dataset = concatenate_datasets(self.dataset)

        self.has_header = self.config["has_header"]
        self.formal_columns = self.config["formal_columns"]

        statement_col = "statement_to_prove"

        def create_statements(example):
            if not self.has_header:
                statement = LEAN4_DEFAULT_HEADER
            else:
                statement = ""
            for column in self.formal_columns:
                statement += example[column]

            example[statement_col] = statement
            return example

        self.dataset = self.dataset.map(create_statements)
        formal_statements = self.dataset[statement_col]
        self.ids = self.dataset[self.config["index_column"]]
        self.ids = [str(idx) for idx in self.ids]

        if self.config["has_sorry"]:
            # remove the sorry from the end
            formal_statements = [s.split("sorry")[0] + "\n" for s in formal_statements]

        self.formal_statements = formal_statements
        self.problems = [{"formal_statement": s} for s in self.formal_statements]

        self.results_dict = dict()
        for idx, formal_statement in zip(self.ids, self.formal_statements):
            self.results_dict[idx] = {
                "formal_statement": formal_statement,
                "is_valid": False,
                "proof_attempts": [],
            }

        self.lean_workers = self.config["lean_workers"]
        self.lean_timeout = self.config["lean_timeout"]
        self.lean_memory_limit = self.config.get("lean_memory_limit", 512)
        self.lean_retries = self.config.get("lean_retries", 1)
        self.lean_feedback = self.config.get("lean_feedback", "server")
        if self.lean_feedback not in ["local", "server"]:
            raise ValueError(f"Invalid lean_feedback option: {self.lean_feedback}")

        if self.lean_feedback == "server":

            self.client = lean4_client.Lean4Client(
                "https://kimina.saas.moonshot.cn/lean4-evaluator",
                api_key=os.environ.get("MOONSHOT_LEAN4_API_KEY"),
            )

        self.prompt_template = self.config["prompt_template"]

        self.cuda_visible_devices = self.config.get("CUDA_VISIBLE_DEVICES", "")
        if not len(self.cuda_visible_devices):
            raise ValueError("Evaluator needs at least 1 GPU in CUDA_VISIBLE_DEVICES.")

        self.cuda_visible_devices = self.cuda_visible_devices.split(",")
        self.n_generation_processes = len(self.cuda_visible_devices)
        self.n_verification_processes = self.config["n_verification_processes"]
        self.batch_size = self.config["batch_size"]

        # Save paths
        self.hf_results_path = self.config["data_path"] + "-results"
        n_samples = self.sampling_params.n
        temp = self.sampling_params.temperature
        top_p = self.sampling_params.top_p
        max_tokens = self.sampling_params.max_tokens
        identifier = f"{n_samples}_{temp}_{top_p}_{max_tokens}"
        model_name_split = self.config["model_path"].split("/")

        # handle HF model case
        if len(model_name_split) > 2 and "checkpoint" in model_name_split[-1]:
            model_name = model_name_split[-2] + "/" + model_name_split[-1]
        else:
            model_name = model_name_split[-1]

        data_split_str = "".join(self.config["data_split"])
        self.hf_results_full_revision = (
            f"{model_name}_{identifier}_{data_split_str}_full"
        )
        self.hf_results_summary_revision = (
            f"{model_name}_{identifier}_{data_split_str}_summary"
        )

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(f"{self.config['model_path']}_evaluator_logger")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(self.log_dir, "log.txt"))
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.WARNING)
        logger.addHandler(stream_handler)
        self.logger = logger
        self.logger.info(f"Loaded configuration from {config_path}")

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate a dataset of problems using batched VLLM inference and parallel verification.
        """

        # Multiprocessing structures
        generation_queue = Queue()
        generation_output_queue = Queue()
        proof_verification_queue = Queue()
        verification_output_queue = Queue()
        generation_done = Event()
        verification_done = Event()
        verification_tasks_done = Value("i", 0)

        # Start coordinator process
        task_scheduler_process = Process(
            target=self._task_scheduler,
            args=(
                generation_queue,
                generation_output_queue,
                proof_verification_queue,
                generation_done,
                verification_done,
                verification_tasks_done,
            ),
        )
        task_scheduler_process.start()

        # Start generation processes
        generation_processes = [
            Process(
                target=self._generator_worker,
                args=(
                    generation_queue,
                    generation_output_queue,
                    generation_done,
                    self.cuda_visible_devices[i],
                ),
            )
            for i in range(self.n_generation_processes)
        ]
        for process in generation_processes:
            process.start()

        # Start verification processes
        verification_processes = [
            Process(
                target=self._verifier_worker,
                args=(
                    proof_verification_queue,
                    verification_output_queue,
                    verification_done,
                    verification_tasks_done,
                    i,
                ),
            )
            for i in range(self.n_verification_processes)
        ]
        for process in verification_processes:
            process.start()

        # Fetch results from the verification output queue
        verification_result_count = 0
        while not verification_output_queue.empty() or not verification_done.is_set():
            try:
                verification_results = verification_output_queue.get(timeout=1)
                verification_result_count += 1
                self.logger.info(
                    f"Received verification results: {verification_result_count}"
                )
                for result in verification_results:
                    problem_id = result.problem_id
                    # Append the verification result to proof_attempts
                    self.results_dict[problem_id]["proof_attempts"].append(result)
                    # Update is_valid status
                    curr_is_valid = self.results_dict[problem_id]["is_valid"]
                    self.results_dict[problem_id]["is_valid"] = (
                        curr_is_valid or result.is_valid
                    )
            except Empty:
                time.sleep(1)
                continue

        # Wait for all processes to finish
        self.logger.info("Waiting for task scheduler to finish...")
        task_scheduler_process.join()

        self.logger.info("Waiting for all generation processes to finish...")
        for process in generation_processes:
            process.join()

        self.logger.info("Waiting for all verification processes to finish...")
        for process in verification_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)

        # Confirm all processes completed
        self.logger.info("All processes completed. Processing final results...")

        results_hf_list = []
        n_success = 0
        passatn = [0] * self.sampling_params.n

        for problem_id, problem_result in self.results_dict.items():
            correct_proofs = []
            all_proofs = []
            lean_feedbacks = []
            n_proofs = len(problem_result["proof_attempts"])
            if problem_result["is_valid"]:
                n_success += 1

            passed = False
            passedat = -1
            for proof_attempt in problem_result["proof_attempts"]:
                all_proofs.append(proof_attempt.proof)
                lean_feedbacks.append(proof_attempt.lean_feedback)
                if proof_attempt.is_valid:
                    correct_proofs.append(proof_attempt.proof)
                    passed = True
                    passedat = proof_attempt.sample_id

                passatn[proof_attempt.sample_id] += 1 if passed else 0

            problem_result["proof_attempts"] = all_proofs
            problem_result["lean_feedbacks"] = lean_feedbacks
            problem_result["correct_proofs"] = correct_proofs
            problem_result["n_correct_proofs"] = len(correct_proofs)
            problem_result["passedat"] = passedat + 1
            problem_result["one_formal_proof"] = (
                correct_proofs[0] if len(correct_proofs) > 0 else None
            )

            self.results_dict[problem_id] = problem_result

            results_hf_list.append(
                {
                    "problem_id": problem_id,
                    "formal_statement": problem_result["formal_statement"],
                    "n_correct_proofs": problem_result["n_correct_proofs"],
                    "n_proofs": n_proofs,
                    "passedat": problem_result["passedat"],
                    "correct_proofs": problem_result["correct_proofs"],
                    "one_formal_proof": problem_result["one_formal_proof"],
                    "proof_attempts": problem_result["proof_attempts"],
                    "lean_feedbacks": problem_result["lean_feedbacks"],
                }
            )

        success_rate = n_success / len(self.dataset) * 100
        passatn = [x / len(self.dataset) for x in passatn]

        self.logger.info(
            f"Attempted {len(self.dataset)} problems with {self.sampling_params.n} samples each"
        )
        self.logger.info(f"Success rate: {success_rate:.2f}%")

        # dump results to json
        results_file = os.path.join(self.log_dir, "results.json")

        self.logger.info(f"Writing results to {results_file}")

        # save with encoding utf 8
        with open(results_file, "w") as f:
            json.dump(self.results_dict, f, indent=2, ensure_ascii=False)

        # dump results to HF
        results_ds = Dataset.from_list(results_hf_list)

        results_ds.push_to_hub(
            self.hf_results_path, revision=self.hf_results_full_revision, private=True
        )

        # save summary results
        summary_results = {
            "n_problems": len(self.dataset),
            "n_correct_proofs": n_success,
            "success_rate": success_rate,
            "passatn": passatn,
        }

        summary_results_ds = Dataset.from_list([summary_results])

        summary_results_ds.push_to_hub(
            self.hf_results_path,
            revision=self.hf_results_summary_revision,
            private=True,
        )

    def _task_scheduler(
        self,
        generation_queue,
        generation_output_queue,
        proof_verification_queue,
        generation_done,
        verification_done,
        verification_tasks_done,
    ):

        # Create generation tasks
        num_tasks = 0

        for i in range(0, len(self.problems), self.batch_size):
            batch = self.problems[i : i + self.batch_size]
            problem_ids = self.ids[i : i + self.batch_size]
            prompts = [
                self._prepare_prompt(problem, self.prompt_template) for problem in batch
            ]
            generation_task = GenerationTask(
                problem_ids=problem_ids,
                prompts=prompts,
                sampling_params=self.sampling_params,
            )
            generation_queue.put(generation_task)
            num_tasks += 1

        self.logger.info(f"Generated {num_tasks} generation tasks")

        generation_tasks_completed = 0
        while not verification_done.is_set():
            if not generation_done.is_set():
                try:
                    # Handle generation results
                    generation_results = generation_output_queue.get(timeout=1)
                    problem_ids = [result.problem_id for result in generation_results]
                    sample_ids = [result.sample_id for result in generation_results]
                    proofs = []
                    for result in generation_results:
                        problem_id = result.problem_id
                        formal_statement = self.results_dict[problem_id][
                            "formal_statement"
                        ]
                        proofs.append(formal_statement + result.output)

                    proof_task = ProofVerificationTask(
                        problem_ids=problem_ids, sample_ids=sample_ids, proofs=proofs
                    )
                    proof_verification_queue.put(proof_task)
                    generation_tasks_completed += 1
                    self.logger.info(
                        f"Generation tasks completed: {generation_tasks_completed}/{num_tasks}"
                    )
                    if generation_tasks_completed == num_tasks:
                        generation_done.set()
                except Empty:
                    pass

            if not verification_done.is_set():
                if verification_tasks_done.value == num_tasks:
                    self.logger.info("All verification tasks done")
                    verification_done.set()

    def _generator_worker(
        self, generation_queue, generation_output_queue, generation_done, gpu_id
    ):
        """
        Worker process for generating results using vLLM.
        """

        self.logger.info(f"Generator worker {gpu_id} started")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        llm = LLM(
            model=self.config["model_path"],
            tensor_parallel_size=1,  # Adjust as needed
            trust_remote_code=True,
        )

        while not generation_queue.empty():
            try:
                task = generation_queue.get(timeout=1)
                start_time = time.time()
                model_outputs = llm.generate(
                    task.prompts, task.sampling_params, use_tqdm=False
                )
                self.logger.info(
                    f"Process {gpu_id} took {(time.time() - start_time):.2f} seconds"
                    f"to generate samples for {len(model_outputs)} problems"
                )
                results = []
                for problem_id, model_output in zip(task.problem_ids, model_outputs):
                    for sample_id, output in enumerate(model_output.outputs):
                        gen_result = GenerationResult(
                            problem_id=problem_id,
                            sample_id=sample_id,
                            output=output.text,
                        )
                        results.append(gen_result)
                generation_output_queue.put(results)
            except Empty:
                if generation_done.is_set():
                    break

    def _verifier_worker(
        self,
        proof_verification_queue,
        verification_output_queue,
        verification_done,
        verification_tasks_done,
        process_id,
    ):
        """
        Worker process for verifying proofs.
        """

        self.logger.info(f"Verifier worker {process_id} started")
        while not verification_done.is_set():
            try:
                task = proof_verification_queue.get(timeout=1)
                start_time = time.time()
                has_error_list, lean_feedbacks = self.get_lean_feedbacks(
                    proofs=task.proofs,
                    problem_ids=task.problem_ids,
                    sample_ids=task.sample_ids,
                )

                self.logger.info(
                    f"Verifier worker {process_id} completed {len(task.problem_ids)}"
                    f" tasks in {(time.time() - start_time):.2f} seconds"
                )

                verification_results = []
                for (
                    problem_id,
                    sample_id,
                    proof,
                    has_error_status,
                    lean_feedback,
                ) in zip(
                    task.problem_ids,
                    task.sample_ids,
                    task.proofs,
                    has_error_list,
                    lean_feedbacks,
                ):
                    verification_result = ProofVerificationResult(
                        problem_id=problem_id,
                        sample_id=sample_id,
                        proof=proof,
                        is_valid=not has_error_status,
                        output=proof,
                        lean_feedback=lean_feedback,
                    )
                    verification_results.append(verification_result)

                self.logger.info(
                    f"Verifier worker {process_id} putting {len(verification_results)} results in the queue"
                )
                verification_output_queue.put(verification_results)

                with verification_tasks_done.get_lock():
                    verification_tasks_done.value += 1
                self.logger.info(
                    f"Verification tasks done: {verification_tasks_done.value}"
                )

                # TODO: remove this hack some time
                time.sleep(2)
                if verification_done.is_set() and proof_verification_queue.empty():
                    self.logger.info(
                        f"Verifier worker {process_id} exiting as verification is complete."
                    )
                    return

            except Empty:
                if verification_done.is_set() and proof_verification_queue.empty():
                    self.logger.info(
                        f"Verifier worker {process_id} exiting as verification is complete."
                    )
                    return

    def get_lean_feedbacks(
        self, proofs: List[str], problem_ids: List[str], sample_ids: List[str]
    ) -> Tuple[List[bool], List[Dict]]:
        """
        This function is used to get the lean feedback for a list of proofs.

        Returns 2 lists of the same length:
        - has_error_list: A list of boolean values indicating if the proof has an error.
        - lean_feedack: The compiler feedback for the proof.
        """

        if self.lean_feedback == "local":

            lean_results = parallel_lean4_feedback(
                lean_codes=proofs,
                num_workers=self.config["lean_workers"],
                max_retries=self.config["lean_retries"],
                timeout=self.config["lean_timeout"],
                memory_limit=self.config["lean_memory_limit"],
            )

            has_error_list = []
            for lean_result in lean_results:
                has_error_stat = has_error(
                    lean_result, accept_sorry=False, return_error_messages=False
                )
                has_error_list.append(has_error_stat)

            return has_error_list, lean_results

        elif self.lean_feedback == "server":

            uuids = []
            for problem_id, sample_id in zip(problem_ids, sample_ids):
                uuids.append(f"{problem_id}_{sample_id}_{get_datetime()}")

            samples = []
            for uuid, problem_id, sample_id, proof in zip(
                uuids, problem_ids, sample_ids, proofs
            ):
                samples.append(
                    {
                        "uuid": uuid,
                        "proof_id": f"{problem_id}_{sample_id}",
                        "proof": proof,
                    }
                )

            lean_results = lean4_client.batch_verify_proof(
                self.client,
                samples,
                timeout=self.config["lean_timeout"],
                num_threads=self.config["lean_workers"],
            )

            # the results are in random order so we need to reorder by uuid
            lean_results_dict = {}

            for res in lean_results:
                has_error_bool = not res.get("is_valid_no_sorry", False)
                lean_feedback_str = res.get("lean_feedback", "")
                lean_results_dict[res["uuid"]] = {
                    "has_error": has_error_bool,
                    "lean_feedback": lean_feedback_str,
                }

            has_error_list = []
            lean_feedbacks = []
            for uuid in uuids:
                lean_result = lean_results_dict[uuid]
                has_error_stat = lean_result["has_error"]
                lean_feedbacks.append(lean_result["lean_feedback"])
                has_error_list.append(has_error_stat)

            return has_error_list, lean_feedbacks

    def _prepare_prompt(self, problem: Dict[str, Any], user_prompt: str) -> str:
        """
        Prepare the prompt for the model based on the problem and an optional user-specified prompt.

        Args:
            problem: A dictionary containing problem details.
            user_prompt: An optional custom prompt template from the user.

        Returns:
            A string prompt for the model.
        """

        return user_prompt.format(**problem)


if __name__ == "__main__":
    """
    Example usage:
    python mp_evaluator.py --config configs/mpconfig_minif2f.yaml --output-dir eval_logs
    """
    parser = argparse.ArgumentParser(description="DeepSeek Evaluator CLI")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to the output directory"
    )

    args = parser.parse_args()
    evaluator = ProverEvaluator(args.config, args.output_dir)
    evaluator.evaluate()
