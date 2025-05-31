import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

from autoformalizer.clients import lean4_client
from autoformalizer.eval_utils import get_datetime
from autoformalizer.eval_utils.lean_feedback import has_error, parallel_lean4_feedback
from evaluation.evaluation_constants import (
    LEAN4_DEFAULT_HEADER,
    ProofVerificationResult,
    VLLMConfig,
)

torch.cuda.empty_cache()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

    def __init__(self, config_path: str, output_dir: str):
        """
        Initialize the evaluator.

        Args:
            config_path: Path to the configuration YAML file.
            output_dir: Path to the output directory to store the results
        """
        self.config = load_config(config_path)

        if "vllm" not in self.config:
            self.vllm_config = VLLMConfig()
        else:
            self.vllm_config = VLLMConfig(**self.config["vllm"])

        self.llm = LLM(
            model=self.config["model_path"],
            tensor_parallel_size=self.vllm_config.tensor_parallel_size,
            gpu_memory_utilization=self.vllm_config.gpu_memory_utilization,
            dtype=self.vllm_config.dtype,
            trust_remote_code=True,
            download_dir=f"{os.getenv('HOME')}/.cache/vllm/",
        )

        sampling_config = self.config["sampling_config"]

        self.sampling_params = SamplingParams(**sampling_config)

        log_dir = f"{self.config['model_path'].split('/')[-1]}_{get_datetime()}"
        self.log_dir = os.path.join(output_dir, log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        # write the config back to the output directory
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

        self.lean_workers = self.config["lean_workers"]
        self.lean_timeout = self.config["lean_timeout"]
        self.lean_memory_limit = self.config.get("lean_memory_limit", 512)
        self.lean_retries = self.config.get("lean_retries", 1)
        self.lean_feedback = self.config.get("lean_feedback", "local")
        if self.lean_feedback not in ["local", "server"]:
            raise ValueError(f"Invalid lean_feedback option: {self.lean_feedback}")

        if self.lean_feedback == "server":
            self.working_dir = Path(f"tmp/res_folder_{get_datetime()}")
            # comment this as not necessary anymore
            #self.working_dir.mkdir(parents=True, exist_ok=True)

            self.client = lean4_client.Lean4Client(
                "https://kimina.saas.moonshot.cn/lean4-evaluator",
                api_key=os.environ.get("MOONSHOT_LEAN4_API_KEY"),
            )

        self.prompt_template = self.config["prompt_template"]

        # whether to run compiler feedback or not
        self.do_evaluate = self.config["do_evaluate"]

        # whether to store intermediate results
        self.store_intermediate_results = self.config["store_intermediate_results"]

        # TODO: some logging related stuff that should be rewritten
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

        batch_size = self.config["batch_size"]

        results = []
        successful_proofs = 0
        total_time_start = time.time()
        self.logger.info(
            f"""Total dataset length: {len(self.problems)},
            Batch_size: {batch_size}\nPrompt template:
            {self.prompt_template}"""
        )

        # we store for each problem list of results (each of size `n_samples`)
        problem_results = {
            "total_problems": len(self.formal_statements),
            "success_rate": 0,
            "successful_proofs": 0,
        }

        for i in tqdm(
            range(0, len(self.formal_statements), batch_size), desc="Processing batches"
        ):
            batch = self.problems[i : i + batch_size]
            batch_ids = self.ids[i : i + batch_size]
            for idx in batch_ids:
                problem_results[idx] = {"results": [], "is_valid": False}

            try:
                batch_results = self.process_batch(
                    batch, batch_ids, self.prompt_template
                )
                results.extend(batch_results)
                self.logger.info(f"Data {i} to {i + batch_size} finished")
            except Exception as e:
                self.logger.error(f"Error processing batch: {str(e)}")

            # dump the intermediate results
            if self.store_intermediate_results:
                intermediate_results_dir = os.path.join(
                    self.log_dir, "intermediate_results"
                )

                os.makedirs(intermediate_results_dir, exist_ok=True)
                intermediate_results_file = os.path.join(
                    intermediate_results_dir,
                    f"intermediate_results_{i + batch_size}.json",
                )
                with open(intermediate_results_file, "w") as f:
                    json.dump(
                        [vars(r) for r in results], f, indent=2, ensure_ascii=False
                    )

                # delete the previous result if exists
                old_intermediate_result_file = os.path.join(
                    intermediate_results_dir, f"intermediate_results_{i}.json"
                )
                if os.path.exists(old_intermediate_result_file):
                    os.remove(old_intermediate_result_file)

                correct_proof_set = set()
                for r in results:
                    if r.is_valid:
                        correct_proof_set.add(r.problem_id)

                self.logger.info(
                    f"Number of correct proofs so far: {len(correct_proof_set)}"
                )
                self.logger.info(f"Ids: {correct_proof_set}")

        total_time = time.time() - total_time_start

        for r in results:
            problem_results[r.problem_id]["results"].append(r.is_valid)
            if r.is_valid:
                problem_results[r.problem_id]["is_valid"] = True

        for idx in self.ids:
            if problem_results[idx]["is_valid"]:
                successful_proofs += 1

        success_rate = successful_proofs / len(self.formal_statements) * 100
        problem_results["success_rate"] = success_rate
        problem_results["successful_proofs"] = successful_proofs

        total_evaluation_results = {
            "success_rate": success_rate,
            "total_problems": len(self.formal_statements),
            "successful_proofs": successful_proofs,
            "total_time": total_time,
        }

        self.logger.info(f"Evaluation results: {total_evaluation_results}")
        total_evaluation_results["results"] = [vars(r) for r in results]

        sample_results_file = os.path.join(self.log_dir, "summary.json")
        all_results_file = os.path.join(self.log_dir, "full_results.json")

        self.logger.info(f"Writing full evaluation results to {all_results_file}")
        # save with encoding utf 8
        with open(all_results_file, "w") as f:
            json.dump(total_evaluation_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Writing summary evaluation results to {sample_results_file}")
        with open(sample_results_file, "w") as f:
            json.dump(problem_results, f, indent=2)

        # delete the working directory altogether with os
        if (
            self.lean_feedback == "server"
            and self.working_dir is not None
            and os.path.exists(self.working_dir)
        ):
            shutil.rmtree(self.working_dir)

        return total_evaluation_results

    def process_batch(
        self,
        batch: List[Dict[str, Any]],
        batch_ids: List[str],
        prompt_template: Optional[str] = None,
    ) -> List[ProofVerificationResult]:
        """
        Wraps the _process_batch with try except block to catch errors.
        """

        try:
            return self._process_batch(batch, batch_ids, prompt_template)
        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}")
            raise e

    def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        batch_ids: List[str],
        prompt_template: Optional[str] = None,
    ) -> List[ProofVerificationResult]:
        """
        Args:
            batch: List of problems to process.
            verification_workers: Number of workers for parallel verification.
            few_shot_dataset: The list dataset containes few-shot examples.
            prompt_template: An optional custom prompt template from the user.
        Returns:
            results: List of output proof for the input batch.
        """
        prompts = [self._prepare_prompt(problem, prompt_template) for problem in batch]

        self.logger.info("Generating proofs...\n")
        model_outputs = self.llm.generate(prompts, self.sampling_params)

        self.logger.info("Done generating. Verifying proofs...\n")

        raw_datas = []
        problem_ids = []
        sample_ids = []
        proofs = []

        for problem, idx, model_output in zip(batch, batch_ids, model_outputs):
            problem_id = idx
            for sample_id, output in enumerate(model_output.outputs):
                proof = output.text
                raw_datas.append(output.text)
                proofs.append(problem["formal_statement"] + proof)
                problem_ids.append(str(problem_id))
                sample_ids.append(str(sample_id))

        results = []

        if self.do_evaluate:
            has_error_list, lean_results = self.get_lean_feedbacks(
                proofs, problem_ids, sample_ids
            )
        else:
            # only store the generation results
            has_error_list = [True] * len(proofs)
            lean_results = [None] * len(proofs)

        for problem_id, sample_id, proof, lean_result, has_error_stat, raw_data in zip(
            problem_ids, sample_ids, proofs, lean_results, has_error_list, raw_datas
        ):

            res = ProofVerificationResult(
                problem_id=problem_id,
                sample_id=sample_id,
                proof=proof,
                output=raw_data,
                is_valid=not has_error_stat,
                lean_feedback=lean_result,
            )

            results.append(res)

        return results

    # returns 2 lists of the same length: has_error, lean_feedback
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
                lean_feedback = res.get("lean_feedback", '')
                '''
                lean_feedback = json.loads(lean_feedback)
                if "error" not in lean_feedback:
                    has_error_bool = True
                if "error" in lean_feedback:
                    has_error_bool = (
                        has_error_bool or lean_feedback["error"] is not None
                    )
                lean_feedback = lean_feedback.get(
                    "response", {"error": "No response from the server"}
                )
                '''
                lean_results_dict[res["uuid"]] = {
                    "has_error": has_error_bool,
                    "lean_feedback": lean_feedback,
                }
                

            has_error_list = []
            lean_feedback = []
            for uuid in uuids:
                lean_result = lean_results_dict[uuid]
                has_error_stat = lean_result["has_error"]
                lean_feedback.append(lean_result["lean_feedback"])
                has_error_stat = has_error_stat or has_error(
                    lean_result["lean_feedback"], accept_sorry=False
                )
                has_error_list.append(has_error_stat)

            # delete the working directory altogether with os to be safe
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)
            # create it again
            # self.working_dir.mkdir(parents=True, exist_ok=True)

            return has_error_list, lean_feedback

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
    python evaluator.py --config configs/config_math.yaml --output-dir eval_logs
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
