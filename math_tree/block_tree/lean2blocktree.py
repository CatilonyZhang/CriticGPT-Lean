import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import yaml
from openai import OpenAI
from tqdm import tqdm
from datasets import concatenate_datasets, load_dataset
from llm.utils import get_datetime
from llm.prompt.lean2blocktree import lean2blocktree
import multiprocessing
import pdb
import re



def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {str(e)}")

    return config


class BlockTree_Generator:
    def __init__(self, config_path: str, output_dir: str):
        self.config = load_config(config_path)
        self.output_dir = output_dir

        # set output log
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

        self.prompt_template = lean2blocktree
        self.formal_proofs = [self.prompt_template + f"The lean4 formal proof is: {proof}" for proof in self.dataset[self.config["formal_proof"][0]]]
        self.formal_proofs = self.formal_proofs
        self.model_path = self.config["model_path"]

        self.batch_size = self.config["batch_size"]

        # Load the sampling configuration
        self.sampling_config = self.config["sampling_config"]

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

    def evaluate_openai_single(self, formal_proofs: str) -> Dict[str, Any]:
        """
        Evaluate a dataset of problems using OpenAI
        """
        client = OpenAI(
            api_key = "",
            base_url = "https://ark.cn-beijing.volces.com/api/v3",
        )
        completion = client.chat.completions.create(
            model = self.model_path,
            messages = [
                {"role": "system", "content": "You are dou bao, an AI assistant developed by ByteDance"},
                {"role": "user", "content": formal_proofs},
            ],
            **self.sampling_config
        )
        whole_response = completion.choices[0].message.content
        # extract the block tree from the result using regex within ```markdown  and ```
        block_tree = re.search(r"```markdown\n(.*?)\n```", whole_response, re.DOTALL)
        if block_tree:
            block_tree = block_tree.group(1)
        else:
            block_tree = ""
        
        result = {
            "block_tree": block_tree,
            "whole_response": whole_response
        }
        return result

    def evaluate_openai_multi(self) -> List[Dict[str, Any]]:
        """
        Evaluate a dataset of problems using OpenAI in parallel
        """
        self.logger.info(
            f"""Total dataset length: {len(self.formal_proofs)},
            Batch_size: {self.batch_size}\nPrompt template:
            {self.prompt_template}"""
        )

        process_num = multiprocessing.cpu_count() # or batch_size
        with multiprocessing.Pool(process_num) as pool:
            results = list(
                tqdm(
                    pool.imap(self.evaluate_openai_single, self.formal_proofs),
                    total=len(self.formal_proofs),
                )
            )
        # add the refined results to the dataset
        whole_responses = [result["whole_response"] for result in results]
        blocktrees = [result["block_tree"] for result in results]
        # if len(results) != len(self.dataset):
        #     # cascade the results with "" if the length is not equal
        #     results = results + [""] * (len(self.dataset) - len(results))
        if len(whole_responses) != len(self.dataset):
            whole_responses = whole_responses + [""] * (len(self.dataset) - len(whole_responses))
        if len(blocktrees) != len(self.dataset):
            blocktrees = blocktrees + [""] * (len(self.dataset) - len(blocktrees))
        self.dataset = self.dataset.add_column("block_tree", blocktrees)
        self.dataset = self.dataset.add_column("whole_response", whole_responses)
        # store the new dataset to the output directory
        self.dataset.save_to_disk(self.log_dir)
        # create a new dataset collection in huggingface 
        # and upload the dataset to the hub
        self.logger.info(f"Saved the blocktree to {self.log_dir}")
        self.dataset.push_to_hub(
            "zhouliang/{dataset_name}".format(dataset_name=self.config["data_path"].split("/")[-1]),
            token="",
            private=True,
            revision="blocktree"
        )
        return results


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
    evaluator = BlockTree_Generator(args.config, args.output_dir)
    evaluator.evaluate_openai_multi()
    

