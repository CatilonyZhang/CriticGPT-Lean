import os
import re
import json
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
import sys
from openai import OpenAI
import yaml
from llm.utils import get_datetime
from llm.prompt.refine import refine_prompt
from datasets import concatenate_datasets, load_dataset
import pdb
import multiprocessing
import logging

def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {str(e)}")

    return config

class Refiner:
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

        self.has_header = self.config["has_header"]
        self.problems = self.dataset[self.config["informal_statement"]]
        self.answers = self.dataset[self.config["answer"]]
        self.prompt_template = refine_prompt
        self.ori_statements = [self.prompt_template + f"Problem: {problem}\nAnswer: {answer}\nthe refined statement is: ```markdown \n" for problem, answer in zip(self.problems, self.answers)]
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

    
    def evaluate_openai_single(self, ori_statement: str) -> Dict[str, Any]:
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
                {"role": "user", "content": ori_statement},
            ],
            **self.sampling_config
        )
        return completion.choices[0].message.content


    def evaluate_openai_multi(self) -> List[Dict[str, Any]]:
        """
        Evaluate a dataset of problems using OpenAI in parallel
        """
        self.logger.info(
            f"""Total dataset length: {len(self.ori_statements)},
            Batch_size: {self.batch_size}\nPrompt template:
            {self.prompt_template}"""
        )

        process_num = multiprocessing.cpu_count() # or batch_size
        with multiprocessing.Pool(process_num) as pool:
            results = list(
                tqdm(
                    pool.imap(self.evaluate_openai_single, self.ori_statements),
                    total=len(self.ori_statements),
                )
            )
        # add the refined results to the dataset
        if len(results) != len(self.dataset):
            # cascade the results with "" if the length is not equal
            results = results + [""] * (len(self.dataset) - len(results))
        self.dataset = self.dataset.add_column("refined_statement", results)
        # store the new dataset to the output directory
        self.dataset.save_to_disk(self.log_dir)
        # create a new dataset collection in huggingface 
        # and upload the dataset to the hub
        self.logger.info(f"Saved the refined dataset to {self.log_dir}")
        self.dataset.push_to_hub(
            "zhouliang/{dataset_name}".format(dataset_name=self.config["data_path"].split("/")[-1]),
            token="",
            private=True
        )

        return results
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Refiner CLI")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to the output directory"
    )

    args = parser.parse_args()
    refiner = Refiner(args.config, args.output_dir)
    refiner.evaluate_openai_multi()