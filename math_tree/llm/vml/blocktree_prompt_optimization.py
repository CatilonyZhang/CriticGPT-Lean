from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
import numpy as np
import random
import argparse
import concurrent
import os
import yaml
import openai 
from llm.utils import get_datetime
from llm.prompt.lean2blocktree import lean2blocktree
from datasets import load_dataset, concatenate_datasets
os.environ["OPENAI_API_KEY"] = ""

def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {str(e)}")

    return config

class VML_PromptOpt:
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

        self.formal_proofs = [self.prompt_template + f"The lean4 formal proof is: {proof}" for proof in self.dataset["proof"]]

        self.model_path = self.config["model_path"]

        

        
        