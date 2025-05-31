"""
This script generates block trees from mathematical statements using an LLM.
"""

import os
import re
import json
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
import sys
import logging
from datetime import datetime
from multiprocessing import Pool
sys.path.append('/lustre/fast/fast/txiao/zly/lean/math_tree')
from llm.prompt.blocktree_prompt import blocktree_prompt_update
from config import block_tree_path, log_path
from datasets import load_dataset
from llm.run_llm_api import run_llm, run_llm_openai
from huggingface_hub import HfApi
import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to dataset containing math problems')
    parser.add_argument('--llm_url', type=str, default='http://127.0.0.1:8000',
                      help='URL endpoint for LLM service')
    parser.add_argument('--model_name', type=str, default='gpt-4o',
                      help='Model name')
    parser.add_argument('--use_openai', action='store_true',
                      help='Whether to use OpenAI API')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Number of parallel processes')
    parser.add_argument('--submit_to_hf', action='store_true',
                      help='Whether to submit to HF')
    return parser.parse_args()

class BlockTreeGenerator:
    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.dataset = load_dataset('json', data_files=self.dataset_path)['train']
        self.dataset = self.dataset.select(range(20))
        self.llm_url = args.llm_url
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.use_openai = args.use_openai
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_json = os.path.join(block_tree_path,
                                      f"blocktree_{self.timestamp}.json")
        self.logging_path = os.path.join(log_path,
                                       f"blocktree_{self.timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.logging_path)
            ]
        )

        logging.info(f"Dataset loaded from: {self.dataset_path}")
        logging.info(f"Using LLM endpoint: {self.llm_url}")
        logging.info(f"Results will be saved to: {self.output_json}")
        self.submit_to_hf = args.submit_to_hf

    def generate_blocktree(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate block tree for a single mathematical statement"""
        try:
            statement = data['latex_code']
            if not statement:
                logging.warning(f"No refined statement found")
                return data
                
            logging.info(f"Generating block tree for statement: {statement}")
            prompt = blocktree_prompt_update + statement
            if self.use_openai:
                block_tree = run_llm_openai(self.model_name, prompt)
            else:
                block_tree = run_llm(self.model_name, prompt, llm_url=self.llm_url)
            
            # Extract the block tree structure and JSON objects
            tree_pattern = r'```\nProposition1(.*?)\n```'
            json_pattern = r'```json\n(\[[\s\S]*?\]\n)```'
            
            tree_match = re.search(tree_pattern, block_tree, re.DOTALL)
            json_match = re.search(json_pattern, block_tree, re.DOTALL)
            
            if tree_match:
                tree_structure = tree_match.group(0).strip()
                data['tree_structure'] = tree_structure
            else:
                logging.warning("No tree structure found in block tree output")
                data['tree_structure'] = None
                
            if json_match:
                try:
                    json_str = json_match.group(1).strip()
                    json_objects = json.loads(json_str)
                    data['json_objects'] = json_objects
                except json.JSONDecodeError:
                    logging.warning("Failed to parse JSON objects from block tree output")
                    data['json_objects'] = None
            else:
                logging.warning("No JSON objects found in block tree output")
                data['json_objects'] = None
            
            data['block_tree'] = block_tree
            
            return data
            
        except Exception as e:
            logging.error(f"Error generating block tree: {str(e)}")
            return data

    def process_dataset(self):
        """Process entire dataset"""
        logging.info(f"Processing dataset with batch size {self.batch_size}")
        results = []
        for data in tqdm(self.dataset):
            refined_data = self.generate_blocktree(data)
            results.append(refined_data)

        # Save results
        logging.info(f"Saving results to {self.output_json}")
        with open(self.output_json, 'w') as f:
            json.dump(results, f, indent=4)
        
        return results

    def process_dataset_parallel(self):
        """Process dataset in parallel"""
        with Pool(self.batch_size) as p:
            results = list(tqdm(p.imap(self.generate_blocktree, self.dataset), total=len(self.dataset)))
        # save the results
        with open(self.output_json, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    args.dataset_path = '/lustre/fast/fast/txiao/zly/lean/math_tree/results/refined_statements_20250103_100225.json'
    block_tree_generator = BlockTreeGenerator(args)
    block_tree_generator.process_dataset()
