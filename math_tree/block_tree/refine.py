"""
This script is used to refine mathematical statements into standard LaTeX format.
"""

import os
import re
import json
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
import sys
sys.path.append('/lustre/fast/fast/txiao/zly/lean/math_tree')
from datasets import load_dataset
from config import *
# from llm.deploy_llm import get_llm_url
from llm.run_llm_api import run_llm, run_llm_openai
from llm.prompt.refine import refine_prompt
import logging
from datetime import datetime
from multiprocessing import Pool
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=os.path.join(data_path, 'Omni-MATH'),
                      help='Path to dataset containing math problems')
    parser.add_argument('--llm_url', type=str, default='http://127.0.0.1:8000',
                      help='URL endpoint for LLM service')
    parser.add_argument('--model_name', type=str, default='llama_3_70b ',
                      help='Model name')
    parser.add_argument('--use_openai', action='store_true',
                      help='Whether to use OpenAI API')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Number of parallel processes')
    return parser.parse_args()

class Refiner:
    def __init__(self, args):
        
        self.dataset_path = args.dataset_path
        self.dataset = load_dataset('json', data_files=os.path.join(self.dataset_path, 'test.jsonl'))['train']
        self.dataset = self.dataset.select(range(20))
        self.llm_url = args.llm_url
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.use_openai = args.use_openai
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_json = os.path.join(results_path, 
                                      f"refined_statements_{self.timestamp}.json")
        self.logging_path = os.path.join(log_path,
                                       f"refiner_{self.timestamp}.log")
        
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

        

    def refine_statement(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine a single mathematical statement"""
        try:
            problem = data['problem']
            solution = data['solution']
            statement = f"Problem: {problem}\nAnswer: {solution}\n"
            logging.info(f"Refining statement: {statement}")
            prompt = refine_prompt + statement
            if self.use_openai:
                refined = run_llm_openai(self.model_name, prompt)
            else:
                refined = run_llm(self.model_name, prompt, llm_url=self.llm_url)
            
            data['refined_statement'] = refined
            # extract the latex from the refined statement
            latex_pattern = r'```latex\n(.*?)\n```'
            latex_match = re.search(latex_pattern, refined, re.DOTALL)
            if latex_match:
                latex_code = latex_match.group(1).strip()
                data['latex_code'] = latex_code
            return data
            
        except Exception as e:
            logging.error(f"Error refining statement: {str(e)}")
            return data

    def process_dataset(self):
        """Process entire dataset"""
        logging.info(f"Processing dataset with batch size {self.batch_size}")
        results = []
        for data in tqdm(self.dataset):
            refined_data = self.refine_statement(data)
            results.append(refined_data)

        # Save results
        logging.info(f"Saving results to {self.output_json}")
        with open(self.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

    def process_dataset_parallel(self):
        """Process dataset in parallel"""
        with Pool(self.batch_size) as p:
            results = list(tqdm(p.imap(self.refine_statement, self.dataset), total=len(self.dataset)))
        # save the results
        with open(self.output_json, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    refiner = Refiner(args)
    refiner.process_dataset()