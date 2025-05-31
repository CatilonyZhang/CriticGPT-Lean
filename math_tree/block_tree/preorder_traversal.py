import json
import logging
from typing import List, Dict, Any
import sys
sys.path.append('/lustre/fast/fast/txiao/zly/lean/math_tree')
from llm.run_llm_api import run_llm, run_llm_openai
import argparse
import pdb
import os
from datetime import datetime
from config import *
from multiprocessing import Pool
from tqdm import tqdm
import logging


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROMPT_TEMPLATES = {
    "Proposition": """
        You are a mathematician solving a proof problem. The main proposition is:

        **Main Proposition: {main_proposition}**

        The current proposition to prove is:

        **Proposition: {content}**

        This proposition depends on the following conditions or cases:
        {dependencies}

        **Instructions for the Proof:**
        1. Clearly state the proposition and its role in the proof of the main proposition.
        2. Provide a step-by-step proof, ensuring each step is logically sound and rigorously justified.
        3. Explicitly reference any dependencies (conditions, cases, or auxiliary results) and explain how they are used in the proof.
        4. Avoid hand-waving or intuitive explanations; use precise mathematical reasoning.
        5. Conclude by summarizing how the proposition contributes to the proof of the main proposition.
    """,
    
    "Auxiliary Condition": """
    
        You are a mathematician solving a proof problem. The main proposition is:

        **Main Proposition: {main_proposition}**

        The current auxiliary condition to establish is:

        **Auxiliary Condition: {content}**

        This condition is used to support the following proposition or case:
        {dependencies}

        **Instructions for the Proof:**
        1. Clearly state the auxiliary condition and its purpose in the proof.
        2. Provide a rigorous proof for this condition, ensuring each step is justified with precise mathematical reasoning.
        3. Explicitly explain how this condition supports the dependent proposition or case.
        4. Avoid digressing into the proof of the main proposition; focus solely on proving the auxiliary condition.
        5. Conclude by summarizing how this condition contributes to the overall proof.
    """,
    
    "Case": """
        You are a mathematician solving a proof problem. The main proposition is:

        **Main Proposition: {main_proposition}**

        The current case to consider is:

        **Case: {content}**

        This case is part of the proof for the following proposition:
        {dependencies}

        **Instructions for the Proof:**
        1. Clearly state the case and its role in the proof of the main proposition.
        2. Provide a step-by-step proof for this case, ensuring each step is rigorously justified.
        3. Explicitly reference any dependencies (conditions, auxiliary results, or sub-cases) and explain how they are used in the proof.
        4. Avoid hand-waving or intuitive explanations; use precise mathematical reasoning.
        5. Conclude by summarizing how this case contributes to the proof of the main proposition.
    """,
    
    "Sub-Case": """
        You are a mathematician solving a proof problem. The main proposition is:

        **Main Proposition: {main_proposition}**
        
        The current sub-case to consider is:
        
        **Sub-Case: {content}**
        
        This sub-case is part of the following parent case:
        **Parent Case: {dependencies}**
        
        **Instructions for the Proof:**
        1. Clearly state the sub-case and its role in the proof of the parent case.
        2. Provide a detailed, step-by-step proof for this sub-case, ensuring each step is rigorously justified.
        3. Explicitly reference the parent case and explain how this sub-case supports it.
        4. Avoid digressing into the proof of the main proposition; focus solely on proving the sub-case.
        5. Conclude by summarizing how this sub-case contributes to the proof of the parent case and, ultimately, the main proposition.
    """
}

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
    return parser.parse_args()


class TraversalProofGenerator:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.use_openai = args.use_openai
        self.model_name = args.model_name
        self.llm_url = args.llm_url
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_json = os.path.join(proof_path,
                                      f"proof_{self.timestamp}.json")
        self.logging_path = os.path.join(log_path,
                                       f"proof_{self.timestamp}.log")
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.logging_path)
            ]
        )

        logging.info(f"Using LLM endpoint: {self.llm_url}")
        logging.info(f"Results will be saved to: {self.output_json}")

    def generate_proof(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate proof for a single block tree"""
        try:
            json_blocktree = data['json_objects']
            if not json_blocktree:
                logging.warning("No JSON block tree found")
                return data
                
            logging.info("Generating proof from block tree")
            proof = self.generate_structured_proof(json_blocktree, data)
            data['proof'] = proof
            return data
        except Exception as e:
            pdb.set_trace()
            logging.error(f"Error generating proof: {str(e)}")
            return data
    
    def generate_structured_proof(self, json_blocktree: List[Dict[str, Any]], data: Dict[str, Any]) -> str:
        """Generate proof steps by traversing the proof tree and using LLM"""
        logger.info("Finding root node")
        # Find root node that has no dependencies
        root = next(node for node in json_blocktree if not node.get("dependencies"))
        logger.info(f"Root node found: {root['id']}")
        
        # Get nodes in preorder traversal order
        traversal_order = self.preorder_traversal(json_blocktree, root["id"])
        
        # Move root node to end of traversal order
        traversal_order.remove(root["id"])
        traversal_order.append(root["id"])
        
        # Generate proof steps following preorder traversal
        proof_steps = []
        context = ""  # Store previous proof steps as context
        logger.info("Starting proof generation following preorder traversal")
        for node_id in traversal_order:
            logger.info(f"Processing node {node_id}")
            node = next(n for n in json_blocktree if n["id"] == node_id)
            
            # Get parent content for dependencies
            parent_contents = ""
            if node.get("dependencies"):
                parent_contents = "\n".join([n["content"] for n in json_blocktree if n["id"] in node["dependencies"]])
            
            # Construct prompt based on node type

            prompt = PROMPT_TEMPLATES.get(node["type"], "").format(
                main_proposition=data.get("latex_code", ""),
                content=node["content"],
                dependencies=parent_contents
            )
            
            # Generate proof using LLM
            logger.info(f"Generating proof for node {node_id} using LLM")
            if self.use_openai:
                proof = run_llm_openai(self.model_name, prompt)
            else:
                proof = run_llm(self.model_name, prompt, llm_url=self.llm_url)
            
            # Update context with current node's proof
            context += f"\n{node['type']}: {node['content']}\nProof: {proof}\n"
            proof_steps.append({
                "id": node_id,
                "type": node["type"],
                "content": node["content"],
                "proof": proof
            })
            logger.debug(f"Added proof step for node {node_id}")
        
        logger.info(f"Completed proof generation for {len(proof_steps)} steps")
        return proof_steps

    def preorder_traversal(self, json_blocktree: List[Dict[str, Any]], root_id: str) -> List[str]:
        """Get nodes in preorder traversal order starting from root"""
        logger.info(f"Starting preorder traversal from root node {root_id}")
        result = []
        def dfs(node_id: str):
            result.append(node_id)
            children = [n["id"] for n in json_blocktree if node_id in n.get("dependencies", [])]
            logger.debug(f"Found {len(children)} children for node {node_id}")
            for child in children:
                dfs(child)
        dfs(root_id)
        logger.info(f"Completed traversal, found {len(result)} nodes")
        return result

    def generate_proof_multi_process(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate proof for a list of block trees"""
        results = []
        with Pool(processes=self.batch_size) as pool:
            results = list(tqdm(pool.imap(self.generate_proof, dataset), total=len(dataset), desc="Generating proofs"))
        return results
        

if __name__ == "__main__":
    args = parse_args()
    generator = TraversalProofGenerator(args)
    
    # Load dataset
    with open(args.dataset_path) as f:
        dataset = json.load(f)
    proofs = generator.generate_proof_multi_process(dataset)
    # save proofs to json
    with open(generator.output_json, "w") as f:
        json.dump(proofs, f, indent=4)
