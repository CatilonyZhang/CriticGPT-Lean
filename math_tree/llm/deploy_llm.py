import vllm 
import argparse
import subprocess
import json
import os
import multiprocessing
import socket
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import sys
sys.path.append('/lustre/fast/fast/txiao/zly/lean/math_tree')
from config import *

def get_internal_ip():
    """Get the internal IP address of the current machine"""
    hostname = socket.gethostname()
    internal_ip = socket.gethostbyname(hostname)
    return internal_ip

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Deploy vLLM model server')
    parser.add_argument("--model_name", type=str, default=llama_3_70b,
                      help="Name/path of the model to deploy")
    parser.add_argument("--port", type=int, default=8000,
                      help="Port to run the server on") 
    parser.add_argument("--num_gpus", type=int, default=4,
                      help="Number of GPUs to use")
    return parser.parse_args()

def start_server(model_name, port, num_gpus):
    """Start the vLLM server"""
    cuda_devices = ",".join(str(i) for i in range(num_gpus))
    cmd = f"CUDA_VISIBLE_DEVICES={cuda_devices} python -m vllm.entrypoints.openai.api_server"
    cmd += f" --model {model_name}"
    cmd += f" --port {port}"
    cmd += " --host 0.0.0.0"
    cmd += f" --tensor-parallel-size {num_gpus}"
    cmd += " --dtype half"
    cmd += f" > {log_path}/llm_output.log 2>&1 &"
    
    subprocess.run(cmd, shell=True)

def main():
    """Main function to deploy the server"""
    args = parse_args()
    
    internal_ip = get_internal_ip()
    server_url = f"http://{internal_ip}:{args.port}"
    
    print("\n=== LLM Server Information ===")
    print(f"Internal IP: {internal_ip}")
    print(f"Port: {args.port}")
    print(f"Server URL: {server_url}")
    print("=========================\n")

    start_server(args.model_name, args.port, args.num_gpus)
    return server_url


def get_llm_url():
    """Get the URL of the running LLM server"""
    args = parse_args()
    internal_ip = get_internal_ip()
    return f"http://{internal_ip}:{args.port}", args.model_name

if __name__ == "__main__":
    llm_url = main()
    