from flask import Flask, request
import random
import requests
import logging
import os
import time
import subprocess
import argparse
import signal
import sys
import torch
import threading

app = Flask(__name__)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Backend servers (populated dynamically)
BACKEND_SERVERS = []

terminate_event = threading.Event()

# Flask Route for Load Balancing
@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def load_balancer(path):
    if not BACKEND_SERVERS:
        logger.error("No backend servers available!")
        return "No backend servers available", 502

    target_server = random.choice(BACKEND_SERVERS)
    target_url = f"{target_server}/{path}"

    logger.info(
        f"Forwarding request to {target_server} | Path: /{path} | Method: {request.method}"
    )

    try:
        response = requests.request(
            method=request.method,
            url=target_url,
            headers={key: value for key, value in request.headers},
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
        )

        logger.info(
            f"Response from {target_server} | Status Code: {response.status_code} | Path: /{path}"
        )

        return (response.content, response.status_code, response.headers.items())
    except requests.RequestException as e:
        logger.error(f"Error connecting to {target_server}: {e}")
        return f"Error connecting to backend: {e}", 502

def check_gpu_availability(tensor_parallel_size, num_vllm_worker):
    """
    Checks if the total required GPUs are available.
    """
    total_required_gpus = tensor_parallel_size * num_vllm_worker
    available_gpus = torch.cuda.device_count()
    
    if available_gpus == 0:
        raise RuntimeError("No GPUs are available. Please ensure CUDA is properly configured.")
    
    if total_required_gpus > available_gpus:
        raise RuntimeError(
            f"Insufficient GPUs: Required {total_required_gpus}, but only {available_gpus} are available. "
            "Please adjust 'tensor_parallel_size' or 'num_vllm_worker'."
        )
    logger.info(f"Available GPUs: {available_gpus}. Total required GPUs: {total_required_gpus}.")
    

def stop_all_processes(processes):
    logger.info("Stopping all VLLM processes...")
    for worker_index, process in processes.items():
        if process.poll() is None:  # 如果进程仍在运行
            logger.info(f"Terminating VLLM worker {worker_index}")
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning(f"Worker {worker_index} did not terminate in time. Killing it.")
                process.kill()
                process.wait()
    logger.info("All VLLM processes stopped.")


def handle_exit_signal(signal_received, frame):
    logger.info("Received exit signal, shutting down...")
    terminate_event.set()  # Signal threads to stop
    stop_all_processes(processes)
    sys.exit(0)


# VLLM Process Management
def start_vllm_process(
    worker_index, model, base_service_port, api_key=None, revision=None, tensor_parallel_size=1
):
    service_port = base_service_port + worker_index
    commands = [
        "vllm",
        "serve",
        "--host",
        "0.0.0.0",
        "--port",
        str(service_port),
    ]

    if api_key:
        commands.extend(["--api-key", api_key])

    if revision:
        commands.extend(["--revision", revision])

    commands.append(model)

    logger.info(f"shell command: {' '.join(commands)}")

    # Configure CUDA_VISIBLE_DEVICES based on tensor_parallel_size
    if tensor_parallel_size > 1:
        visible_devices = ",".join(
            str(worker_index * tensor_parallel_size + i) for i in range(tensor_parallel_size)
        )
        commands.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    else:
        visible_devices = str(worker_index)

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": visible_devices,
    }
    process = subprocess.Popen(commands, env=env)
    BACKEND_SERVERS.append(f"http://127.0.0.1:{service_port}")
    logger.info(f"Started VLLM worker {worker_index} on port {service_port} with CUDA_VISIBLE_DEVICES={visible_devices}")
    return process


def manage_processes(args, processes):
    while not terminate_event.is_set():
        for worker_index, process in processes.items():
            if process.poll() is not None:
                logger.error(
                    f"Process {worker_index} exited with code {process.returncode}"
                )

                # Restart the server after a delay
                time.sleep(30.0)
                logger.info(f"Restarting VLLM worker {worker_index}")
                processes[worker_index] = start_vllm_process(
                    worker_index,
                    args.model,
                    args.base_service_port,
                    args.api_key,
                    args.revision,
                )
            else:
                logger.info(f"Process {worker_index} is running normally")
        time.sleep(10.0)


if __name__ == "__main__":
    """
    python load_balancer.py "deepseek-ai/DeepSeek-Prover-V1.5-RL" --num_vllm_worker=2 --base_service_port=5001
    """
    parser = argparse.ArgumentParser(
        description="Start multiple VLLM processes with load balancing."
    )
    parser.add_argument("model", type=str, help="Path/Name to the model.")

    parser.add_argument(
        "--num_vllm_worker", type=int, required=True, help="The number of VLLM workers."
    )

    parser.add_argument(
        "--base_service_port",
        type=int,
        default=5001,
        help="Base port for VLLM workers.",
    )

    parser.add_argument(
        "--flask_port", type=int, default=8000, help="Port for the Flask server."
    )

    parser.add_argument(
        "--api_key", type=str, default=None, help="APIKey used by the OpenAI library."
    )

    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="The number of GPUs used by each VLLM worker"
    )

    parser.add_argument("--revision", type=str, default=None, help="Model revision.")

    args = parser.parse_args()
    
    num_vllm_worker = args.num_vllm_worker
    tensor_parallel_size = args.tensor_parallel_size

    check_gpu_availability(tensor_parallel_size, num_vllm_worker)

    # Start VLLM workers
    processes = {}
    for worker_index in range(num_vllm_worker):
        processes[worker_index] = start_vllm_process(
            worker_index,
            args.model,
            args.base_service_port,
            args.api_key,
            args.revision,
            tensor_parallel_size
        )

    # Start process management in a separate thread
    import threading

    process_manager_thread = threading.Thread(
        target=manage_processes, args=(args, processes), daemon=True
    )
    process_manager_thread.start()

    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    # Start Flask server
    logger.info(f"Starting Flask server on port {args.flask_port}")
    app.run(host="0.0.0.0", port=args.flask_port)
