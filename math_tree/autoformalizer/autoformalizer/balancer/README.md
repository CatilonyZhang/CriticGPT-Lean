# **Load Balancer and VLLM Process Manager**

This script serves as a load balancer and process manager for VLLM workers, dynamically managing backend servers and forwarding requests through a Flask application.

---

## **Features**

- Starts multiple VLLM workers as backend servers.
- Dynamically balances HTTP requests across the backend servers.
- Monitors the health of backend processes and restarts them if they fail.
- Provides a simple Flask-based HTTP API for load balancing.

---

## **Installation**

### **1. Clone the Repository**

Ensure you have cloned the repository where the script resides.

```bash
git clone <repository-url>
cd <repository-directory>
```

### **2. Install Dependencies**

Use `pip` to install the required Python dependencies. It's recommended to use a virtual environment.

#### Create and Activate a Virtual Environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install Required Packages:

```bash
pip install -r requirements.txt
```

**`requirements.txt` example:**

```text
flask
requests
```

---

## **Usage**

### **1. Command Line Arguments**

Run the script with the following arguments:

```bash
python load_balancer.py <model> --num_vllm_worker <num_workers> --base_service_port <port> [--flask_port <flask_port>] [--api_key <api_key>]
```

#### **Arguments:**

- `<model>`: Path or name of the model to be served by VLLM (e.g., `deepseek-ai/DeepSeek-Prover-V1.5-RL`).
- `--num_vllm_worker`: Number of VLLM workers to start (e.g., `2`).
- `--base_service_port`: Base port for backend VLLM workers (default: `5001`).
- `--flask_port`: Port for the Flask server (default: `8000`).
- `--api_key`: APIKey used by the OpenAI library.
- `--revision`: The specific version or commit of the model to use.
- `--tensor_parallel_size`: The number of GPUs used by each VLLM worker. (default: `1`)

### **2. Example Usage**

Start the load balancer with 2 VLLM workers, each VLLM worker uses 2 GPUs, using `5001` and `5002` as backend ports, and `8000` as the Flask server port:

```bash
python load_balancer.py "deepseek-ai/DeepSeek-Prover-V1.5-RL" --num_vllm_worker=2 --base_service_port=5001 --flask_port=12500 --api_key="test_apikey" --tensor_parallel_size=2
```

This requires a total of 4 GPUs (`num_vllm_worker` * `tensor_parallel_size`). 

If the computer has only 3 GPUs, you will see the following error:

```
Traceback (most recent call last):
  File "/home/wangran/Workspace/autoformalizer/autoformalizer/balancer/load_balancer.py", line 199, in <module>
    check_gpu_availability(tensor_parallel_size, num_vllm_worker)
  File "/home/wangran/Workspace/autoformalizer/autoformalizer/balancer/load_balancer.py", line 71, in check_gpu_availability
    raise RuntimeError(
RuntimeError: Insufficient GPUs: Required 4, but only 3 are available. Please adjust 'tensor_parallel_size' or 'num_vllm_worker'
```

And please note that tensor parallel must be divisible by 64, otherwise the following error will occur:

```
File "/home/wangran/miniconda/lib/python3.12/site-packages/vllm/config.py", line 407, in verify_with_parallel_config
    raise ValueError(
ValueError: Total number of attention heads (64) must be divisible by tensor parallel size ({tensor_parallel_size}).
```


You can call the model endpoint using the following code

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:12500/v1",
    api_key="test_apikey",
)

completion = client.chat.completions.create(
  model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```

- If apikey is not specified, any string can be used as apikey to access this endpoint.

---

## **How It Works**

1. **Starts VLLM Workers:**

   - Launches the specified number of VLLM workers on ports incrementing from the base service port.
   - Each worker is dynamically registered as a backend server.

2. **Flask-Based Load Balancer:**

   - Forwards incoming HTTP requests to one of the backend servers using a random selection strategy.

3. **Process Monitoring:**
   - Periodically checks the health of the VLLM processes and restarts them if they fail.

---

## **Endpoints**

### **1. Load Balancer Root (Dynamic Forwarding)**

- **URL:** `http://<flask_host>:<flask_port>/<path>`
- **Methods:** `GET`, `POST`, `PUT`, `DELETE`

**Example:**

```bash
curl -X GET http://127.0.0.1:8000/v1/models
```

---

## **Stopping the Script**

The script supports graceful termination, ensuring that both the Flask server and all VLLM backend processes are stopped properly. However, **Ctrl+C** (or sending termination signals like `SIGINT` or `SIGTERM`) must be pressed at the correct time to ensure a clean shutdown.

### **When to Press Ctrl+C**

1. **After the script has fully initialized:**

   - Wait for all VLLM workers to start. You will see log messages indicating that each VLLM worker has started on its respective port (e.g., `Started VLLM worker 0 on port 5001`).
   - Ensure the Flask server has started. Look for the log message:
     ```
     Starting Flask server on port <flask_port>
     ```
     Example:
     ```
     INFO - Starting Flask server on port 8000
     ```

2. **While the script is running normally:**
   - You can press **Ctrl+C** at any time after initialization to trigger the shutdown process. This will:
     - Terminate all active VLLM worker processes.
     - Stop the Flask server.

### **What Happens on Shutdown**

When **Ctrl+C** is pressed:

1. A `SIGINT` signal is captured by the script.
2. The script logs the following message:
   ```
   INFO - Received exit signal, shutting down...
   ```
3. It gracefully terminates all VLLM workers:
   - Each worker process is sent a termination signal (`terminate`).
   - The script waits for each worker to exit cleanly.
4. The Flask server stops.
5. The script exits fully.

### **What Happens If Ctrl+C Is Pressed Too Early**

- If you press **Ctrl+C** **before** the script has fully initialized (e.g., while VLLM workers are still starting), some processes might not be terminated properly.
- In such cases:
  - The script will attempt to terminate processes that have started.
  - Any partially started VLLM workers may continue running in the background. You may need to manually terminate them using their process IDs (`PID`).

### **Restart Policies**

- VLLM workers that fail are restarted after a 30-second delay.
