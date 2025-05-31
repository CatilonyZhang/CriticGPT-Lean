import json
import os
import pathlib
import random
import sys
import time

import requests
from datasets import load_dataset
from loguru import logger

from autoformalizer.eval_utils import lean_feedback


class Lean4Client(object):

    def __init__(self, url, api_key) -> None:
        self.url = url
        self.api_key = api_key
        self.contexts = {}
        self.process_id = None

    def one_pass_verify(self, code, timeout):
        """verify the proof code and get result"""
        json_data = {
            "method": "one_pass_verify",
            "code": code,
            "timeout": timeout,
        }
        return self._query(json_data, n_retries=3)

    def _query(
        self, json_data: dict, n_retries: int = 0, retry_on_timeout=False
    ) -> dict:
        """One single method for sending all requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        method = json_data["method"]
        for attemp_i in range(n_retries + 1):
            logger.debug(
                f"[{attemp_i}] Sending request to {self.url}/{method} with data: {json_data}"
            )
            try:
                response = requests.post(
                    f"{self.url}/{method}", headers=headers, json=json_data
                )
                res = json.loads(response.text)
                if res.get("error") == "Lean process timed out" and retry_on_timeout:
                    time.sleep(0.1)
                    continue
                break
            except json.decoder.JSONDecodeError:
                res = {"error": f"JSONDecodeError with text: {response.text}"}
                if attemp_i < n_retries:
                    time.sleep(0.1 * attemp_i**2)
            except requests.exceptions.Timeout:
                res = {"error": "Server Inner TimeoutError"}
                if attemp_i < n_retries:
                    time.sleep(0.1)

            logger.debug(
                f"[{attemp_i}] Received from {self.url}/{method} with data: {res}"
            )
        return res


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    client = Lean4Client(
        "https://kimina.saas.moonshot.cn/lean4-evaluator",
        api_key=os.environ.get("MOONSHOT_LEAN4_API_KEY"),
    )
    ds = load_dataset("AI-MO/auto-proof-v1", split="train")
    working_dir = pathlib.Path("./scripts/jia/auto_proof_v1")

    def verify_proof(sample):
        code = sample["proof"]
        filepath = working_dir / f"{sample['uuid']}_{sample['proof_id']}.json"
        if filepath.exists():
            with open(filepath, "r") as f:
                res = json.load(f)
        else:
            res = client.one_pass_verify(code, timeout=60)
            with open(filepath, "w") as f:
                json.dump(res, f)
        if res.get("error") is not None:
            logger.error(f"Error in proof verification: {res.get('error')}")
        if res.get("response", None) is None:
            logger.error(f"None in respone: {res}")
        else:
            if "error decoding json response in leanrepl" in str(res):
                logger.error(f"Error decoding json response in leanrepl: {res}")

        if random.random() < 0.001:
            logger.info(f"Respone: {res}")

        has_error = res.get("error") or lean_feedback.has_error(res["response"])
        return {
            "server_feedback": json.dumps(res),
            "is_valid_server": not has_error,
        }

    # randomly select 100 samples
    # import random
    # ds = ds.select(random.sample(range(len(ds)), 2000))
    # ds = ds.select(range(2000))
    ds = ds.map(verify_proof, num_proc=300)

    df = ds.to_pandas()
    print(df.loc[:, ("server_feedback", "proof")])
    # Log statistics
    total_problems = len(df.uuid.unique())
    valid_problems = len(df.loc[df.is_valid].uuid.unique())
    valid_server_problems = len(df.loc[df.is_valid_server].uuid.unique())
    print(f"Total unique problems: {total_problems}")
    print(f"Valid unique problems: {valid_problems}")
    print(f"Server Valid unique problems: {valid_server_problems}")
    # ds.push_to_hub("AI-MO/math-test-inference-results-server", private=True)
    # ds.push_to_hub("AI-MO/aops-inference-results-server", private=True)
    ds.push_to_hub("AI-MO/auto-proof-v1-server", private=True)
