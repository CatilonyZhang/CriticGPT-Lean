import json
import os
import time

import requests
from loguru import logger


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
    # client = Lean4Client("http://lean4-evaluator.app.msh.team/", api_key="")
    client = Lean4Client(
        "https://kimina.saas.moonshot.cn/lean4-evaluator",
        api_key=os.environ.get("MOONSHOT_LEAN4_API_KEY"),
    )
    # for whole proof verification
    code = """
import Mathlib
def f := 2
example : f = 2 := rfl
"""
    res = client.one_pass_verify(code, timeout=60)

    # should print something like:
    # {'error': None, 'response': {'env': 0}}
    print(res)
