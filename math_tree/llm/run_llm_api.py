import requests
import json
from openai import OpenAI
import base64
from typing import Union, Tuple
import pdb
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *




def run_llm(
    model_name: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    llm_url: str = None,
):
    llm_client = OpenAI(
        base_url=llm_url + '/v1',
        api_key="EMPTY"
    )

    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def run_llm_openai(
    model_name: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 8192,
):
    llm_client = OpenAI(
        base_url=openai_api_base,
        api_key=openai_api_key
    )
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    model_name = 'gpt-4o'
    math_problem = 'Prove that the sum of the first n odd numbers is n^2'
    print(run_llm_openai(model_name, math_problem))

