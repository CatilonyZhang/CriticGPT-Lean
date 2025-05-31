import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import openai
from loguru import logger
from transformers import AutoTokenizer


def filter_deepseek_output(
    choice: openai.types.completion.CompletionChoice,
) -> openai.types.completion.CompletionChoice:
    """
    Overview:
        Filter out the unnecessary fields in the output of DeepSeek-Prover.
    """
    logger.debug(f"Original generated tactics: {[choice.text]}")
    if not ("\n" in choice.text or "```" in choice.text):
        return choice

    in_comment = False
    in_multiline_comment = False
    first_tactic_tokens = []
    for i, token in enumerate(choice.logprobs.tokens):
        if token in ["--", " --"]:
            in_comment = True
        if token in ["/-", " /"]:
            in_multiline_comment = True

        if in_comment and token == "\n":
            in_comment = False
        if in_multiline_comment and token in ["-/", " -/", " -"]:
            in_multiline_comment = False

        if not in_comment and not in_multiline_comment and token in ["\n", "```"]:
            break
        first_tactic_tokens.append(token)
    choice.text = "".join(first_tactic_tokens)
    choice.logprobs.tokens = first_tactic_tokens
    choice.logprobs.token_logprobs = choice.logprobs.token_logprobs[
        : len(first_tactic_tokens)
    ]
    logger.debug(f"filtered generated tactics: {[choice.text]}")
    return choice


class TacticGenerator(ABC):
    """
    Overview:
        Base class for all tactic generators. A tactic generator takes a state and generates multiple tactic candidates.
    """

    @abstractmethod
    def generate(
        self, messages: str, num_samples: int, **kwargs
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError


class APITacticGenerator(TacticGenerator):
    def __init__(
        self,
        model_id: str = None,
        api_key: str = "EMPTY",
        base_url: str = None,
        tokenizer_path: str = None,
    ) -> None:
        """
        Overview:
            Initialize the API tactic generator.
        Arguments:
            - model_id (str): The ID of the language model.
            - api_key (str): The API key of the language model.
            - base_url (str): The base URL of the language model.
            - tokenizer_path (str): The path to the tokenizer.
        """
        if base_url is None:
            self.base_url = f"https://{model_id}.app.msh.team/v1/"
        else:
            self.base_url = base_url
        self.client = openai.OpenAI(base_url=self.base_url, api_key=api_key)
        self.model = model_id
        self.tokenizer = None
        self.tokenizer_path = tokenizer_path

        if self.tokenizer_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def build_prompt(
        self, messages: List[Dict[str, str]], prompt_channel: str = "original"
    ) -> str:
        """
        Overview:
            Build a prompt to feed into LLM.
        Arguments:
            - messages (List[Dict[str, str]]): The messages to build the prompt.
            - prompt_channel (str): The channel to build the prompt, default to "original".
        Returns:
            - prompt_text (str): The built prompt.
        """
        user_message = messages[0] if len(messages) == 1 else messages[1]
        assert user_message["role"] == "user"
        if prompt_channel == "original":
            prompt_text = user_message["content"]
        elif prompt_channel == "gptf":
            prompt_text = f"GOAL {user_message['content']} PROOFSTEP\n"
        else:
            raise ValueError(f"Unknown prompt channel: {prompt_channel}.")
        prompt_token_count = len(self.tokenizer.encode(prompt_text))
        return prompt_text, prompt_token_count

    def build_messages(
        self, user_content: str, sys_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Overview:
            Build a prompt to feed into LLM.
        Arguments:
            - user_content (str): The user content to build the prompt.
            - sys_prompt (str): The system prompt to build the prompt, default to None.
        Returns:
            - messages (List[Dict[str, str]]): The built messages.
        """
        if sys_prompt is None:
            messages = [
                {
                    "role": "user",
                    "content": user_content,
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": sys_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]
        return messages

    def get_response(
        self,
        messages: List[Dict[str, str]],
        n_samples: int = 1,
        max_retry: Optional[int] = None,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """
        Overview:
            Get response from LLM with chat messages.
        Arguments:
            - messages (List[Dict[str, str]]): The messages to get the response.
            - n_samples (int): The number of samples to get, default to 1.
            - max_retry (Optional[int]): The maximum number of retries, default to None.
            - kwargs: The keyword arguments to pass to the LLM.
        Returns:
            - List[Tuple[str, float]]: The generated tactics and their log-probabilities.
        """
        n_retry = 0
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    logprobs=True,
                    top_logprobs=1,
                    n=n_samples,
                    seed=42,
                    **kwargs,
                )
                break
            except Exception as e:
                logger.error(
                    f"Encountered exception:\n{e}\n" + f"when querying {self.base_url}."
                )
                time.sleep(1)
                n_retry += 1
                if max_retry is not None and n_retry >= max_retry:
                    break
                else:
                    logger.error(f"Retry for the {n_retry}-th time...")
        responses = []
        logprobs = []
        seen = set()
        for choice in completion.choices:
            if choice.message.content in seen:
                continue
            seen.add(choice.message.content)
            logprob = 0
            for token_logprob in choice.logprobs.content:
                logprob += token_logprob.logprob
            # logprob /= len(choice.logprobs.content)
            if logprob < -720:  # math domain error
                continue
            logprobs.append(logprob)
            responses.append(choice.message.content)
        return list(zip(logprobs, responses))

    def get_response_with_text_completion(
        self,
        prompt_text: str,
        n_samples: int = 1,
        max_retry: Optional[int] = None,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """
        Overview:
            Get response from LLM with text completion mode.
        Arguments:
            - prompt_text (str): The prompt to get the response.
            - n_samples (int): The number of samples to get, default to 1.
            - max_retry (Optional[int]): The maximum number of retries, default to None.
            - kwargs: The keyword arguments to pass to the LLM.
        Returns:
            - List[Tuple[str, float]]: The generated tactics and their log-probabilities.
        """
        n_retry = 0
        while True:
            try:
                completion = self.client.completions.create(
                    model=self.model,
                    prompt=prompt_text,
                    logprobs=True,
                    n=n_samples,
                    # seed=42,
                    **kwargs,
                )
                break
            except Exception as e:
                logger.error(
                    f"Encountered exception:\n{e}\n" + f"when querying {self.base_url}."
                )
                time.sleep(1)
                n_retry += 1
                if max_retry is not None and n_retry >= max_retry:
                    break
                else:
                    logger.error(f"Retry for the {n_retry}-th time...")
        responses = []
        logprobs = []
        seen = set()
        for choice in completion.choices:
            if choice.text in seen:
                continue
            seen.add(choice.text)
            logprobs.append(choice.logprobs.top_logprobs)
            responses.append(choice.text)
        return list(zip(responses, logprobs))

    def generate(
        self,
        messages: List[Dict[str, str]],
        num_samples: int,
        prompt_type: str = "chat",
        max_length: int = 2048,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """
        Overview:
            Generate deduplicated text `n_samples` times.
        Arguments:
            - messages (List[Dict[str, str]]): The messages to generate the tactics.
            - num_samples (int): The number of samples to generate.
            - prompt_type (str): The type of the prompt, default to "chat", choices are "chat" and "text".
            - max_length (int): The maximum length of the prompt, default to 2048.
            - kwargs: The keyword arguments to pass to the LLM.
        Returns:
            - List[Tuple[str, float]]: The generated tactics and their log-probabilities.
        """
        if prompt_type == "chat":
            responses = self.get_response(messages, n_samples=num_samples, **kwargs)
        elif prompt_type == "text":
            assert self.tokenizer is not None, (
                f"Tokenizer is {self.tokenizer}, to use text completion, "
                + "you must specify tokenizer_path when initializing LLMClinet"
            )
            prompt_text, prompt_length = self.build_prompt(messages)
            max_tokens = max_length - prompt_length - 2
            if max_tokens < 1024:
                return []
            responses = self.get_response_with_text_completion(
                prompt_text,
                n_samples=num_samples,
                max_tokens=max_tokens,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}.")
        return responses


if __name__ == "__main__":
    tg = APITacticGenerator(
        model_id="deepseekproverrl",
        api_key="EMPTY",
        tokenizer_path="/mnt/moonfs/wanghaiming-m2/models/deepseekprover/DeepSeek-Prover-V1.5-RL",
    )

    message = tg.build_messages(user_content="prove 1 + 1 = 2")
    responses = tg.generate(message, num_samples=1, prompt_type="text")
