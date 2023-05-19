from __future__ import annotations

import openai

from autogpt.config import Config
from autogpt.llm.modelsinfo import COSTS
from autogpt.logs import logger
from autogpt.singleton import Singleton


class ApiManager(metaclass=Singleton):
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0.0

    def reset(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0.0

    def create_chat_completion(
        self,
        messages: list,
        model: str | None = None,
        temperature: float = None,
        max_tokens: int | None = None,
        deployment_id: str | None = None,
    ) -> str:
        """
        Create a chat completion and update the cost.
        Args:
            messages (list): The list of messages to send to the API.
            model (str): The model to use for the API call.
            temperature (float): The temperature to use for the API call.
            max_tokens (int): The maximum number of tokens for the API call.
            deployment_id (str): The deployment ID for the API call.
        Returns:
            str: The AI's response.
        """
        cfg = Config()
        if temperature is None:
            temperature = cfg.temperature
        if deployment_id is not None:
            response = openai.ChatCompletion.create(
                deployment_id=deployment_id,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=cfg.openai_api_key,
            )
        else:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=cfg.openai_api_key,
            )
        if hasattr(response, "error"):
            if "This model's maximum context length" in response.error.message:
                self.handle_token_limit_error(response.error, messages, model)
            else:
                logger.error(f"API Error: {response.error}")
                return ""
        else:
            logger.debug(f"Response: {response}")
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            self.update_cost(prompt_tokens, completion_tokens, model)
        return response.choices[0].message["content"]

    def update_cost(self, prompt_tokens, completion_tokens, model):
        """
        Update the total cost, prompt tokens, and completion tokens.
        Args:
            prompt_tokens (int): The number of tokens used in the prompt.
            completion_tokens (int): The number of tokens used in the completion.
            model (str): The model used for the API call.
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += (
            prompt_tokens * COSTS[model]["prompt"]
            + completion_tokens * COSTS[model]["completion"]
        ) / 1000
        logger.debug(f"Total running cost: ${self.total_cost:.3f}")

    def set_total_budget(self, total_budget):
        """
        Sets the total user-defined budget for API calls.
        Args:
            total_budget (float): The total budget for API calls.
        """
        self.total_budget = total_budget

    def get_total_budget(self):
        """
        Get the total user-defined budget for API calls.
        Returns:
            float: The total budget for API calls.
        """
        return self.total_budget

    def get_total_cost(self):
        """
        Get the total cost of API calls.
        Returns:
            float: The total cost of API calls.
        """
        return self.total_cost

    def get_total_prompt_tokens(self):
        """
        Get the total number of prompt tokens.
        Returns:
            int: The total number of prompt tokens.
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        Get the total number of completion tokens.
        Returns:
            int: The total number of completion tokens.
        """
        return self.total_completion_tokens

    def handle_token_limit_error(self, error, messages, model):
        """
        Handle the error when the token limit is reached for a model.
        Args:
            error (openai.error.InvalidRequestError): The error response from the API.
            messages (list): The list of messages sent to the API.
            model (str): The model used for the API call.
        """
        cfg = Config()
        logger.warning(f"Token limit error: {error}")
        logger.warning("Attempting to truncate the conversation and retry...")
        num_tokens_to_remove = (
            self.total_prompt_tokens + self.total_completion_tokens - model.max_token_limit
        )
        if num_tokens_to_remove <= 0:
            logger.error("Token limit exceeded. Cannot truncate further.")
            return
        for i in range(len(messages) - 1, -1, -1):
            message = messages[i]
            if len(message["content"].split()) <= num_tokens_to_remove:
                num_tokens_to_remove -= len(message["content"].split())
                messages.pop(i)
            else:
                split_content = message["content"].split()
                num_tokens_to_remove = 0
                split_content = split_content[: len(split_content) - num_tokens_to_remove]
                message["content"] = " ".join(split_content)
                break
        logger.warning(f"Truncated conversation: {messages}")
        self.create_chat_completion(
            messages=messages,
            model=model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            deployment_id=model.deployment_id,
        )
