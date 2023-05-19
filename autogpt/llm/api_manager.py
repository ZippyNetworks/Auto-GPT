from __future__ import annotations

import openai

from autogpt.config import Config
from autogpt.llm.modelsinfo import COSTS
from autogpt.logs import logger
from autogpt.singleton import Singleton

model_token_limit = {
    "gpt-3.5-turbo": 4096,
    # Add more models and their respective token limits if needed
}


def handle_token_limit_error(messages, model, max_tokens):
    """
    Handles the token limit error by splitting the messages into smaller chunks.

    Args:
        messages (list): The list of messages.
        model (str): The model used for the API call.
        max_tokens (int): The maximum number of tokens for the API call.

    Returns:
        list: List of responses for each message chunk.
    """
    # Calculate the maximum tokens per message based on the model's context length limit
    max_context_tokens = model_token_limit[model] - len(messages[0]["role"]) - len(messages[0]["content"]) - 1
    max_tokens_per_message = max_context_tokens + max_tokens - 1

    # Split the messages into smaller chunks
    message_chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    for message in messages:
        message_tokens = len(message["role"]) + len(message["content"]) + 1

        if current_chunk_tokens + message_tokens <= max_tokens_per_message:
            current_chunk.append(message)
            current_chunk_tokens += message_tokens
        else:
            message_chunks.append(current_chunk)
            current_chunk = [message]
            current_chunk_tokens = message_tokens

    if current_chunk:
        message_chunks.append(current_chunk)

    # Make API calls for each message chunk
    responses = []
    for chunk in message_chunks:
        response = api_manager.create_chat_completion(chunk, model=model, max_tokens=max_tokens)
        responses.append(response)

    return responses


class ApiManager(metaclass=Singleton):
    def __init__(self):
        # Initialize total counts and cost
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0

    def reset(self):
        # Reset total counts and cost
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
        deployment_id=None,
    ) -> str:
        """
        Create a chat completion and update the cost.

        Args:
            messages (list): The list of messages to send to the API.
            model (str): The model to use for the API call.
            temperature (float): The temperature to use for the API call.
            max_tokens (int): The maximum number of tokens for the API call.
            deployment_id: The deployment ID for Chat models.

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
            error_message = response.error.message
            if "token limit" in error_message.lower():
                logger.info("Received token limit error. Attempting to handle the error.")

                # Handle the token limit error
                responses = handle_token_limit_error(messages, model, max_tokens)

                # Combine the responses into a single response
                combined_response = {
                    "id": response.id,
                    "object": response.object,
                    "created": response.created,
                    "model": response.model,
                    "usage": response.usage,
                    "choices": [],
                }

                for resp in responses:
                    combined_response["choices"].extend(resp.choices)

                prompt_tokens = combined_response["usage"]["prompt_tokens"]
                completion_tokens = combined_response["usage"]["completion_tokens"]
                self.update_cost(prompt_tokens, completion_tokens, model)

                return combined_response

            logger.error(f"API error: {response.error}")

        logger.debug(f"Response: {response}")
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        self.update_cost(prompt_tokens, completion_tokens, model)

        return response

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

    def get_total_cost(self):
        """
        Get the total cost of API calls.

        Returns:
            float: The total cost of API calls.
        """
        return self.total_cost

    def get_total_budget(self):
        """
        Get the total user-defined budget for API calls.

        Returns:
            float: The total budget for API calls.
        """
        return self.total_budget
