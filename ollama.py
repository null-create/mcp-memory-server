import json
import logging
from typing import Any

import requests

logging.basicConfig(level="INFO")
logger = logging.getLogger(__file__)


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:3b",
        history_limit: int = 10,
    ) -> None:
        """
        Initialize the Ollama client.

        Args:
            base_url: The base URL for the Ollama API (default: http://localhost:11434)
            model_name: Model to interact with
            history_limit: Total number of chat history entries to retain per session
        """
        self.base_url: str = base_url
        self.chat_endpoint: str = f"{base_url}/api/chat"
        self.chat_history: list[dict] = []
        self.history_limit: int = history_limit
        self.model_name: str = model_name

    def _prune_history(self) -> None:
        """
        Keeps the chat history within a set limit to avoid memory leaks
        """
        self.chat_history = self.chat_history[-self.history_limit :]

    def chat(self, user_input: str, stream: bool = False) -> str:
        """
        Send a chat message to Ollama and get a response.

        Args:
            user_input: The user's message/prompt
            stream: Whether to stream the response (default: True)

        Returns:
            The model's response as a string
        """
        try:
            current_message = {"role": "user", "content": user_input}

            response = requests.post(
                self.chat_endpoint,
                json={
                    "model": self.model_name,
                    "messages": [current_message] + self.chat_history,
                    "stream": stream,
                },
                timeout=60,
            )
            response.raise_for_status()

            content = ""
            if stream:
                content = self._handle_stream(response)
            else:
                result: dict = response.json()
                content: str = result["message"]["content"]

            self.chat_history.append(current_message)
            self.chat_history.append({"role": "assistant", "content": content})

            if len(self.chat_history) > self.history_limit:
                self._prune_history()

            return content

        except Exception as e:
            logger.error("Failed to communicate with ollama:", e)
            return f"Error communicating with Ollama: {e}"

    def _handle_stream(self, response: Any) -> str:
        """Handle streaming responses from Ollama"""
        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "message" in chunk:
                    content = chunk["message"].get("content", "")
                elif "response" in chunk:
                    content = chunk.get("response", "")
                else:
                    content = ""

                if content:
                    full_response += content

        return full_response