from abc import ABC, abstractmethod
import time
from typing import List, Dict, Any

class BaseAssistant(ABC):
    """
    Base class for all assistant implementations.
    Defines the interface for interacting with models and managing memory.
    """
    def __init__(self, model_name: str, api_key: str = None, system_prompt: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.system_prompt = system_prompt or (
            "You are a helpful, respectful, and safe personal assistant. "
            "Provide clear, accurate, and concise answers."
        )

    @abstractmethod
    def generate_response(self, prompt: str, history: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Generates a response given a user prompt and conversation history.
        
        Args:
            prompt: The current user message.
            history: List of past messages in format [{'role': 'user'/'assistant', 'content': '...'}]
            
        Returns:
            Dict containing:
                - 'content': str (the generated response text)
                - 'latency_sec': float (time taken for the API call)
                - 'prompt_tokens': int (tokens in the prompt)
                - 'completion_tokens': int (tokens in the response)
                - 'total_tokens': int
                - 'cost_usd': float (estimated cost of the call)
                - 'error': str or None (if any error occurred)
        """
        pass
