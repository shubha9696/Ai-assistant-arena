import time
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
from assistants.base import BaseAssistant

class OpenSourceAssistant(BaseAssistant):
    """
    Assistant implementation utilizing open-source models hosted on Hugging Face Serverless API.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", api_key: str = None, system_prompt: str = None):
        super().__init__(model_name, api_key, system_prompt)
        # Use Hugging Face serverless client
        self.client = InferenceClient(model=self.model_name, token=self.api_key)

    def generate_response(self, prompt: str, history: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        
        # Prepare messages including system prompt and history
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        result = {
            "content": "",
            "latency_sec": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "error": None
        }

        try:
            # Call HF Serverless API using chat completion interface
            # Set a reasonable max_tokens and temperature
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 512)
            
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            latency = time.time() - start_time
            response_content = response.choices[0].message.content
            
            # Extract token details if available
            prompt_tokens = response.usage.prompt_tokens if (response.usage and hasattr(response.usage, 'prompt_tokens')) else len(str(messages)) // 4
            completion_tokens = response.usage.completion_tokens if (response.usage and hasattr(response.usage, 'completion_tokens')) else len(response_content) // 4
            total_tokens = prompt_tokens + completion_tokens

            # Calculate cost (estimating HF dedicated endpoint / Serverless API rates: Qwen 2.5 7B is ~$0.15/M input, $0.15/M output)
            cost_rate_input = 0.15 / 1000000  # $0.15 per million
            cost_rate_output = 0.15 / 1000000
            cost_usd = (prompt_tokens * cost_rate_input) + (completion_tokens * cost_rate_output)

            result.update({
                "content": response_content,
                "latency_sec": latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd
            })

        except Exception as e:
            latency = time.time() - start_time
            result.update({
                "error": f"Hugging Face API Error: {str(e)}",
                "content": f"Sorry, I encountered an error communicating with the Hugging Face API: {str(e)}",
                "latency_sec": latency
            })

        return result
