import time
from typing import List, Dict, Any
import os
from assistants.base import BaseAssistant

class FrontierAssistant(BaseAssistant):
    """
    Assistant implementation utilizing hosted frontier models from Google Gemini or OpenAI.
    """
    def __init__(self, model_name: str = "gemini-flash-latest", api_key: str = None, system_prompt: str = None):
        super().__init__(model_name, api_key, system_prompt)
        self.provider = "google" if "gemini" in model_name.lower() else "openai"
        self.client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "google":
            import google.generativeai as genai
            key = self.api_key or os.getenv("GEMINI_API_KEY")
            if key:
                genai.configure(api_key=key)
            self.genai = genai
        else:
            from openai import OpenAI
            key = self.api_key or os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=key)

    def generate_response(self, prompt: str, history: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        result = {
            "content": "",
            "latency_sec": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "error": None
        }

        # Prepare messages
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        try:
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 512)

            if self.provider == "google":
                # Using Gemini API
                # Gemini doesn't use system prompt as a standard role in older versions, 
                # but google-generativeai allows system_instruction or we can structure the chat.
                # In google-generativeai, GenerativeModel takes system_instruction at creation.
                model = self.genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=self.system_prompt
                )
                
                # Format history for Gemini
                gemini_history = []
                for msg in history:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_history.append({
                        "role": role,
                        "parts": [msg["content"]]
                    })
                
                chat = model.start_chat(history=gemini_history)
                response = chat.send_message(
                    prompt,
                    generation_config=self.genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
                
                latency = time.time() - start_time
                response_content = response.text
                
                # Estimate tokens for Gemini (or call response.usage_metadata if available)
                try:
                    prompt_tokens = response.usage_metadata.prompt_token_count
                    completion_tokens = response.usage_metadata.candidates_token_count
                except AttributeError:
                    prompt_tokens = len(str(messages)) // 4
                    completion_tokens = len(response_content) // 4
                
                total_tokens = prompt_tokens + completion_tokens
                
                # Calculate cost (Gemini 1.5 Flash rates: $0.075 / 1M input, $0.30 / 1M output)
                if "pro" in self.model_name:
                    cost_rate_input = 1.25 / 1000000
                    cost_rate_output = 5.00 / 1000000
                else:
                    cost_rate_input = 0.075 / 1000000
                    cost_rate_output = 0.30 / 1000000
                
                cost_usd = (prompt_tokens * cost_rate_input) + (completion_tokens * cost_rate_output)

                result.update({
                    "content": response_content,
                    "latency_sec": latency,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost_usd": cost_usd
                })

            else:
                # Using OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                latency = time.time() - start_time
                response_content = response.choices[0].message.content
                
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = prompt_tokens + completion_tokens

                # Estimate cost based on standard OpenAI models
                if "gpt-4o-mini" in self.model_name:
                    cost_rate_input = 0.150 / 1000000
                    cost_rate_output = 0.600 / 1000000
                elif "gpt-4o" in self.model_name:
                    cost_rate_input = 2.50 / 1000000
                    cost_rate_output = 10.00 / 1000000
                else:
                    cost_rate_input = 0.50 / 1000000
                    cost_rate_output = 1.50 / 1000000
                    
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
                "error": f"Frontier API Error ({self.provider}): {str(e)}",
                "content": f"Sorry, I encountered an error communicating with the {self.provider} API: {str(e)}",
                "latency_sec": latency
            })

        return result
