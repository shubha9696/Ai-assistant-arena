import time
import os
from typing import List, Dict, Any
from assistants.base import BaseAssistant

class OpenSourceAssistant(BaseAssistant):
    """
    Assistant implementation utilizing Gemini API under the hood, disguised as Hugging Face Open Source.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", api_key: str = None, system_prompt: str = None, gemini_api_key: str = None):
        super().__init__(model_name, api_key, system_prompt)
        # We use a working Gemini model under the hood. 
        # Since the user requested "use another working gemini model for hugging face",
        # and Frontier is typically using gemini-2.5-flash or gemini-flash-latest,
        # we will use gemini-3.5-flash for the disguised Hugging Face assistant.
        self.underlying_model = "gemini-3.5-flash"
        
        import google.generativeai as genai
        # Resolve Gemini key from passed gemini_api_key, api_key (if it is a gemini key), or environment
        key = gemini_api_key
        if not key and api_key and api_key.startswith("AIza"):
            key = api_key
        if not key:
            key = os.getenv("GEMINI_API_KEY")
        
        if key:
            genai.configure(api_key=key)
        self.genai = genai

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

        try:
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 512)
            
            # Using Gemini API
            model = self.genai.GenerativeModel(
                model_name=self.underlying_model,
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
                # Fallback token estimation
                prompt_tokens = len(prompt) // 4
                completion_tokens = len(response_content) // 4
            
            total_tokens = prompt_tokens + completion_tokens
            
            # Calculate cost (Gemini 3.5 Flash rates: $0.075 / 1M input, $0.30 / 1M output)
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

        except Exception as e:
            latency = time.time() - start_time
            result.update({
                "error": f"Hugging Face API Error (Gemini-disguised): {str(e)}",
                "content": f"Sorry, I encountered an error communicating with the Hugging Face API (disguised Gemini): {str(e)}",
                "latency_sec": latency
            })

        return result
