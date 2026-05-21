import modal
import os

# Modal configuration for running an open-source LLM (like Qwen 2.5 7B) using vLLM
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("vllm", "huggingface_hub")
)

app = modal.App("qwen-assistant-deployment")

@app.cls(
    image=image,
    gpu="A100",  # Requesting A100 GPU for fast inference
    container_idle_timeout=300,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
class QwenModel:
    def __enter__(self):
        from vllm import LLMEngine, EngineArgs, SamplingParams
        from vllm.entrypoints.openai.api_server import OpenAIAPIServer
        
        # Initialize the vLLM engine with Qwen 2.5 7B Instruct
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        print(f"Loading model: {model_name}")
        
        # We initialize vLLM engine args
        engine_args = EngineArgs(
            model=model_name,
            gpu_memory_utilization=0.90,
            max_model_len=4096
        )
        self.engine = LLMEngine.from_engine_args(engine_args)
        
    @modal.method()
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        from vllm import SamplingParams
        
        formatted_prompt = f"<|im_start|>system\n{system_prompt or 'You are a helpful assistant.'}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop=["<|im_end|>"]
        )
        
        # Request generation
        import uuid
        request_id = str(uuid.uuid4())
        self.engine.add_request(request_id, formatted_prompt, sampling_params)
        
        output_text = ""
        while self.engine.has_unfinished_requests():
            request_outputs = self.engine.step()
            for request_output in request_outputs:
                if request_output.request_id == request_id:
                    output_text = request_output.outputs[0].text
                    
        return output_text

# To deploy this API, run:
# modal deploy modal_deploy.py
