import os
import sys
import time
import re
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure the root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load env variables
load_dotenv()

from assistants.oss_assistant import OpenSourceAssistant
from assistants.frontier_assistant import FrontierAssistant
from guardrails.safety_filter import SafetyGuardrail
from tools.search_tool import AssistantTools
from utils.observability import ObservabilityLogger

app = FastAPI(title="AI Assistant Arena API", version="1.0")

# CORS middleware to allow the frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]]
    oss_model: str = "Qwen/Qwen2.5-7B-Instruct"
    frontier_model: str = "gemini-flash-latest"
    system_prompt: str = "You are a helpful assistant."
    enable_guardrails: bool = True
    enable_tools: bool = True
    temperature: float = 0.7
    max_tokens: int = 512
    hf_token: Optional[str] = None
    gemini_key: Optional[str] = None
    openai_key: Optional[str] = None

@app.get("/api/results")
def get_evaluation_results():
    """
    Returns the pre-computed evaluation results.
    """
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation", "precomputed_results.json")
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Precomputed evaluation results not found.")
    
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation results: {str(e)}")

@app.get("/api/stats")
def get_observability_stats():
    """
    Returns aggregated metrics from live usage logs.
    """
    try:
        return ObservabilityLogger.get_aggregated_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
def get_observability_logs():
    """
    Returns recent live usage logs.
    """
    try:
        return ObservabilityLogger.get_logs()[-50:]  # Return last 50 logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/logs/clear")
def clear_observability_logs():
    """
    Clears the observability logs file.
    """
    try:
        ObservabilityLogger.clear_logs()
        return {"status": "success", "message": "Logs cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def run_chat_arena(req: ChatRequest):
    """
    Runs a single query through the dual assistant arena, implementing agentic tools,
    guardrails, observability, and cost estimation.
    """
    user_prompt = req.prompt
    
    # 1. Input Guardrail Check
    guardrail = SafetyGuardrail(enable_keyword_check=req.enable_guardrails)
    input_safe, input_reason = guardrail.check_input(user_prompt)
    
    # 2. Tool Routing (Agentic Behavior)
    tool_status = {"triggered": False, "name": "", "query": "", "result": ""}
    augmented_prompt = user_prompt
    
    if req.enable_tools and input_safe:
        # Check for calculator triggers
        calc_match = re.search(r"(?:calculate|what is)\s+([0-9\+\-\*\/\(\)\.\s]+)", user_prompt, re.IGNORECASE)
        if calc_match:
            expr = calc_match.group(1).strip()
            result = AssistantTools.calculate(expr)
            tool_status = {"triggered": True, "name": "calculate", "query": expr, "result": result}
            augmented_prompt = f"Using this calculator tool result:\n{result}\n\nAnswer the user request: {user_prompt}"
        # Check for search triggers
        elif any(kw in user_prompt.lower() for kw in ["search", "who is", "capital", "distance", "discover", "history"]):
            result = AssistantTools.web_search(user_prompt)
            tool_status = {"triggered": True, "name": "web_search", "query": user_prompt, "result": result}
            augmented_prompt = f"Using this web search tool result:\n{result}\n\nAnswer the user request: {user_prompt}"
            
    # 3. Format history for memory
    oss_memory = []
    frontier_memory = []
    # Take last 5 turns
    for past in req.history[-5:]:
        # history elements should be dict with format {"user": "...", "oss": "...", "frontier": "..."}
        # or list of standard openai roles. We adapt to the dashboard's representation:
        # if keys are "role" and "content", adapt accordingly.
        # Otherwise, assume format {"user": "...", "oss": "...", "frontier": "..."}
        if "role" in past and "content" in past:
            # simple single sequence
            pass
        else:
            oss_memory.append({"role": "user", "content": past.get("user", "")})
            oss_memory.append({"role": "assistant", "content": past.get("oss", "")})
            frontier_memory.append({"role": "user", "content": past.get("user", "")})
            frontier_memory.append({"role": "assistant", "content": past.get("frontier", "")})

    # Resolve keys
    hf_token = req.hf_token or os.getenv("HF_TOKEN")
    is_gemini = "gemini" in req.frontier_model.lower()
    frontier_key = req.gemini_key if is_gemini else req.openai_key
    if not frontier_key:
        frontier_key = os.getenv("GEMINI_API_KEY") if is_gemini else os.getenv("OPENAI_API_KEY")
        
    # Instantiate models
    oss_assistant = OpenSourceAssistant(model_name=req.oss_model, api_key=hf_token, system_prompt=req.system_prompt)
    frontier_assistant = FrontierAssistant(model_name=req.frontier_model, api_key=frontier_key, system_prompt=req.system_prompt)
    
    # 4. Generate Responses & Apply Output Guardrails
    if not input_safe:
        oss_response = {"content": "Blocked", "latency_sec": 0.0, "total_tokens": 0, "cost_usd": 0.0}
        oss_guard = {"triggered": True, "reason": input_reason}
        
        frontier_response = {"content": "Blocked", "latency_sec": 0.0, "total_tokens": 0, "cost_usd": 0.0}
        frontier_guard = {"triggered": True, "reason": input_reason}
    else:
        # OSS Model response
        try:
            oss_response = oss_assistant.generate_response(augmented_prompt, history=oss_memory, temperature=req.temperature, max_tokens=req.max_tokens)
            out_safe, out_reason = guardrail.check_output(oss_response.get("content", ""))
            oss_guard = {"triggered": not out_safe, "reason": out_reason}
            if not out_safe:
                oss_response["content"] = f"⚠️ Guardrail Blocked Output: {out_reason}"
        except Exception as e:
            oss_response = {"content": f"Error: {str(e)}", "latency_sec": 0.0, "total_tokens": 0, "cost_usd": 0.0, "error": str(e)}
            oss_guard = {"triggered": False, "reason": ""}
            
        # Frontier Model response
        try:
            frontier_response = frontier_assistant.generate_response(augmented_prompt, history=frontier_memory, temperature=req.temperature, max_tokens=req.max_tokens)
            out_safe, out_reason = guardrail.check_output(frontier_response.get("content", ""))
            frontier_guard = {"triggered": not out_safe, "reason": out_reason}
            if not out_safe:
                frontier_response["content"] = f"⚠️ Guardrail Blocked Output: {out_reason}"
        except Exception as e:
            frontier_response = {"content": f"Error: {str(e)}", "latency_sec": 0.0, "total_tokens": 0, "cost_usd": 0.0, "error": str(e)}
            frontier_guard = {"triggered": False, "reason": ""}

    # 5. Log interactions
    try:
        ObservabilityLogger.log_interaction(
            provider="huggingface", model_name=req.oss_model, prompt=user_prompt,
            response=oss_response, guardrail_status=oss_guard
        )
        ObservabilityLogger.log_interaction(
            provider=frontier_assistant.provider, model_name=req.frontier_model, prompt=user_prompt,
            response=frontier_response, guardrail_status=frontier_guard
        )
    except Exception as log_error:
        print(f"Logging error: {str(log_error)}")

    return {
        "prompt": user_prompt,
        "tool_status": tool_status,
        "oss_response": {
            "content": oss_response.get("content", ""),
            "latency_sec": oss_response.get("latency_sec", 0.0),
            "total_tokens": oss_response.get("total_tokens", 0),
            "cost_usd": oss_response.get("cost_usd", 0.0),
            "guardrail_triggered": oss_guard["triggered"],
            "guardrail_reason": oss_guard["reason"]
        },
        "frontier_response": {
            "content": frontier_response.get("content", ""),
            "latency_sec": frontier_response.get("latency_sec", 0.0),
            "total_tokens": frontier_response.get("total_tokens", 0),
            "cost_usd": frontier_response.get("cost_usd", 0.0),
            "guardrail_triggered": frontier_guard["triggered"],
            "guardrail_reason": frontier_guard["reason"]
        }
    }
