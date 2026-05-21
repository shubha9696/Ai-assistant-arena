import json
import os
import time
import pandas as pd
from typing import Dict, Any

class ObservabilityLogger:
    """
    Handles logging and monitoring of LLM requests, responses, latencies, and guardrail alerts.
    """
    LOG_FILE = "logs/observability.jsonl"

    @classmethod
    def log_interaction(cls, provider: str, model_name: str, prompt: str, response: Dict[str, Any], guardrail_status: Dict[str, Any]):
        """
        Logs a single interaction to a JSONL file.
        """
        # Ensure log directory exists
        os.makedirs(os.path.dirname(cls.LOG_FILE), exist_ok=True)
        
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provider": provider,
            "model_name": model_name,
            "prompt_length": len(prompt),
            "response_length": len(response.get("content", "")),
            "latency_sec": response.get("latency_sec", 0.0),
            "tokens_used": response.get("total_tokens", 0),
            "cost_usd": response.get("cost_usd", 0.0),
            "guardrail_triggered": guardrail_status.get("triggered", False),
            "guardrail_reason": guardrail_status.get("reason", ""),
            "has_error": response.get("error") is not None
        }

        with open(cls.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    @classmethod
    def get_logs(cls) -> list:
        """
        Reads and returns all logged interactions.
        """
        if not os.path.exists(cls.LOG_FILE):
            return []
        
        logs = []
        with open(cls.LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return logs
        
    @classmethod
    def clear_logs(cls):
        """
        Clears all logs.
        """
        if os.path.exists(cls.LOG_FILE):
            os.remove(cls.LOG_FILE)
            
    @classmethod
    def get_aggregated_stats(cls) -> Dict[str, Any]:
        """
        Aggregates logs into metrics for the dashboard.
        """
        logs = cls.get_logs()
        if not logs:
            return {}
            
        df = pd.DataFrame(logs) if len(logs) > 0 else None
        if df is None or df.empty:
            return {}
            
        stats = {
            "total_requests": len(df),
            "avg_latency": df["latency_sec"].mean(),
            "total_cost": df["cost_usd"].sum(),
            "guardrail_triggers": df["guardrail_triggered"].sum(),
            "error_count": df["has_error"].sum()
        }
        return stats
