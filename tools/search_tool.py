import urllib.parse
import json
import requests
import re
from typing import Dict, Any

class AssistantTools:
    """
    Implements tools that can be invoked by the assistant or the evaluation harness.
    """
    
    @staticmethod
    def calculate(expression: str) -> str:
        """
        Safely evaluates a mathematical expression.
        """
        # Clean expression
        clean_expr = re.sub(r"[^0-9\+\-\*\/\(\)\.\s]", "", expression)
        try:
            # Evaluate clean mathematical expression
            if not clean_expr.strip():
                return "Error: Empty expression"
            # Prevent division by zero or code execution exploits
            result = eval(clean_expr, {"__builtins__": None}, {})
            return f"Result: {result}"
        except ZeroDivisionError:
            return "Error: Division by zero is undefined."
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    @staticmethod
    def web_search(query: str) -> str:
        """
        Performs a search using a public API or fallbacks to mock results for stability.
        """
        # Try a quick DuckDuckGo API request (lite/api search)
        # DuckDuckGo has a simple Instant Answer API
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                abstract = data.get("AbstractText", "")
                if abstract:
                    return f"Search result for '{query}': {abstract}"
                
                # Try related topics
                related = data.get("RelatedTopics", [])
                if related:
                    text = related[0].get("Text", "")
                    if text:
                        return f"Search result for '{query}': {text}"
        except Exception:
            pass

        # Fallback dictionary of common facts for evaluating hallucination and factual queries
        factual_kb = {
            "capital of France": "Paris is the capital and most populous city of France.",
            "distance from Earth to Moon": "The average distance from the Earth to the Moon is about 384,400 km (238,855 miles).",
            "who wrote Hamlet": "Hamlet was written by William Shakespeare between 1599 and 1601.",
            "speed of light": "The speed of light in a vacuum is exactly 299,792,458 meters per second (approx. 300,000 km/s or 186,000 miles/s).",
            "largest ocean": "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
            "who discovered gravity": "Sir Isaac Newton is widely recognized for developing the law of universal gravitation in the 17th century."
        }

        query_lower = query.lower()
        for key, value in factual_kb.items():
            if key in query_lower:
                return f"Factual Database: {value}"

        return f"Search result for '{query}': No highly relevant results found. (Using offline fallback)"
        
    @classmethod
    def execute_tool(cls, tool_name: str, arguments: str) -> str:
        """
        Executes a tool by name with arguments.
        """
        if tool_name == "calculate":
            return cls.calculate(arguments)
        elif tool_name == "web_search":
            return cls.web_search(arguments)
        else:
            return f"Error: Tool '{tool_name}' is not recognized."
