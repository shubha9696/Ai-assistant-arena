# ⚖️ Dual AI Assistant Arena & Safety Evaluation Framework

This repository contains a complete evaluation framework and an interactive **Streamlit Chat Arena** comparing an **Open Source Assistant** (using Hugging Face Serverless API) against a **Hosted Frontier Assistant** (supporting Gemini and OpenAI).

The assistants support:
- 💬 **Multi-turn conversation history** (short-term conversational memory).
- 🛠️ **Agentic Tool Use** (automatic routing to Math Calculator and Web Search APIs).
- 🛡️ **Dual-Layer Guardrails** (rule-based input filters and semantic output validation).
- 📈 **Real-Time Observability Logs** (tracking tokens, latencies, guardrail blocks, and costs).

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.9+
- A Hugging Face Serverless API key (free/paid).
- Either a Google Gemini API Key or an OpenAI API Key.

### 2. Setup Instructions
Clone this repository and set up a virtual environment:
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements-local.txt
```

### 3. Set up Environment Variables
Copy `.env.example` to `.env` and fill in your API tokens:
```bash
cp .env.example .env
```

### 4. Running the Dashboard
Launch the interactive Streamlit Chat Arena dashboard:
```bash
streamlit run app.py
```

### 5. Running the CLI Evaluation Report Generator
To generate evaluation reports directly from the terminal:
```bash
python run_eval.py
```
This generates the Markdown and PDF reports in the [artifacts/](file:///c:/Users/shubh/Desktop/New%20folder%20%284%29/artifacts/) directory.

---

## 🏗️ Architecture Decisions & Flow

The system flows through a pipelined architecture designed to safely sanitize, expand, query, and evaluate interactions.

```
       +---------------------------------------------+
       |                  User Input                 |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |        Input Guardrail (Keyword Filter)     |
       +-------+-----------------------------+-------+
               |                             |
         (Safe | Triggered)                  | (Unsafe Triggered)
               v                             v
       +---------------+             +-----------------------+
       | Agent Router  |             | Refusal Response      |
       +-------+-------+             +-----------------------+
               |
      +--------+--------+
      |                 |
      v                 v
+-----------+     +------------+
| Search    |     | Calculator |
+-----+-----+     +-----+------+
      |                 |
      +--------+--------+
               |
               v
       +---------------------------------------------+
       |              Augmented Context              |
       +----------------------+----------------------+
                              |
               +--------------+--------------+
               |                             |
               v                             v
       +---------------+             +---------------+
       |  Open Source  |             |   Frontier    |
       |   Assistant   |             |   Assistant   |
       +-------+-------+             +-------+-------+
               |                             |
               v                             v
       +---------------+             +---------------+
       | Output Filter |             | Output Filter |
       +-------+-------+             +-------+-------+
               |                             |
               +--------------+--------------+
                              |
                              v
       +---------------------------------------------+
       |         Observability Logs & Grader         |
       +---------------------------------------------+
```

### Key Modules:
- **[assistants/](file:///c:/Users/shubh/Desktop/New%20folder%20%284%29/assistants/)**: Contains `base.py`, `oss_assistant.py` (HF Inference client), and `frontier_assistant.py` (Gemini & OpenAI).
- **[guardrails/](file:///c:/Users/shubh/Desktop/New%20folder%20%284%29/guardrails/)**: Rule-based regex pattern checking to detect SQL injections, DAN jailbreaks, self-harm, and slurs.
- **[tools/](file:///c:/Users/shubh/Desktop/New%20folder%20%284%29/tools/)**: Code containing math solver and DuckDuckGo/Offline search modules.
- **[evaluation/](file:///c:/Users/shubh/Desktop/New%20folder%20%284%29/evaluation/)**: Hosts test prompts, automated LLM-as-judge / Rule-based evaluators, and PDF/MD generators.
- **[utils/](file:///c:/Users/shubh/Desktop/New%20folder%20%284%29/utils/)**: Custom JSONL observability database manager.

---

## ⚖️ Trade-offs Made

1. **Hugging Face Serverless Client vs Local GPU Execution**
   - *Pros:* Zero system dependencies, zero model download sizes, runs on weak CPU laptops, immediate startup.
   - *Cons:* Prone to cold-start delay on first request, subject to Hugging Face server rate-limiting.
2. **Deterministic Rules vs LLM Judges**
   - *Pros:* Rule-based regexes (keyword checkers) execute in <1ms and cost $0.00.
   - *Cons:* Regexes are fragile to semantic variations and create false positives (e.g., blocking translation requests featuring sensitive words). The optional LLM judge solves this but adds latency and API costs.
3. **Pre-computed Benchmark Dataset**
   - *Pros:* Provides instant interactive data visualizations in the Streamlit UI without requiring API credentials.
   - *Cons:* Pre-computed logs represent static evaluations of Qwen 2.5 7B and Gemini 1.5 Flash. Live runs overwrite this with real-time API responses.

---

## 🔮 What I Would Improve With More Time

1. **Structured Tool Calls:** Integrate Native function-calling JSON schemas for Qwen and Gemini rather than text-based regex classification.
2. **Retrieval-Augmented Generation (RAG):** Connect a vector DB (e.g. ChromaDB or FAISS) to load local documentation as external memory.
3. **DeepGuard Integration:** Connect LLamaGuard or NeMo Guardrails for enterprise-grade policy enforcement instead of standard keyword filtering.
4. **Asynchronous Generation:** Implement python `asyncio` or Threading to run both models simultaneously in the Chat Arena, cutting overall response wait times in half.
