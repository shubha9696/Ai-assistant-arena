# 🚀 Public OSS Model Deployment Options

This directory contains code and documentation for deploying the Open Source Assistant model (`Qwen2.5-7B-Instruct` or similar) publicly.

## 📊 Deployment Cost & Latency Comparison Table

The table below outlines options for deploying the OSS model publicly, detailing hardware requirements, average latency, and operational cost.

| Platform | Deployment Type | Hardware / Spec | Avg Latency (s) | Cost Estimate | Best For |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Hugging Face Spaces (Free)** | Serverless API (CPU) | Shared Serverless | 1.0 - 2.5s | **$0.00 / month** | Prototyping, zero-cost demos, light usage. |
| **Hugging Face Spaces (Paid)** | Dedicated Space | Nvidia T4 GPU (16GB VRAM) | 0.8 - 1.5s | **~$73.00 / month** ($0.10/hr) | Production demos requiring guaranteed uptime. |
| **Modal** | Serverless vLLM Container | Nvidia A100 GPU (40GB VRAM) | 0.2 - 0.5s | **~$0.0014 / call** (Pay-as-you-go: $0.00028/sec) | Scaling to high volumes with rapid cold-starts. |
| **RunPod / Vast.ai** | Persistent VM | 1x RTX 4090 GPU (24GB VRAM) | 0.3 - 0.6s | **~$0.30 - $0.45 / hour** (~$210 - $320/month) | Dedicated, long-running deployments, high traffic. |
| **Replicate / DeepInfra** | Hosted API Endpoint | Serverless API | 0.3 - 0.7s | **~$0.15 - $0.20 / 1M tokens** | Zero-infra operations, API-only applications. |

---

## 🛠️ Step-by-Step Deployment Instructions

### 1. Serverless API Deployment (Hugging Face Spaces - Free CPU)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Choose **Streamlit** as the SDK.
3. Upload `deployment/space_app.py` and rename it to `app.py`.
4. Create a `requirements.txt` file in the root containing:
   ```text
   streamlit>=1.35.0
   huggingface_hub>=0.23.0
   ```
5. (Optional) Add your `HF_TOKEN` in the Space settings (Variables & Secrets) for higher rate limits.

### 2. Serverless GPU Deployment (Modal - Paid Pay-as-you-go)
1. Install Modal: `pip install modal`
2. Authenticate: `modal setup`
3. Run `modal deploy deployment/modal_deploy.py` to deploy the app.
4. Integrate the generated endpoint URL with your Streamlit UI or web application.
5. In Modal, you only pay for the exact milliseconds the GPU compiles and runs your query, with scale-to-zero capabilities.

### 3. Dedicated GPU VM (RunPod / Ollama)
1. Rent an RTX 3090/4090 instance.
2. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
3. Pull the model:
   ```bash
   ollama pull qwen2.5:7b
   ```
4. Expose Ollama's port (`11434`) to your application.
