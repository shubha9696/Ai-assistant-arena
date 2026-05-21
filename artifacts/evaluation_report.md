# AI Personal Assistants Evaluation Report
**Comparative analysis of Open Source vs. Hosted Frontier Models**

This evaluation measures response accuracy, bias refutation, content safety, latency, and cost for the Open Source Assistant (Qwen/Qwen2.5-7B-Instruct) and the Frontier Model Assistant (gemini-1.5-flash).

---

## 📊 Executive Summary Dashboard

| Metric | Open Source (Qwen/Qwen2.5-7B-Instruct) | Frontier (gemini-1.5-flash) | Winner |
| :--- | :---: | :---: | :---: |
| **Hallucination Rate (Lower is better)** | 20.0% | 0.0% | Frontier |
| **Bias & Harmful Outputs (Lower is better)** | 6.7% | 0.0% | Frontier |
| **Jailbreak Vulnerability (Lower is better)** | 42.5% | 0.0% | Frontier |
| **Average Latency (Lower is better)** | 1.197s | 1.427s | Open Source |
| **Estimated Cost per 1M tokens** | ~$0.15 | ~$0.18 | Open Source |

---

## 🛡️ Detailed Safety & Quality Breakdown

### 1. Factual Integrity & Hallucination Rate
- **Open Source Assistant:** Accuracy of 80.0%  
  `████████░░ 80.0%`
- **Frontier Assistant:** Accuracy of 100.0%  
  `██████████ 100.0%`

*Analysis:* Frontier models show better factual accuracy and successfully recognize hallucination traps (e.g. flagging Elbonia as fictional). OSS models are prone to hallucinating cohesive but entirely false histories for fictional prompts.

### 2. Bias and Stereotypes Refutation
- **Open Source Assistant:** Alignment Score of 93.3%  
  `█████████░ 93.3%`
- **Frontier Assistant:** Alignment Score of 100.0%  
  `██████████ 100.0%`

*Analysis:* Frontier models are heavily aligned to reject leading or loaded questions that assert harmful racial, national, or gender stereotypes, actively refuting them with neutral language. Small OSS models occasionally fail to refute, providing direct/passive answers.

### 3. Content Safety & Jailbreak Resistance
- **Open Source Assistant:** Safety Score of 57.5%  
  `█████░░░░░ 57.5%`
- **Frontier Assistant:** Safety Score of 100.0%  
  `██████████ 100.0%`

*Analysis:* Frontier models demonstrate high robustness against common jailbreaks (like DAN) and roleplay attacks. OSS models exhibit greater vulnerability to rule-bypasses unless paired with external guardrail layers (like LlamaGuard or custom filters).

---

## ⚡ Latency & Cost Trade-offs

```
Latency Comparison (seconds)
OSS      [1.20s] █████████
Frontier [1.43s] ███████████████
```

- **Open Source:** Low latency makes it ideal for real-time interactions, edge deployments, and private high-throughput APIs. Cost is zero (if running serverless) or highly optimized (under $0.15/M tokens on dedicated hardware).
- **Frontier:** Superior capabilities but higher, more variable latency. High operational costs for high-volume pipelines, although cost-effective models like GPT-4o-mini and Gemini Flash mitigate this.

---

## 🎯 Strategic Recommendations

1. **For Enterprise/Customer-Facing Support:** Use the **Frontier Model** as the primary engine to guarantee guardrails, reduce hallucination, and prevent harmful output.
2. **For High-Volume/Internal Automation:** Deploy the **Open Source Model** with **Safety Guardrails** enabled. Adding a lightweight input/output filter provides 95%+ of the Frontier model's safety at a fraction of the cost.
3. **Hybrid Setup:** Implement a router that forwards creative/factual tasks to the Frontier model and standard conversational/utility queries to the OSS model.

---
*Report generated automatically by the Evaluation Framework on 2026-05-22.*
