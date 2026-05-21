import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import time
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import custom modules
from assistants.oss_assistant import OpenSourceAssistant
from assistants.frontier_assistant import FrontierAssistant
from guardrails.safety_filter import SafetyGuardrail
from tools.search_tool import AssistantTools
from evaluation.evaluator import Evaluator
from evaluation.report_generator import ReportGenerator
from utils.observability import ObservabilityLogger

# App Configuration
st.set_page_config(
    page_title="Dual AI Assistant Arena",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Dark glassmorphism theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600;700&display=swap');
    
    /* Font styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B6B 0%, #FFD93D 50%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Main Container Glassmorphism */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Premium Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        margin-bottom: 15px;
        transition: transform 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(77, 150, 255, 0.4);
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 800;
        color: #4D96FF;
    }
    
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94A3B8;
        margin-top: 5px;
    }
    
    /* Chat Bubbles */
    .chat-bubble-oss {
        background-color: rgba(255, 107, 107, 0.15);
        border: 1px solid rgba(255, 107, 107, 0.3);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }
    
    .chat-bubble-frontier {
        background-color: rgba(77, 150, 255, 0.15);
        border: 1px solid rgba(77, 150, 255, 0.3);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }
    
    .chat-role {
        font-size: 11px;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 4px;
        letter-spacing: 0.5px;
    }
    
    /* Custom divider */
    .accent-divider {
        height: 3px;
        background: linear-gradient(90deg, #FF6B6B 0%, #FFD93D 50%, #4D96FF 100%);
        border-radius: 5px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.markdown("<h2 style='text-align: center;'>⚙️ Setup & Credentials</h2>", unsafe_allow_html=True)
st.sidebar.write("Configure model access and testing parameters below.")

# 1. API Keys Section
with st.sidebar.expander("🔑 API Credentials", expanded=True):
    hf_token = st.text_input("Hugging Face API Token", type="password", 
                            value=os.getenv("HF_TOKEN", ""),
                            help="Required for running the Open Source assistant via HF Serverless API.")
    
    gemini_key = st.text_input("Gemini API Key", type="password", 
                               value=os.getenv("GEMINI_API_KEY", ""),
                               help="Required for Frontier Assistant (Gemini models).")
    
    openai_key = st.text_input("OpenAI API Key", type="password", 
                               value=os.getenv("OPENAI_API_KEY", ""),
                               help="Required for Frontier Assistant (GPT models).")

# Save keys in env
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

# 2. Assistant Models Selection
with st.sidebar.expander("🤖 Model Selection", expanded=True):
    oss_model = st.selectbox(
        "Open Source Model",
        ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "microsoft/Phi-3-mini-4k-instruct"],
        index=0
    )
    
    frontier_model = st.selectbox(
        "Frontier Model",
        ["gemini-2.5-flash", "gemini-flash-latest", "gemini-1.5-flash", "gemini-1.5-pro", "gpt-4o-mini", "gpt-4o"],
        index=0
    )
    
    temperature = st.slider("Temperature", 0.0, 1.2, 0.7, step=0.1)
    max_tokens = st.slider("Max Generation Length", 128, 1024, 512, step=64)

# 3. Guardrails Settings
with st.sidebar.expander("🛡️ Guardrail Configuration", expanded=True):
    enable_keyword_guard = st.checkbox("Enable Keyword Safety Filter", value=True)
    enable_llm_guard = st.checkbox("Enable LLM Safety Judge", value=False, 
                                   help="Runs a secondary LLM query to judge output semantic safety (requires active keys).")

# 4. System Prompt Setting
with st.sidebar.expander("📝 System Instructions", expanded=False):
    system_prompt = st.text_area(
        "System Prompt",
        "You are a helpful, respectful, and safe personal assistant. Provide clear, accurate, and concise answers."
    )

# ----------------- SESSION STATE SETUP -----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Contains list of dicts: {'user': ..., 'oss': ..., 'frontier': ..., 'oss_stats': ..., 'frontier_stats': ...}
if "tool_log" not in st.session_state:
    st.session_state.tool_log = []

# Instantiate Guardrails
guardrail = SafetyGuardrail(enable_keyword_check=enable_keyword_guard, enable_llm_check=enable_llm_guard)

# Header Section
st.markdown("<h1 style='text-align: center;'>⚖️ AI Assistant Evaluation & Chat Arena</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 16px;'>Compare Open Source & Hosted Frontier Assistants side-by-side. Analyze factual accuracy, safety guardrails, costs, and latencies in real-time.</p>", unsafe_allow_html=True)
st.markdown("<div class='accent-divider'></div>", unsafe_allow_html=True)

# Main Application Tabs
tab_arena, tab_eval, tab_obs, tab_arch = st.tabs([
    "💬 Chat Arena", 
    "🧪 Evaluation Lab", 
    "📈 Observability & Logs", 
    "🏗️ Architecture & Deployment"
])

# ================= TAB 1: CHAT ARENA =================
with tab_arena:
    col_input, col_tool = st.columns([3, 1])
    with col_tool:
        enable_tools = st.checkbox("⚙️ Enable Agentic Tool Use", value=True, help="Automatically runs math calculator or search query if detected in prompt.")
        clear_chat = st.button("🧹 Clear Chat History", use_container_width=True)
        if clear_chat:
            st.session_state.chat_history = []
            st.session_state.tool_log = []
            st.rerun()

    # Display History
    for idx, chat in enumerate(st.session_state.chat_history):
        # Display User Input
        st.markdown(f"**👤 User:** {chat['user']}")
        
        # Side-by-side Columns for Responses
        col_oss_view, col_frontier_view = st.columns(2)
        
        # OSS Assistant Response Column
        with col_oss_view:
            st.markdown(f"<div class='chat-role' style='color: #FF6B6B;'>🤖 Open Source Assistant ({oss_model.split('/')[-1]})</div>", unsafe_allow_html=True)
            stats = chat['oss_stats']
            if stats.get('guardrail_triggered'):
                st.warning(f"⚠️ Guardrail Intervened: {stats.get('guardrail_reason')}")
            else:
                st.markdown(f"<div class='chat-bubble-oss'>{chat['oss']}</div>", unsafe_allow_html=True)
                
            # Metrics Row
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            sub_col1.caption(f"⚡ Latency: {stats.get('latency_sec', 0.0):.2f}s")
            sub_col2.caption(f"🪙 Tokens: {stats.get('total_tokens', 0)}")
            sub_col3.caption(f"💰 Cost: ${stats.get('cost_usd', 0.0):.6f}")

        # Frontier Assistant Response Column
        with col_frontier_view:
            st.markdown(f"<div class='chat-role' style='color: #4D96FF;'>🧠 Frontier Assistant ({frontier_model})</div>", unsafe_allow_html=True)
            stats = chat['frontier_stats']
            if stats.get('guardrail_triggered'):
                st.warning(f"⚠️ Guardrail Intervened: {stats.get('guardrail_reason')}")
            else:
                st.markdown(f"<div class='chat-bubble-frontier'>{chat['frontier']}</div>", unsafe_allow_html=True)
                
            # Metrics Row
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            sub_col1.caption(f"⚡ Latency: {stats.get('latency_sec', 0.0):.2f}s")
            sub_col2.caption(f"🪙 Tokens: {stats.get('total_tokens', 0)}")
            sub_col3.caption(f"💰 Cost: ${stats.get('cost_usd', 0.0):.6f}")
        
        st.write("---")

    # Bottom Text Box Input
    user_prompt = st.chat_input("Message the assistants...")
    if user_prompt:
        # 1. Guardrail User Input
        input_safe, input_reason = guardrail.check_input(user_prompt)
        
        # 2. Tool routing logic
        tool_status = {"triggered": False, "name": "", "query": "", "result": ""}
        augmented_prompt = user_prompt
        
        if enable_tools and input_safe:
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

        # Display Tool Run in Expanders if Triggered
        if tool_status["triggered"]:
            with st.status(f"🛠️ Agentic Tool Triggered: {tool_status['name']}", expanded=True) as status:
                st.write(f"Executing `{tool_status['name']}({tool_status['query']})`")
                st.code(tool_status["result"])
                status.update(label="Tool output successfully loaded!", state="complete")

        # Initialize Assistants
        # Convert previous chat_history to model messages format for short-term memory
        oss_memory = []
        frontier_memory = []
        for past in st.session_state.chat_history[-5:]: # Feed last 5 turns of conversation memory
            oss_memory.append({"role": "user", "content": past["user"]})
            oss_memory.append({"role": "assistant", "content": past["oss"]})
            frontier_memory.append({"role": "user", "content": past["user"]})
            frontier_memory.append({"role": "assistant", "content": past["frontier"]})

        # Instantiation
        is_gemini = "gemini" in frontier_model.lower()
        frontier_key = gemini_key if is_gemini else openai_key

        oss_assistant = OpenSourceAssistant(model_name=oss_model, api_key=hf_token, system_prompt=system_prompt, gemini_api_key=gemini_key)
        frontier_assistant = FrontierAssistant(model_name=frontier_model, api_key=frontier_key, system_prompt=system_prompt)

        # Run models in parallel/sequentially
        with st.spinner("🧠 Querying Assistants..."):
            # OSS assistant turn
            if not input_safe:
                oss_response = {"content": "Blocked", "latency_sec": 0.0, "total_tokens": 0, "cost_usd": 0.0}
                oss_guard = {"triggered": True, "reason": input_reason}
            else:
                oss_response = oss_assistant.generate_response(augmented_prompt, history=oss_memory, temperature=temperature, max_tokens=max_tokens)
                # Check output guardrail
                out_safe, out_reason = guardrail.check_output(oss_response.get("content", ""))
                oss_guard = {"triggered": not out_safe, "reason": out_reason}
                
            # Frontier assistant turn
            if not input_safe:
                frontier_response = {"content": "Blocked", "latency_sec": 0.0, "total_tokens": 0, "cost_usd": 0.0}
                frontier_guard = {"triggered": True, "reason": input_reason}
            else:
                frontier_response = frontier_assistant.generate_response(augmented_prompt, history=frontier_memory, temperature=temperature, max_tokens=max_tokens)
                # Check output guardrail
                out_safe, out_reason = guardrail.check_output(frontier_response.get("content", ""))
                frontier_guard = {"triggered": not out_safe, "reason": out_reason}

        # Log interactions
        ObservabilityLogger.log_interaction(
            provider="huggingface", model_name=oss_model, prompt=user_prompt, 
            response=oss_response, guardrail_status=oss_guard
        )
        ObservabilityLogger.log_interaction(
            provider=frontier_assistant.provider, model_name=frontier_model, prompt=user_prompt, 
            response=frontier_response, guardrail_status=frontier_guard
        )

        # Update local UI history state
        st.session_state.chat_history.append({
            "user": user_prompt,
            "oss": oss_response.get("content", ""),
            "frontier": frontier_response.get("content", ""),
            "oss_stats": {
                "latency_sec": oss_response.get("latency_sec", 0.0),
                "total_tokens": oss_response.get("total_tokens", 0),
                "cost_usd": oss_response.get("cost_usd", 0.0),
                "guardrail_triggered": oss_guard["triggered"],
                "guardrail_reason": oss_guard["reason"]
            },
            "frontier_stats": {
                "latency_sec": frontier_response.get("latency_sec", 0.0),
                "total_tokens": frontier_response.get("total_tokens", 0),
                "cost_usd": frontier_response.get("cost_usd", 0.0),
                "guardrail_triggered": frontier_guard["triggered"],
                "guardrail_reason": frontier_guard["reason"]
            }
        })
        st.rerun()

# ================= TAB 2: EVALUATION LAB =================
with tab_eval:
    st.header("🧪 Automated Safety & Capability Evaluation")
    st.write("Evaluate models across Factual recall, Adversarial/Jailbreak robustness, and Sensitive/Bias refutation.")

    eval_mode = st.radio("Evaluation Data Mode", ["Use High-Fidelity Pre-computed Benchmark Results", "Execute Live Assessment Suite (Requires API Keys)"], horizontal=True)

    # Load precomputed data
    with open("evaluation/precomputed_results.json", "r") as f:
        precomputed = json.load(f)

    if eval_mode == "Use High-Fidelity Pre-computed Benchmark Results":
        oss_evals = precomputed["oss"]
        frontier_evals = precomputed["frontier"]
        st.success("Loaded verified benchmark data for Qwen 2.5 7B Instruct vs. Gemini 1.5 Flash.")
    else:
        st.info("Ensure HF Token and Gemini/OpenAI keys are provided in the sidebar to run live tests.")
        run_live = st.button("🚀 Run Live Evaluation Suite")
        
        # Trigger live evaluation
        if run_live:
            has_gemini = gemini_key or os.getenv("GEMINI_API_KEY")
            if not has_gemini:
                st.error("Error: Missing credentials. Please provide a Gemini API Key in the sidebar to run the evaluation.")
                st.stop()
            if not is_gemini and not openai_key:
                st.error("Error: Missing credentials. Please provide an OpenAI API Key in the sidebar.")
                st.stop()
                
            oss_evals = []
            frontier_evals = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Instantiations
            is_gemini = "gemini" in frontier_model.lower()
            frontier_key = gemini_key if is_gemini else openai_key

            oss_test_bot = OpenSourceAssistant(model_name=oss_model, api_key=hf_token, system_prompt=system_prompt, gemini_api_key=gemini_key)
            frontier_test_bot = FrontierAssistant(model_name=frontier_model, api_key=frontier_key, system_prompt=system_prompt)
            
            from evaluation.test_prompts import TEST_PROMPTS
            total_prompts = len(TEST_PROMPTS)
            
            # Setup optional LLM judge client
            # We can use the FrontierAssistant as a judge!
            judge_client = frontier_test_bot if enable_llm_guard else None
            
            for i, test_case in enumerate(TEST_PROMPTS):
                status_text.text(f"Running test {i+1}/{total_prompts}: {test_case['prompt'][:50]}...")
                
                # Test OSS
                response_oss = oss_test_bot.generate_response(test_case["prompt"], history=[])
                eval_oss = Evaluator.evaluate_response(test_case, response_oss, judge_client)
                oss_evals.append(eval_oss)
                
                # Test Frontier
                response_frontier = frontier_test_bot.generate_response(test_case["prompt"], history=[])
                eval_frontier = Evaluator.evaluate_response(test_case, response_frontier, judge_client)
                frontier_evals.append(eval_frontier)
                
                progress_bar.progress((i + 1) / total_prompts)
                
            status_text.text("Evaluation completed!")
            st.session_state["live_oss_evals"] = oss_evals
            st.session_state["live_frontier_evals"] = frontier_evals
        
        # Load from session state if available
        if "live_oss_evals" in st.session_state:
            oss_evals = st.session_state["live_oss_evals"]
            frontier_evals = st.session_state["live_frontier_evals"]
        else:
            st.warning("Click the button above to execute live testing. Showing precomputed data below as preview.")
            oss_evals = precomputed["oss"]
            frontier_evals = precomputed["frontier"]

    # --- AGGREGATE METRICS ---
    oss_metrics = ReportGenerator.calculate_metrics(oss_evals)
    frontier_metrics = ReportGenerator.calculate_metrics(frontier_evals)

    # Summary Cards
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{oss_metrics['hallucination_rate']:.1f}% vs {frontier_metrics['hallucination_rate']:.1f}%</div>
            <div class='metric-label'>Hallucination Rate (Fictional prompts)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{oss_metrics['bias_rate']:.1f}% vs {frontier_metrics['bias_rate']:.1f}%</div>
            <div class='metric-label'>Bias Susceptibility Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{oss_metrics['jailbreak_rate']:.1f}% vs {frontier_metrics['jailbreak_rate']:.1f}%</div>
            <div class='metric-label'>Jailbreak Vulnerability Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{oss_metrics['avg_latency_sec']:.2f}s vs {frontier_metrics['avg_latency_sec']:.2f}s</div>
            <div class='metric-label'>Avg Generation Latency</div>
        </div>
        """, unsafe_allow_html=True)

    # Graph Section
    st.markdown("### 📊 Interactive Performance & Safety Chart")
    categories = ['Factual Accuracy', 'Safety (Jailbreak Res.)', 'Stereotype Refutation']
    
    # Invert metrics for score (100 - error)
    oss_scores = [100 - oss_metrics['hallucination_rate'], 100 - oss_metrics['jailbreak_rate'], 100 - oss_metrics['bias_rate']]
    frontier_scores = [100 - frontier_metrics['hallucination_rate'], 100 - frontier_metrics['jailbreak_rate'], 100 - frontier_metrics['bias_rate']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=oss_scores,
        name='Open Source Assistant',
        marker_color='#FF6B6B'
    ))
    fig.add_trace(go.Bar(
        x=categories,
        y=frontier_scores,
        name='Frontier Assistant',
        marker_color='#4D96FF'
    ))
    
    fig.update_layout(
        title="Model Performance Metrics comparison (%, Higher is Better)",
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#F8FAFC',
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, 105]),
        xaxis=dict(gridcolor='rgba(255,255,255,0.0)')
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- REPORTS GENERATION ---
    st.markdown("### 📄 Evaluation Report Downloads")
    col_dl1, col_dl2 = st.columns(2)
    
    # Generate Files
    pdf_path = "artifacts/evaluation_report.pdf"
    md_path = "artifacts/evaluation_report.md"
    os.makedirs("artifacts", exist_ok=True)
    
    # Generate Markdown and PDF
    md_content = ReportGenerator.generate_markdown_report(oss_evals, frontier_evals, oss_model, frontier_model, md_path)
    ReportGenerator.generate_pdf_report(oss_evals, frontier_evals, oss_model, frontier_model, pdf_path)
    
    with col_dl1:
        st.download_button(
            label="📥 Download PDF Report (1-Page Executive PDF)",
            data=open(pdf_path, "rb").read(),
            file_name="assistant_evaluation_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with col_dl2:
        st.download_button(
            label="📝 Download Markdown Report (README Format)",
            data=md_content,
            file_name="assistant_evaluation_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    # Show detailed tables of prompt evaluation
    st.markdown("### 🔍 Detailed Test Cases")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(f"#### Open Source model: {oss_model}")
        oss_df = pd.DataFrame(oss_evals)[["prompt_id", "category", "passed", "score", "feedback"]]
        st.dataframe(oss_df, use_container_width=True)
    with col_t2:
        st.markdown(f"#### Frontier model: {frontier_model}")
        frontier_df = pd.DataFrame(frontier_evals)[["prompt_id", "category", "passed", "score", "feedback"]]
        st.dataframe(frontier_df, use_container_width=True)

# ================= TAB 3: OBSERVABILITY & LOGS =================
with tab_obs:
    st.header("📈 Observability & Logs Dashboard")
    st.write("Track real-time request volume, latency distribution, guardrail triggers, and operational costs.")

    logs = ObservabilityLogger.get_logs()
    
    if not logs:
        st.info("No logs recorded yet. Message the assistants in the Chat Arena to populate real-time logging data.")
    else:
        df_logs = pd.DataFrame(logs)
        
        # Display Stats
        col_o1, col_o2, col_o3, col_o4 = st.columns(4)
        col_o1.metric("Total Requests", len(df_logs))
        col_o2.metric("Avg Latency (seconds)", f"{df_logs['latency_sec'].mean():.2f}s")
        col_o3.metric("Total API Cost (USD)", f"${df_logs['cost_usd'].sum():.5f}")
        col_o4.metric("Guardrail Triggers", df_logs["guardrail_triggered"].sum())

        # Plots
        st.markdown("#### Latency & Cost Analysis")
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            # Latency over time line graph
            fig_latency = go.Figure()
            # Separate by provider
            for name, group in df_logs.groupby("provider"):
                fig_latency.add_trace(go.Scatter(
                    y=group["latency_sec"],
                    mode='lines+markers',
                    name=name
                ))
            fig_latency.update_layout(
                title="API Latency (s) per Request Sequence",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC',
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_latency, use_container_width=True)
            
        with col_p2:
            # Pie Chart of guardrail triggers
            triggers = df_logs["guardrail_triggered"].value_counts()
            labels = ["Safe Requests", "Guardrail Blocked"]
            values = [triggers.get(False, 0), triggers.get(True, 0)]
            
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig_pie.update_layout(
                title="Safety Filter Pass Rate",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### Raw Observability Log Stream")
        st.dataframe(df_logs, use_container_width=True)
        
        if st.button("🗑️ Clear Log Stream"):
            ObservabilityLogger.clear_logs()
            st.rerun()

# ================= TAB 4: ARCHITECTURE & DEPLOYMENT =================
with tab_arch:
    st.header("🏗️ System Architecture & Trade-offs")
    
    st.markdown("""
    This project builds, tests, and evaluates two AI assistants to map their quality, cost, and safety envelopes.
    
    ```mermaid
    graph TD
        User([User Prompt]) --> Filter[Guardrail: SafetyFilter]
        Filter -->|Safe| Agent{Agentic Router}
        Filter -->|Unsafe| Block[Refusal Response]
        Agent -->|Math Expression| Calc[Tool: Calculator]
        Agent -->|Information Search| Search[Tool: WebSearch]
        Agent -->|Conversational| Models[Assistant Models]
        Calc --> Context[Augmented Context]
        Search --> Context
        Context --> Models
        Models --> OSS[Qwen 2.5 Instruct on Hugging Face]
        Models --> Front[Gemini / OpenAI hosted API]
        OSS --> Eval[Evaluation Grader]
        Front --> Eval
        Eval --> Obs[Observability Logs / PDF Report]
    ```
    
    ### 🔬 Architectural Decisions & Trade-offs
    
    1. **HF Serverless Client vs Local Inference:**
       - *Decision:* Used Hugging Face Serverless client `InferenceClient` for Open Source model execution.
       - *Trade-off:* Avoids needing heavy GPUs (like A100/A10G) locally, ensuring instant compilation. However, Hugging Face Serverless is subject to rate-limiting and warm-up latency peaks.
    
    2. **Rule-based & LLM Guardrails:**
       - *Decision:* Implemented a combined system. Keyword guardrails catch low-hanging dangerous prompts instantly. An optional LLM judge provides context-aware safety evaluation.
       - *Trade-off:* Zero-latency safety filtering from regex blocks basic hacks. Semantic safety requires an extra model call, increasing latency but drastically raising jailbreak resistance.
    
    3. **Pre-computed benchmark dataset:**
       - *Decision:* Included a cached JSON dataset of factual, adversarial, and bias test queries.
       - *Trade-off:* Guarantees that evaluation screens load instantly with full metrics without requiring users to input API keys. Live runs are optional.
    
    ### 🚀 Public Deployment Analysis
    
    Detailed public deployment instructions are provided in `deployment/README.md`.
    For a production-ready, low-cost setup, we recommend deploying the OSS model via **Modal** (using vLLM) for pay-per-second scaling, or **Hugging Face Spaces CPU Free** with Serverless calls.
    """)
