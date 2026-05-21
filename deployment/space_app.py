import os
import streamlit as st
from huggingface_hub import InferenceClient

# Streamlit App optimized for free Hugging Face Spaces CPU deployment
# Using Hugging Face Serverless API for inference to bypass GPU requirements

st.set_page_config(page_title="OSS Personal Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Open Source Personal Assistant")
st.write("Deployed publicly on Hugging Face Spaces using the Serverless Inference Client.")

# Sidebar for Model Selection and Token Configuration
st.sidebar.header("Configuration")
model_options = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct"
]
selected_model = st.sidebar.selectbox("Select Model", model_options, index=0)

# Optional HF Token, falls back to environment variable
hf_token = st.sidebar.text_input("Hugging Face API Token (Optional)", type="password")
if not hf_token:
    hf_token = os.getenv("HF_TOKEN")

system_prompt = st.sidebar.text_area(
    "System Prompt",
    "You are a helpful, respectful, and safe personal assistant. Provide clear and concise answers."
)

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Input
if prompt := st.chat_input("Ask me anything..."):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Initialize HF Client
                client = InferenceClient(model=selected_model, token=hf_token if hf_token else None)
                
                # Format messages
                api_messages = [{"role": "system", "content": system_prompt}]
                # Include last 5 messages for short-term memory
                for msg in st.session_state.messages[-6:-1]:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
                api_messages.append({"role": "user", "content": prompt})
                
                # Inference API call
                response = client.chat_completion(
                    messages=api_messages,
                    max_tokens=512,
                    temperature=0.7
                )
                
                assistant_response = response.choices[0].message.content
                st.write(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
            except Exception as e:
                error_msg = f"Error during generation: {str(e)}"
                st.error(error_msg)
                if not hf_token:
                    st.info("💡 Tip: Try providing a Hugging Face API Token in the sidebar to bypass rate limits.")
