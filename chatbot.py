import os
from dotenv import load_dotenv
import streamlit as st
import requests

load_dotenv()

from rag_utility import process_documents_to_chroma_db, answer_question_with_agent

# --- NEW: Define the FastAPI Backend URL ---
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="ShinzoGPT",
    page_icon="🤖",
    layout="centered"
)
st.title("💬 ShinzoGPT")

# --- CONFIGURATION (Remains mostly the same, just gathering variables) ---
col1, col2 = st.columns(2)

with col1:
    provider = st.selectbox("Select Provider", ["Groq", "Gemini", "Moonshot Kimi"])

api_key = None
is_nvidia_key = False

if provider == "Groq":
    api_key = os.getenv("GROQ_API_KEY")
elif provider == "Gemini":
    api_key = os.getenv("GOOGLE_API_KEY")
elif provider == "Moonshot Kimi":
    api_key = os.getenv("MOONSHOT_API_KEY")
    if api_key and api_key.startswith("nvapi-"):
        is_nvidia_key = True

with col2:
    selected_model = None
    if provider == "Groq":
        selected_model = st.selectbox("Model:", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
    elif provider == "Gemini":
        selected_model = st.selectbox("Model:", ["gemini-2.0-flash", "gemini-1.5-flash"])
    elif provider == "Moonshot Kimi":
        if is_nvidia_key:
            selected_model = st.selectbox("Model (NVIDIA):", ["moonshotai/kimi-k2.5", "moonshotai/kimi-k2-thinking"])
        else:
            selected_model = st.selectbox("Model:", ["kimi-k2.5", "moonshot-v1-8k", "moonshot-v1-32k"])


# --- STATE MANAGEMENT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db_path" not in st.session_state:
    st.session_state.vector_db_path = None

# --- UPLOAD SECTION ---
with st.popover("➕ Attach Knowledge", help="Upload PDF documents"):
    st.markdown("### 📂 Upload Files")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Uploading and analyzing on backend..."):
                    try:
                        # NEW: Prepare files to send via HTTP POST
                        files_data = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
                        
                        # Hit the FastAPI /upload endpoint
                        response = requests.post(f"{API_URL}/upload", files=files_data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            # Save the path returned by the backend
                            st.session_state.vector_db_path = result.get("vector_db_path")
                            st.success("✅ RAG Active!")
                        else:
                            st.error(f"Backend Failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error connecting to backend: {e}. Is FastAPI running?")
    with col_b:
        if st.button("🗑️ Reset Chat"):
            st.session_state.chat_history = []
            st.session_state.vector_db_path = None
            st.rerun()

# --- DISPLAY CHAT ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- INPUT HANDLING ---
user_prompt = st.chat_input("Ask Chatbot...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    if not api_key:
        st.error(f"⚠️ API Key for {provider} is missing.")
    else:
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    
                    # NEW: Build the payload for the FastAPI /chat endpoint
                    payload = {
                        "query": user_prompt,
                        "provider": provider,
                        "model": selected_model,
                        "api_key": api_key,
                        "vector_db_path": st.session_state.vector_db_path,
                        "is_nvidia_key": is_nvidia_key
                    }
                    
                    # Send the request to the backend
                    response = requests.post(f"{API_URL}/chat", json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        bot_reply = result.get("response", "No response from agent.")
                        
                        st.markdown(bot_reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                    else:
                        st.error(f"Backend Error: {response.text}")
                        
            except Exception as e:
                st.error(f"Error connecting to backend: {str(e)}. Is FastAPI running?")