import os
import time

from dotenv import load_dotenv
import requests
import streamlit as st

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

PROVIDER_OPTIONS = ["Groq", "Gemini", "Moonshot Kimi"]
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
GEMINI_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash"]
MOONSHOT_MODELS = ["kimi-k2.5", "moonshot-v1-8k", "moonshot-v1-32k"]
MOONSHOT_NVIDIA_MODELS = ["moonshotai/kimi-k2.5", "moonshotai/kimi-k2-thinking"]


def resolve_provider_config(provider: str):
    if provider == "Groq":
        return os.getenv("GROQ_API_KEY"), False, "Model", GROQ_MODELS
    if provider == "Gemini":
        return os.getenv("GOOGLE_API_KEY"), False, "Model", GEMINI_MODELS

    moonshot_key = os.getenv("MOONSHOT_API_KEY")
    is_nvidia_key = bool(moonshot_key and moonshot_key.startswith("nvapi-"))
    if is_nvidia_key:
        return moonshot_key, True, "Model (NVIDIA)", MOONSHOT_NVIDIA_MODELS
    return moonshot_key, False, "Model", MOONSHOT_MODELS


st.set_page_config(page_title="ShinzoGPT", page_icon=":robot_face:", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --bg-top: #05182b;
        --bg-bottom: #040b14;
        --panel: #0b243f;
        --panel-soft: #102b49;
        --text-main: #f3f7fd;
        --text-muted: #9ab2ca;
        --accent: #1ec7ba;
        --accent-2: #6ac5ff;
        --border: rgba(117, 162, 209, 0.34);
    }

    html, body, [class*="css"] {
        font-family: "IBM Plex Sans", sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(1100px 360px at 10% -10%, rgba(30, 199, 186, 0.16), transparent 60%),
            radial-gradient(1100px 360px at 88% -10%, rgba(106, 197, 255, 0.16), transparent 60%),
            linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(5, 10, 18, 0.5);
        backdrop-filter: blur(8px);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1240px;
        padding-top: 2rem;
    }

    .hero {
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.35rem 1.45rem;
        margin-bottom: 1.2rem;
        background: linear-gradient(140deg, rgba(16, 43, 73, 0.92), rgba(11, 36, 63, 0.92));
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.28);
    }

    .hero h1 {
        margin: 0;
        color: var(--text-main);
        font-size: 2.15rem;
        font-family: "Space Grotesk", sans-serif;
        letter-spacing: 0.2px;
    }

    .hero p {
        margin: 0.38rem 0 0;
        color: var(--text-muted);
        font-size: 0.98rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border) !important;
        border-radius: 16px !important;
        background: linear-gradient(180deg, rgba(11, 36, 63, 0.9), rgba(8, 26, 45, 0.9));
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.2);
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-bottom: 0.7rem;
    }

    .chip {
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.28rem 0.68rem;
        font-size: 0.78rem;
        color: #d7e7f8;
        background: rgba(16, 43, 73, 0.75);
    }

    .chip.ok {
        border-color: rgba(30, 199, 186, 0.58);
        color: #b9fcf5;
    }

    .chip.warn {
        border-color: rgba(255, 188, 93, 0.58);
        color: #ffe5b0;
    }

    .empty-state {
        border: 1px dashed var(--border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-top: 0.3rem;
        color: var(--text-muted);
        background: rgba(16, 43, 73, 0.45);
    }

    .meta {
        margin-top: 0.35rem;
        color: var(--text-muted);
        font-size: 0.78rem;
    }

    .small-muted {
        color: var(--text-muted);
        font-size: 0.84rem;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid var(--border);
    }

    .stButton > button:hover {
        border-color: rgba(30, 199, 186, 0.7);
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px;
        border: 1px solid var(--border);
        background: rgba(16, 43, 73, 0.62);
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed var(--border);
        border-radius: 12px;
        padding: 0.35rem 0.45rem 0.15rem;
        background: rgba(16, 43, 73, 0.42);
    }

    [data-testid="stChatInput"] > div {
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(11, 36, 63, 0.85);
    }

    [data-testid="stChatMessage"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(11, 36, 63, 0.72);
        padding: 0.25rem 0.58rem;
        margin-bottom: 0.7rem;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        border-color: rgba(30, 199, 186, 0.5);
        background: rgba(30, 199, 186, 0.1);
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-color: rgba(106, 197, 255, 0.45);
        background: rgba(106, 197, 255, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db_path" not in st.session_state:
    st.session_state.vector_db_path = None
if "uploaded_sources" not in st.session_state:
    st.session_state.uploaded_sources = []

st.markdown(
    """
    <div class="hero">
        <h1>ShinzoGPT</h1>
        <p>Document-aware AI chat with fast model switching and source-grounded answers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

control_col, chat_col = st.columns([1, 2.1], gap="large")

with control_col:
    with st.container(border=True):
        st.markdown("### Session Controls")

        provider = st.selectbox("Provider", PROVIDER_OPTIONS)
        api_key, is_nvidia_key, model_label, model_options = resolve_provider_config(provider)
        selected_model = st.selectbox(model_label, model_options)

        key_chip = "Connected" if api_key else "Missing"
        key_class = "ok" if api_key else "warn"
        st.markdown(
            f'<div class="chip-row"><span class="chip {key_class}">API Key: {key_chip}</span></div>',
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown("### Knowledge")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        process_col, reset_col = st.columns(2)
        with process_col:
            process_docs = st.button(
                "Process PDFs",
                type="primary",
                use_container_width=True,
                disabled=not uploaded_files,
            )
        with reset_col:
            reset_chat = st.button("Reset Chat", use_container_width=True)

        if process_docs and uploaded_files:
            with st.spinner("Indexing documents..."):
                try:
                    files_data = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
                    response = requests.post(f"{API_URL}/upload", files=files_data, timeout=300)

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.vector_db_path = result.get("vector_db_path")
                        st.session_state.uploaded_sources = [f.name for f in uploaded_files]
                        st.success("Knowledge base updated.")
                    else:
                        st.error(f"Backend failed: {response.text}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

        if reset_chat:
            st.session_state.chat_history = []
            st.session_state.vector_db_path = None
            st.session_state.uploaded_sources = []
            st.rerun()

        rag_active = bool(st.session_state.vector_db_path)
        rag_chip = "RAG Active" if rag_active else "RAG Inactive"
        rag_class = "ok" if rag_active else "warn"
        doc_count = len(st.session_state.uploaded_sources)
        st.markdown(
            f"""
            <div class="chip-row">
                <span class="chip {rag_class}">{rag_chip}</span>
                <span class="chip">Docs: {doc_count}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.uploaded_sources:
            st.markdown("<div class='small-muted'>Indexed files:</div>", unsafe_allow_html=True)
            for doc_name in st.session_state.uploaded_sources[:5]:
                st.markdown(f"- {doc_name}")

with chat_col:
    with st.container(border=True):
        mode_text = "Using Knowledge Base" if st.session_state.vector_db_path else "General Chat Mode"
        mode_class = "ok" if st.session_state.vector_db_path else "warn"
        st.markdown(
            f"""
            <div class="chip-row">
                <span class="chip">Provider: {provider}</span>
                <span class="chip">Model: {selected_model}</span>
                <span class="chip {mode_class}">{mode_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not st.session_state.chat_history:
            st.markdown(
                """
                <div class="empty-state">
                    Start by asking anything, or upload PDFs on the left to enable document-grounded answers.
                </div>
                """,
                unsafe_allow_html=True,
            )

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("meta"):
                    st.markdown(f'<div class="meta">{message["meta"]}</div>', unsafe_allow_html=True)

    user_prompt = st.chat_input("Ask ShinzoGPT...")

if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    if not api_key:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"API key for {provider} is missing. Add it in Space secrets."}
        )
        st.rerun()

    try:
        with st.spinner("Thinking..."):
            start = time.perf_counter()
            payload = {
                "query": user_prompt,
                "provider": provider,
                "model": selected_model,
                "api_key": api_key,
                "vector_db_path": st.session_state.vector_db_path,
                "is_nvidia_key": is_nvidia_key,
            }
            response = requests.post(f"{API_URL}/chat", json=payload, timeout=300)
            latency_ms = int((time.perf_counter() - start) * 1000)

            if response.status_code == 200:
                result = response.json()
                bot_reply = result.get("response", "No response from agent.")
                mode = "RAG" if st.session_state.vector_db_path else "Chat"
                meta = f"{provider} | {selected_model} | {mode} | {latency_ms} ms"
                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply, "meta": meta})
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": f"Backend error: {response.text}"})
    except Exception as e:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"Connection error: {e}. Is FastAPI running?"}
        )

    st.rerun()
