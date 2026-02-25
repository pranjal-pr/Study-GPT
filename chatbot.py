import base64
import html
import os
from pathlib import Path
import re
import time

from dotenv import load_dotenv
import requests
import streamlit as st

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "shinzogpt-logo.svg"

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


def load_logo_data_uri() -> str | None:
    if not LOGO_PATH.exists():
        return None
    try:
        svg_text = LOGO_PATH.read_text(encoding="utf-8")
        encoded = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
        return f"data:image/svg+xml;base64,{encoded}"
    except OSError:
        return None


def normalize_message_content(role: str, content: str) -> str:
    text = content or ""

    # Defensive cleanup: older custom markup could leak a trailing literal </div> in user messages.
    if role == "user":
        text = re.sub(r"(?:\r?\n)?\s*</div>\s*$", "", text, count=1, flags=re.IGNORECASE)

    return text


def render_message(role: str, content: str, meta: str | None = None, is_latest: bool = False) -> None:
    role_class = "user" if role == "user" else "assistant"
    latest_class = " new" if is_latest else ""
    clean_content = normalize_message_content(role, content)
    safe_content = html.escape(clean_content).replace("\n", "<br>")
    meta_html = f'<div class="sg-meta">{html.escape(meta)}</div>' if meta else ""

    message_html = (
        f'<div class="sg-msg {role_class}{latest_class}">'
        '<div class="sg-body">'
        f'<div class="sg-text">{safe_content}</div>'
        f"{meta_html}"
        "</div>"
        "</div>"
    )
    st.markdown(message_html, unsafe_allow_html=True)


page_icon = str(LOGO_PATH) if LOGO_PATH.exists() else ":robot_face:"
st.set_page_config(page_title="ShinzoGPT", page_icon=page_icon, layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --bg-base: #060606;
        --bg-panel: #121212;
        --bg-panel-soft: #171717;
        --bg-pill: #1c1c1c;
        --txt-main: #f1f1f1;
        --txt-muted: #a4a4a4;
        --line: rgba(255, 255, 255, 0.13);
        --line-strong: rgba(255, 255, 255, 0.24);
    }

    html, body, [class*="css"] {
        font-family: "IBM Plex Sans", sans-serif;
        color: var(--txt-main);
        scroll-behavior: smooth;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(1px 1px at 40px 90px, rgba(255, 255, 255, 0.40), transparent 100%),
            radial-gradient(1px 1px at 120px 40px, rgba(255, 255, 255, 0.28), transparent 100%),
            radial-gradient(1.2px 1.2px at 220px 160px, rgba(255, 255, 255, 0.36), transparent 100%),
            radial-gradient(1px 1px at 300px 100px, rgba(255, 255, 255, 0.30), transparent 100%),
            radial-gradient(1.3px 1.3px at 430px 230px, rgba(255, 255, 255, 0.35), transparent 100%),
            radial-gradient(1.1px 1.1px at 510px 60px, rgba(255, 255, 255, 0.28), transparent 100%),
            radial-gradient(800px 300px at 50% -180px, rgba(255, 255, 255, 0.09), transparent 70%),
            linear-gradient(180deg, #040404 0%, #0a0a0a 100%);
        background-size:
            260px 260px,
            330px 330px,
            420px 420px,
            500px 500px,
            640px 640px,
            760px 760px,
            100% 100%,
            100% 100%;
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 8, 8, 0.62);
        backdrop-filter: blur(8px);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1220px;
        padding-top: 1.6rem;
    }

    .hero {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.15rem 1.3rem;
        margin-bottom: 1.05rem;
        background:
            linear-gradient(130deg, rgba(255, 255, 255, 0.05), transparent 40%),
            linear-gradient(180deg, rgba(18, 18, 18, 0.97), rgba(11, 11, 11, 0.97));
        box-shadow: 0 16px 34px rgba(0, 0, 0, 0.38);
        display: flex;
        align-items: center;
        gap: 0.95rem;
        animation: sg-fade-up 260ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
        transition: box-shadow 200ms ease, border-color 200ms ease;
    }

    .hero:hover {
        border-color: var(--line-strong);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.43);
    }

    .hero-logo {
        width: 62px;
        height: 62px;
        border-radius: 14px;
        border: 1px solid var(--line-strong);
        background: #0f0f0f;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: inset 0 0 22px rgba(255, 255, 255, 0.05);
    }

    .hero-logo img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    .hero-logo-fallback {
        font-family: "Space Grotesk", sans-serif;
        font-weight: 700;
        font-size: 1.08rem;
        letter-spacing: 0.5px;
        color: #d8d8d8;
    }

    .hero h1 {
        margin: 0;
        font-family: "Space Grotesk", sans-serif;
        font-size: 2rem;
        letter-spacing: 0.2px;
        color: var(--txt-main);
    }

    .hero p {
        margin: 0.3rem 0 0;
        color: var(--txt-muted);
        font-size: 0.94rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        background:
            linear-gradient(180deg, rgba(20, 20, 20, 0.94), rgba(12, 12, 12, 0.95));
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.25);
        transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: var(--line-strong) !important;
        box-shadow: 0 14px 26px rgba(0, 0, 0, 0.32);
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.42rem;
        margin-bottom: 0.68rem;
    }

    .chip {
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.27rem 0.66rem;
        font-size: 0.77rem;
        color: #dedede;
        background: var(--bg-pill);
        transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
    }

    .chip:hover {
        border-color: var(--line-strong);
        transform: translateY(-1px);
    }

    .chip.ok {
        border-color: rgba(255, 255, 255, 0.22);
        color: #f1f1f1;
    }

    .chip.warn {
        border-color: rgba(170, 170, 170, 0.34);
        color: #cfcfcf;
    }

    .empty-state {
        border: 1px dashed var(--line);
        border-radius: 14px;
        padding: 0.95rem 1.05rem;
        margin-top: 0.25rem;
        color: var(--txt-muted);
        background: rgba(255, 255, 255, 0.01);
    }

    .small-muted {
        color: var(--txt-muted);
        font-size: 0.84rem;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid var(--line);
        background: #1d1d1d;
        color: #efefef;
        transition: border-color 180ms ease, background 180ms ease, transform 180ms ease, box-shadow 180ms ease;
    }

    .stButton > button:hover {
        border-color: var(--line-strong);
        background: #232323;
        transform: translateY(-1px);
    }

    .stButton > button:focus,
    .stButton > button:focus-visible {
        box-shadow: none !important;
        border-color: var(--line-strong);
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px;
        border: 1px solid var(--line);
        background: #181818;
        transition: border-color 160ms ease, box-shadow 160ms ease, background 160ms ease;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: var(--line-strong);
        background: #1d1d1d;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed var(--line);
        border-radius: 12px;
        padding: 0.35rem 0.44rem 0.15rem;
        background: rgba(255, 255, 255, 0.012);
        transition: border-color 180ms ease, background 180ms ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--line-strong);
        background: rgba(255, 255, 255, 0.03);
    }

    [data-testid="stChatInput"] {
        margin-top: 0.34rem;
    }

    [data-testid="stChatInput"] > div {
        border: 1px solid var(--line) !important;
        border-radius: 18px;
        background:
            linear-gradient(180deg, rgba(30, 30, 30, 0.95), rgba(20, 20, 20, 0.95));
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.3);
        transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
    }

    [data-testid="stChatInput"] > div:focus-within {
        border-color: var(--line-strong) !important;
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.06), 0 10px 18px rgba(0, 0, 0, 0.35) !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        color: #f2f2f2 !important;
    }

    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] input:focus {
        box-shadow: none !important;
        outline: none !important;
    }

    .sg-msg {
        display: block;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.62rem 0.82rem;
        margin-bottom: 0.68rem;
        background: #151515;
        transition: border-color 180ms ease, box-shadow 180ms ease, background 180ms ease;
    }

    .sg-msg:hover {
        border-color: var(--line-strong);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.22);
    }

    .sg-msg.new {
        animation: sg-fade-up 260ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
    }

    .sg-msg.user {
        border-color: rgba(255, 255, 255, 0.2);
        background: #1c1c1c;
    }

    .sg-msg.assistant {
        border-color: rgba(255, 255, 255, 0.14);
        background: #141414;
    }

    .sg-body {
        min-width: 0;
        margin: 0;
        padding: 0;
    }

    .sg-text {
        color: var(--txt-main);
        font-size: 1.03rem;
        line-height: 1.52;
        word-break: break-word;
        margin: 0;
        padding: 0;
        text-indent: 0;
    }

    .sg-meta {
        margin-top: 0.36rem;
        color: #9a9a9a;
        font-size: 0.77rem;
    }

    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.16);
        border-radius: 8px;
        border: 2px solid transparent;
        background-clip: content-box;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.28);
        background-clip: content-box;
    }

    @keyframes sg-fade-up {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @media (prefers-reduced-motion: reduce) {
        * {
            animation: none !important;
            transition: none !important;
            scroll-behavior: auto !important;
        }
    }

    @media (max-width: 980px) {
        [data-testid="stMainBlockContainer"] {
            padding-top: 1.2rem;
        }

        .hero h1 {
            font-size: 1.7rem;
        }

        .hero-logo {
            width: 54px;
            height: 54px;
        }
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

logo_data_uri = load_logo_data_uri()
logo_markup = (
    f'<img src="{logo_data_uri}" alt="ShinzoGPT logo" />'
    if logo_data_uri
    else '<div class="hero-logo-fallback">SG</div>'
)

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-logo">{logo_markup}</div>
        <div>
            <h1>ShinzoGPT</h1>
            <p>Document-aware AI chat with fast model switching and source-grounded answers.</p>
        </div>
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

        history_len = len(st.session_state.chat_history)
        for idx, message in enumerate(st.session_state.chat_history):
            if message.get("role") == "user":
                message["content"] = normalize_message_content("user", message.get("content", ""))
            render_message(
                message["role"],
                message["content"],
                message.get("meta"),
                is_latest=(idx == history_len - 1),
            )

    user_prompt = st.chat_input("Ask ShinzoGPT...")

if user_prompt:
    cleaned_prompt = normalize_message_content("user", user_prompt).strip()
    if not cleaned_prompt:
        st.rerun()

    st.session_state.chat_history.append({"role": "user", "content": cleaned_prompt})

    if not api_key:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"API key for {provider} is missing. Add it in Space secrets."}
        )
        st.rerun()

    try:
        with st.spinner("Thinking..."):
            start = time.perf_counter()
            payload = {
                "query": cleaned_prompt,
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
