import base64
import html
import os
import re
import time
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "shinzogpt-logo.svg"
MAX_PROMPT_CHARS = int(os.getenv("MAX_QUERY_CHARS", "2000"))
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "5"))
MAX_UPLOAD_FILE_MB = int(os.getenv("MAX_UPLOAD_FILE_MB", "20"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "12"))
HTTP_CONNECT_TIMEOUT_SEC = float(os.getenv("HTTP_CONNECT_TIMEOUT_SEC", "6"))
HTTP_READ_TIMEOUT_CHAT_SEC = float(os.getenv("HTTP_READ_TIMEOUT_CHAT_SEC", "180"))
HTTP_READ_TIMEOUT_UPLOAD_SEC = float(os.getenv("HTTP_READ_TIMEOUT_UPLOAD_SEC", "360"))

PROVIDER_OPTIONS = ["Groq", "Moonshot Kimi"]
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
MOONSHOT_MODELS = ["moonshot-v1-8k", "moonshot-v1-32k"]
MOONSHOT_NVIDIA_MODELS = ["moonshotai/kimi-k2-thinking"]
ROUTING_OPTIONS = ["Auto", "Chat only", "RAG only"]
ROUTING_MODE_MAP = {"Auto": "auto", "Chat only": "chat_only", "RAG only": "rag_only"}
TOOL_LABELS = {
    "calculator": "Calculator",
    "current_time": "Current Time",
    "weather": "Weather",
    "asset_price": "Price",
    "news": "News",
    "web_search": "Web Search",
}
AVAILABLE_TOOLS = [
    "Current Time",
    "Weather",
    "Price",
    "News",
    "Web Search",
    "Calculator",
]


def build_http_session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


HTTP_SESSION = build_http_session()


def parse_backend_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("message")
            if detail:
                return str(detail)
    except ValueError:
        pass
    return response.text or f"HTTP {response.status_code}"


def resolve_provider_config(provider: str):
    if provider == "Groq":
        return os.getenv("GROQ_API_KEY"), False, "Model", GROQ_MODELS

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


def fetch_runtime_summary():
    try:
        response = HTTP_SESSION.get(
            f"{API_URL}/metrics/summary",
            timeout=(HTTP_CONNECT_TIMEOUT_SEC, 10),
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


def format_usd_value(value) -> str:
    if value in (None, "", "n/a"):
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    formatted = f"{number:.8f}".rstrip("0").rstrip(".")
    if "." not in formatted:
        formatted = f"{formatted}.00"
    return formatted


def build_history_payload(chat_history: list[dict]) -> list[dict]:
    history_payload = []
    for message in chat_history[-MAX_HISTORY_TURNS:]:
        role = (message.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = normalize_message_content(role, message.get("content", "")).strip()
        if not content:
            continue
        history_payload.append({"role": role, "content": content[:MAX_PROMPT_CHARS]})
    return history_payload


def render_message(role: str, content: str, meta: str | None = None, is_latest: bool = False) -> None:
    role_class = "user" if role == "user" else "assistant"
    latest_class = " new" if is_latest else ""
    clean_content = normalize_message_content(role, content)
    safe_content = html.escape(clean_content).replace("\n", "<br>")
    meta_html = f'<div class="sg-meta">{html.escape(meta)}</div>' if meta else ""

    message_html = (
        f'<div class="sg-row {role_class}">'
        f'<div class="sg-msg {role_class}{latest_class}">'
        '<div class="sg-body">'
        f'<div class="sg-text">{safe_content}</div>'
        f"{meta_html}"
        "</div>"
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
        max-width: 980px;
        padding-top: 2.3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(14, 14, 16, 0.98), rgba(10, 10, 12, 0.98));
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .left-rail {
        position: fixed;
        top: 110px;
        left: 10px;
        display: flex;
        flex-direction: column;
        gap: 11px;
        z-index: 5;
    }

    .rail-dot {
        width: 36px;
        height: 36px;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        background: rgba(16, 16, 19, 0.9);
        color: #d2d6de;
        font-size: 0.92rem;
        display: flex;
        align-items: center;
        justify-content: center;
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

    .hero-center {
        max-width: 780px;
        margin: 0 auto 0.8rem;
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

    .chip-row-center {
        justify-content: center;
        margin-bottom: 1rem;
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

    .prompt-hint {
        color: var(--txt-muted);
        text-align: center;
        font-size: 0.88rem;
        margin: 0.2rem 0 0.55rem;
    }

    .small-muted {
        color: var(--txt-muted);
        font-size: 0.84rem;
    }

    [data-testid="stForm"] {
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 0.72rem 0.78rem 0.38rem;
        background: linear-gradient(180deg, rgba(24, 24, 28, 0.94), rgba(15, 15, 18, 0.94));
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.24);
        margin-bottom: 1.1rem;
    }

    [data-testid="stTextInput"] > div > div > input {
        border-radius: 999px !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        background: rgba(23, 24, 29, 0.96) !important;
        min-height: 46px !important;
        color: #f0f2f6 !important;
        font-size: 1rem !important;
    }

    [data-testid="stTextInput"] > div > div > input:focus {
        border-color: rgba(255, 255, 255, 0.26) !important;
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.08) !important;
    }

    .stFormSubmitButton > button {
        min-height: 46px;
        border-radius: 999px;
        font-weight: 600;
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

    .sg-row {
        display: flex;
        width: 100%;
        margin-bottom: 0.68rem;
    }

    .sg-row.user {
        justify-content: flex-end;
    }

    .sg-row.assistant {
        justify-content: flex-start;
    }

    .sg-msg {
        display: block;
        max-width: min(82%, 760px);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.62rem 0.82rem;
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

        .left-rail {
            display: none;
        }

        .hero h1 {
            font-size: 1.7rem;
        }

        .hero-logo {
            width: 54px;
            height: 54px;
        }

        .sg-msg {
            max-width: 100%;
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
if "runtime_summary" not in st.session_state:
    st.session_state.runtime_summary = None
if "runtime_summary_ts" not in st.session_state:
    st.session_state.runtime_summary_ts = 0.0

logo_data_uri = load_logo_data_uri()
logo_markup = (
    f'<img src="{logo_data_uri}" alt="ShinzoGPT logo" />'
    if logo_data_uri
    else '<div class="hero-logo-fallback">SG</div>'
)

st.markdown(
    """
    <div class="left-rail">
        <div class="rail-dot">S</div>
        <div class="rail-dot">C</div>
        <div class="rail-dot">M</div>
        <div class="rail-dot">R</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Session Controls")
    provider = st.selectbox("Provider", PROVIDER_OPTIONS)
    api_key, is_nvidia_key, model_label, model_options = resolve_provider_config(provider)
    selected_model = st.selectbox(model_label, model_options)
    routing_mode_label = st.selectbox(
        "Routing",
        ROUTING_OPTIONS,
        index=0,
        help="Auto chooses between chat and RAG. Chat only ignores docs. RAG only forces doc-grounded answers.",
    )
    routing_mode = ROUTING_MODE_MAP[routing_mode_label]
    enable_tools = st.toggle(
        "Enable Agent Tools",
        value=True,
        help="Used in chat mode for live lookups and calculations. RAG answers still come from your uploaded docs.",
    )

    key_chip = "Connected" if api_key else "Missing"
    key_class = "ok" if api_key else "warn"
    tools_chip = "On" if enable_tools else "Off"
    tools_class = "ok" if enable_tools else "warn"
    st.markdown(
        (
            '<div class="chip-row">'
            f'<span class="chip {key_class}">API Key: {key_chip}</span>'
            f'<span class="chip {tools_class}">Tools: {tools_chip}</span>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.caption("Available tools: " + ", ".join(AVAILABLE_TOOLS))

    st.markdown("---")
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
        if len(uploaded_files) > MAX_UPLOAD_FILES:
            st.error(f"Upload limit exceeded. Max {MAX_UPLOAD_FILES} files.")
            st.stop()

        for file in uploaded_files:
            size_mb = len(file.getvalue()) / (1024 * 1024)
            if size_mb > MAX_UPLOAD_FILE_MB:
                st.error(f"{file.name} exceeds {MAX_UPLOAD_FILE_MB}MB limit.")
                st.stop()

        with st.spinner("Indexing documents..."):
            try:
                files_data = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
                response = HTTP_SESSION.post(
                    f"{API_URL}/upload",
                    files=files_data,
                    timeout=(HTTP_CONNECT_TIMEOUT_SEC, HTTP_READ_TIMEOUT_UPLOAD_SEC),
                )

                if response.status_code == 200:
                    result = response.json()
                    st.session_state.vector_db_path = result.get("vector_db_path")
                    st.session_state.uploaded_sources = [f.name for f in uploaded_files]
                    st.success("Knowledge base updated.")
                else:
                    st.error(f"Backend failed: {parse_backend_error(response)}")
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

    st.markdown("---")
    summary_refresh = st.button("Refresh Runtime Metrics", use_container_width=True)
    if summary_refresh or (time.time() - st.session_state.runtime_summary_ts) > 30:
        st.session_state.runtime_summary = fetch_runtime_summary()
        st.session_state.runtime_summary_ts = time.time()

    with st.expander("Runtime Metrics", expanded=False):
        summary = st.session_state.runtime_summary
        if summary:
            left_m, right_m = st.columns(2)
            left_m.metric("Requests", summary.get("requests_total", 0))
            right_m.metric("Errors", summary.get("errors_total", 0))
            left_m.metric("Avg Latency (ms)", summary.get("avg_request_latency_ms", 0))
            right_m.metric("Est. Cost (USD)", format_usd_value(summary.get("estimated_cost_usd_total", 0)))
        else:
            st.caption("Metrics unavailable.")

if routing_mode == "chat_only":
    mode_text = "Chat Only Mode"
    mode_class = "warn"
elif routing_mode == "rag_only":
    mode_text = "RAG Only Mode" if st.session_state.vector_db_path else "RAG Only (No Knowledge Base)"
    mode_class = "ok" if st.session_state.vector_db_path else "warn"
else:
    mode_text = "Using Knowledge Base" if st.session_state.vector_db_path else "General Chat Mode"
    mode_class = "ok" if st.session_state.vector_db_path else "warn"

st.markdown(
    f"""
    <div class="hero hero-center">
        <div class="hero-logo">{logo_markup}</div>
        <div>
            <h1>ShinzoGPT</h1>
            <p>Ask anything. Attach knowledge only when you want grounded answers.</p>
        </div>
    </div>
    <div class="chip-row chip-row-center">
        <span class="chip">Provider: {provider}</span>
        <span class="chip">Model: {selected_model}</span>
        <span class="chip">Routing: {routing_mode_label}</span>
        <span class="chip {'ok' if enable_tools else 'warn'}">Tools: {'On' if enable_tools else 'Off'}</span>
        <span class="chip {mode_class}">{mode_text}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
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

st.markdown('<div class="prompt-hint">What is on your mind?</div>', unsafe_allow_html=True)
with st.form("prompt_form", clear_on_submit=True):
    prompt_col, send_col = st.columns([8.5, 1.5], gap="small")
    with prompt_col:
        user_prompt = st.text_input(
            "Prompt",
            placeholder="Ask ShinzoGPT...",
            label_visibility="collapsed",
        )
    with send_col:
        send_pressed = st.form_submit_button("Send", use_container_width=True)

if not send_pressed:
    user_prompt = ""

if user_prompt:
    cleaned_prompt = normalize_message_content("user", user_prompt).strip()
    if not cleaned_prompt:
        st.rerun()
    if len(cleaned_prompt) > MAX_PROMPT_CHARS:
        st.error(f"Prompt too long. Max {MAX_PROMPT_CHARS} characters.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": cleaned_prompt})

    if not api_key:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"API key for {provider} is missing. Add it in Space secrets."}
        )
        st.rerun()

    try:
        with st.spinner("Thinking..."):
            start = time.perf_counter()
            history_payload = build_history_payload(st.session_state.chat_history[:-1])
            payload = {
                "query": cleaned_prompt,
                "provider": provider,
                "model": selected_model,
                "api_key": api_key,
                "vector_db_path": st.session_state.vector_db_path,
                "is_nvidia_key": is_nvidia_key,
                "routing_mode": routing_mode,
                "enable_tools": enable_tools,
                "chat_history": history_payload,
            }
            response = HTTP_SESSION.post(
                f"{API_URL}/chat",
                json=payload,
                timeout=(HTTP_CONNECT_TIMEOUT_SEC, HTTP_READ_TIMEOUT_CHAT_SEC),
            )
            latency_ms = int((time.perf_counter() - start) * 1000)

            if response.status_code == 200:
                result = response.json()
                bot_reply = result.get("response", "No response from agent.")
                backend_metrics = result.get("metrics", {})
                route_used = (result.get("route_used") or "").lower()
                tool_used = str(backend_metrics.get("tool_used", "none")).strip().lower()
                if route_used == "rag":
                    mode = "RAG"
                elif route_used == "chat_tools":
                    mode = f"Chat + {TOOL_LABELS.get(tool_used, tool_used.replace('_', ' ').title())}"
                else:
                    mode = "Chat"
                server_latency = backend_metrics.get("latency_ms", latency_ms)
                input_tokens = backend_metrics.get("estimated_input_tokens", "-")
                output_tokens = backend_metrics.get("estimated_output_tokens", "-")
                cost_usd = format_usd_value(backend_metrics.get("estimated_cost_usd"))
                cost_label = f"${cost_usd}" if cost_usd != "n/a" else "n/a"
                meta = (
                    f"{provider} | {selected_model} | {mode} | {server_latency} ms"
                    f" | in:{input_tokens} tok | out:{output_tokens} tok | {cost_label}"
                )
                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply, "meta": meta})
            else:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"Backend error: {parse_backend_error(response)}"}
                )
    except Exception as e:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"Connection error: {e}. Is FastAPI running?"}
        )

    st.rerun()
