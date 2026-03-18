import json
import os
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Iterator, List, Optional, cast

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent_tools import prepare_agent_tool_run, run_agent_with_tools
from observability import (
    estimate_cost_usd,
    estimate_tokens,
    extract_usage_metrics,
    has_model_pricing,
    log_event,
    metrics_store,
)
from rag_utility import answer_question_with_agent, build_rag_prompt, process_documents_to_chroma_db

load_dotenv()

MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "2000"))
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "5"))
MAX_UPLOAD_FILE_MB = int(os.getenv("MAX_UPLOAD_FILE_MB", "20"))
MAX_UPLOAD_FILE_BYTES = MAX_UPLOAD_FILE_MB * 1024 * 1024
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "12"))
LLM_RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", "2"))
LLM_RETRY_BASE_DELAY_SEC = float(os.getenv("LLM_RETRY_BASE_DELAY_SEC", "0.5"))
OBSERVABILITY_TOKEN = os.getenv("OBSERVABILITY_TOKEN", "")
REQUEST_METRICS_EXCLUDED_PATHS = {
    "/health",
    "/metrics/summary",
    "/metrics/events",
}


PROVIDER_MODELS = {
    "Groq": {"llama-3.3-70b-versatile", "llama-3.1-8b-instant"},
    "Moonshot Kimi": {
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshotai/kimi-k2-thinking",
    },
}

PROVIDER_ENV_KEYS = {
    "Groq": "GROQ_API_KEY",
    "Moonshot Kimi": "MOONSHOT_API_KEY",
}


class InMemoryRateLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._requests = defaultdict(deque)
        self._limits = {
            "chat": (int(os.getenv("CHAT_RATE_LIMIT_PER_MIN", "60")), 60),
            "upload": (int(os.getenv("UPLOAD_RATE_LIMIT_PER_MIN", "10")), 60),
        }

    def is_allowed(self, bucket: str, client_key: str) -> tuple[bool, float]:
        max_requests, window_sec = self._limits[bucket]
        now = time.time()
        rate_key = f"{bucket}:{client_key}"

        with self._lock:
            entries = self._requests[rate_key]
            while entries and (now - entries[0]) > window_sec:
                entries.popleft()

            if len(entries) >= max_requests:
                retry_after = max(0.0, window_sec - (now - entries[0]))
                return False, retry_after

            entries.append(now)
            return True, 0.0


rate_limiter = InMemoryRateLimiter()


app = FastAPI(
    title="ShinzoGPT Agentic RAG API",
    description="Backend API for the Agentic Retrieval-Augmented Generation chatbot.",
    version="1.1",
)


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    provider: str
    model: str
    api_key: Optional[str] = None
    vector_db_path: Optional[str] = None
    is_nvidia_key: bool = False
    routing_mode: str = "auto"
    enable_tools: bool = True
    chat_history: List[ChatTurn] = Field(default_factory=list)


def _build_history_context(chat_history: List["ChatTurn"]) -> str:
    if not chat_history:
        return ""

    lines = []
    for turn in chat_history[-MAX_HISTORY_TURNS:]:
        role = turn.role.lower().strip()
        if role not in {"user", "assistant"}:
            continue

        content = (turn.content or "").strip()
        if not content:
            continue

        # Keep context bounded and clean.
        content = re.sub(r"\s+", " ", content)[:600]
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")

    return "\n".join(lines)


def _normalize_query_for_routing(query: str) -> str:
    q = (query or "").strip().lower()
    q = re.sub(r"\s+", " ", q)

    # Remove conversational fillers so "so tell me about ..." behaves like "tell me about ...".
    filler_patterns = [
        r"^(hey|hi|hello|yo|hii+|hola|ok|okay|well|so|please|pls)\s+",
        r"^(can you|could you|would you|will you|kindly)\s+",
    ]

    changed = True
    while changed:
        changed = False
        for pattern in filler_patterns:
            new_q = re.sub(pattern, "", q)
            if new_q != q:
                q = new_q.strip()
                changed = True

    return q


def _history_suggests_document_context(chat_history: List["ChatTurn"]) -> bool:
    if not chat_history:
        return False

    history_markers = (
        "sources:",
        ".pdf",
        "document",
        "documents",
        "uploaded",
        "from the paper",
        "from the doc",
        "according to",
    )
    for turn in chat_history[-MAX_HISTORY_TURNS:]:
        content = (turn.content or "").lower()
        if any(marker in content for marker in history_markers):
            return True
    return False


def _is_follow_up_query(normalized_query: str) -> bool:
    if not normalized_query:
        return False

    follow_up_patterns = (
        "what about",
        "tell me more",
        "more detailed",
        "the last topic",
        "last topic",
        "continue",
        "elaborate",
        "and ",
        "also ",
    )
    if any(normalized_query.startswith(pattern) for pattern in follow_up_patterns):
        return True

    follow_up_tokens = ("that", "those", "it", "its", "them", "this", "these", "previous", "earlier", "same")
    tokens = normalized_query.split()
    return len(tokens) <= 10 and any(token in follow_up_tokens for token in tokens)


def should_use_rag(query: str, chat_history: Optional[List["ChatTurn"]] = None) -> bool:
    """
    Route clearly general-knowledge prompts to normal LLM chat even when a vector DB exists.
    """
    q = _normalize_query_for_routing(query)
    if not q:
        return False

    doc_markers = [
        "document",
        "documents",
        "pdf",
        "file",
        "uploaded",
        "upload",
        "admit card",
        "resume",
        "invoice",
        "this card",
        "this file",
        "from the doc",
        "from this",
        "according to the document",
        "in the document",
        "this paper",
        "that paper",
        "from the paper",
    ]
    if any(marker in q for marker in doc_markers):
        return True

    history_is_doc_context = _history_suggests_document_context(chat_history or [])
    if history_is_doc_context and _is_follow_up_query(q):
        return True

    # If a KB is attached and the question is clearly referential ("its", "that", "the last topic"),
    # prefer RAG even when prior turns were not explicitly source-tagged.
    if _is_follow_up_query(q):
        return True

    general_prefixes = (
        "what is",
        "what are",
        "who is",
        "who are",
        "when is",
        "when was",
        "where is",
        "why is",
        "how does",
        "how do",
        "explain",
        "define",
        "tell me about",
        "give me an overview",
        "give me an overview of",
        "overview of",
        "example of",
    )
    if q.startswith(general_prefixes):
        # If recent turns were document-grounded, keep follow-up/general questions in RAG.
        if history_is_doc_context:
            return True
        return False

    return True


def get_llm(provider: str, model: str, api_key: str, is_nvidia_key: bool):
    """Instantiates the correct LLM based on the request payload."""
    if provider == "Groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, api_key=cast(Any, api_key))
    if provider == "Moonshot Kimi":
        from langchain_openai import ChatOpenAI

        base_url = "https://integrate.api.nvidia.com/v1" if is_nvidia_key else "https://api.moonshot.cn/v1"
        return ChatOpenAI(model=model, api_key=cast(Any, api_key), base_url=base_url)
    raise ValueError("Invalid LLM Provider")


def invoke_with_retries(func):
    """Simple bounded retry for flaky LLM/API operations."""
    last_error = None
    for attempt in range(LLM_RETRY_ATTEMPTS + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt == LLM_RETRY_ATTEMPTS:
                raise
            delay = LLM_RETRY_BASE_DELAY_SEC * (2**attempt)
            time.sleep(delay)
    raise RuntimeError(f"Retry loop exited unexpectedly: {last_error}")


def _client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return "unknown"


def _enforce_rate_limit(request: Request, bucket: str) -> None:
    ip = _client_ip(request)
    allowed, retry_after = rate_limiter.is_allowed(bucket=bucket, client_key=ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {bucket}. Retry in {int(retry_after) + 1} seconds.",
        )


def _authorize_observability(request: Request) -> None:
    if not OBSERVABILITY_TOKEN:
        return
    token = request.headers.get("x-observability-token", "")
    if token != OBSERVABILITY_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized observability access.")


def _resolve_api_key(provider: str, request_api_key: Optional[str]) -> str:
    if request_api_key:
        return request_api_key
    env_key_name = PROVIDER_ENV_KEYS.get(provider)
    if not env_key_name:
        return ""
    env_value = os.getenv(env_key_name, "")
    return env_value


def _should_record_request_metrics(path: str) -> bool:
    return path not in REQUEST_METRICS_EXCLUDED_PATHS


def _validate_chat_payload(request: ChatRequest) -> None:
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(query) > MAX_QUERY_CHARS:
        raise HTTPException(status_code=400, detail=f"Query exceeds {MAX_QUERY_CHARS} characters.")

    if request.provider not in PROVIDER_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported provider.")
    if request.model not in PROVIDER_MODELS[request.provider]:
        raise HTTPException(status_code=400, detail="Unsupported model for selected provider.")
    if request.routing_mode not in {"auto", "chat_only", "rag_only"}:
        raise HTTPException(status_code=400, detail="Invalid routing_mode. Use auto, chat_only, or rag_only.")
    if len(request.chat_history) > MAX_HISTORY_TURNS:
        raise HTTPException(status_code=400, detail=f"chat_history exceeds max turns ({MAX_HISTORY_TURNS}).")
    for turn in request.chat_history:
        if turn.role.lower() not in {"user", "assistant"}:
            raise HTTPException(status_code=400, detail="chat_history role must be user or assistant.")
        if len((turn.content or "").strip()) > MAX_QUERY_CHARS:
            raise HTTPException(status_code=400, detail="A chat_history message exceeds max allowed length.")

    if request.vector_db_path:
        abs_path = os.path.abspath(request.vector_db_path)
        working_dir = os.path.dirname(os.path.abspath(__file__))
        if not abs_path.startswith(working_dir):
            raise HTTPException(status_code=400, detail="Invalid vector database path.")
        if not os.path.isdir(abs_path):
            raise HTTPException(status_code=400, detail="Vector database path does not exist.")


def _resolve_use_rag(payload: ChatRequest) -> bool:
    if payload.routing_mode == "chat_only":
        return False
    if payload.routing_mode == "rag_only":
        if not payload.vector_db_path:
            raise HTTPException(status_code=400, detail="RAG-only mode selected, but no knowledge base is attached.")
        return True
    return bool(payload.vector_db_path and should_use_rag(payload.query, payload.chat_history))


def _build_chat_prompt(query: str, history_context: str) -> str:
    if history_context:
        return (
            "You are a helpful conversational assistant.\n"
            "Use the conversation history for continuity (references like 'that', 'previous', 'last topic').\n"
            "If the history is irrelevant, prioritize the latest user question.\n\n"
            f"Conversation history:\n{history_context}\n\n"
            f"Current user question:\n{query}"
        )
    return query


def _finalize_chat_payload(
    payload: ChatRequest,
    history_context: str,
    response: str,
    usage_metrics: dict[str, int],
    started: float,
    tool_used: str,
    use_rag: bool,
):
    latency_ms = (time.perf_counter() - started) * 1000
    token_usage_source = "estimated"
    if usage_metrics:
        input_tokens = int(usage_metrics.get("input_tokens", 0))
        output_tokens = int(usage_metrics.get("output_tokens", 0))
        token_usage_source = "provider"
    else:
        input_tokens = estimate_tokens(payload.query) + estimate_tokens(history_context)
        output_tokens = estimate_tokens(response)

    estimated_cost = estimate_cost_usd(payload.model, input_tokens, output_tokens)
    pricing_configured = has_model_pricing(payload.model)
    route_used = "rag" if use_rag else ("chat_tools" if tool_used != "none" else "chat")

    metrics_store.record_chat(
        provider=payload.provider,
        model=payload.model,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimated_cost or 0.0,
    )
    log_event(
        "chat_completed",
        provider=payload.provider,
        model=payload.model,
        latency_ms=round(latency_ms, 2),
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_cost_usd=estimated_cost,
        token_usage_source=token_usage_source,
        pricing_configured=pricing_configured,
        rag_used=use_rag,
        tool_used=tool_used,
        routing_mode=payload.routing_mode,
    )
    return {
        "response": response,
        "route_used": route_used,
        "metrics": {
            "latency_ms": round(latency_ms, 2),
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "estimated_cost_usd": estimated_cost,
            "tool_used": tool_used,
            "token_usage_source": token_usage_source,
            "pricing_configured": pricing_configured,
        },
    }


def _serialize_stream_event(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True) + "\n"


def _chunk_text_for_stream(text: str, chunk_size: int = 24) -> Iterator[str]:
    if not text:
        return

    buffer = ""
    for piece in re.split(r"(\s+)", text):
        if not piece:
            continue
        if buffer and len(buffer) + len(piece) > chunk_size:
            yield buffer
            buffer = piece
            continue
        buffer += piece

    if buffer:
        yield buffer


def _extract_stream_text(chunk: Any) -> str:
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content or "")


@app.middleware("http")
async def request_telemetry_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    status_code = 500
    path = request.url.path
    method = request.method
    ip = _client_ip(request)

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency_ms = (time.perf_counter() - started) * 1000
        if _should_record_request_metrics(path):
            metrics_store.record_request(
                endpoint=path,
                method=method,
                status_code=status_code,
                latency_ms=latency_ms,
            )
        log_event(
            "request_completed",
            request_id=request_id,
            method=method,
            path=path,
            status_code=status_code,
            latency_ms=round(latency_ms, 2),
            client_ip=ip,
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_documents(request: Request, files: List[UploadFile] = File(...)):
    """Handles PDF uploads, saves them temporarily, and triggers RAG ingestion."""
    _enforce_rate_limit(request, "upload")

    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"Upload limit exceeded. Max files: {MAX_UPLOAD_FILES}.")

    class StreamlitMockFile:
        """Helper class to match the Streamlit UploadedFile interface used by the RAG utility."""

        def __init__(self, name: str, buffer: bytes):
            self.name = name
            self.buffer = buffer

        def getbuffer(self):
            return self.buffer

    processed_files = []
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Each uploaded file must have a filename.")
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are allowed: {file.filename}")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"File is empty: {file.filename}")
        if len(content) > MAX_UPLOAD_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{file.filename} exceeds {MAX_UPLOAD_FILE_MB}MB upload limit.",
            )
        processed_files.append(StreamlitMockFile(file.filename, content))

    try:
        db_path = process_documents_to_chroma_db(processed_files)
        metrics_store.record_upload(files_count=len(processed_files))
        log_event("upload_processed", files_count=len(processed_files), vector_db_path=db_path)
        return {
            "message": "Documents successfully ingested into the vector database.",
            "vector_db_path": db_path,
        }
    except HTTPException:
        raise
    except Exception as exc:
        log_event("upload_failed", error_type=type(exc).__name__)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")


@app.post("/chat")
async def chat(request: Request, payload: ChatRequest):
    """Processes user queries with direct chat or RAG, and returns response + telemetry metrics."""
    _enforce_rate_limit(request, "chat")
    _validate_chat_payload(payload)

    api_key = _resolve_api_key(payload.provider, payload.api_key)
    if not api_key:
        raise HTTPException(status_code=400, detail=f"API key for {payload.provider} is missing.")

    started = time.perf_counter()

    try:
        llm = get_llm(payload.provider, payload.model, api_key, payload.is_nvidia_key)
        history_context = _build_history_context(payload.chat_history)
        tool_used = "none"
        usage_metrics: dict[str, int] = {}
        use_rag = _resolve_use_rag(payload)

        if use_rag:
            vector_db_path = payload.vector_db_path or ""
            rag_result = invoke_with_retries(
                lambda: answer_question_with_agent(
                    payload.query,
                    llm,
                    vector_db_path,
                    chat_history_context=history_context,
                )
            )
            if isinstance(rag_result, dict):
                response = cast(str, rag_result.get("response", ""))
                usage_metrics = cast(dict[str, int], rag_result.get("usage", {}))
            else:
                rag_response = cast(Optional[str], rag_result)
                response = rag_response or "I couldn't find relevant information for that in your uploaded documents."
        else:
            tool_result = None
            if payload.enable_tools:
                tool_result = invoke_with_retries(
                    lambda: run_agent_with_tools(
                        payload.query,
                        llm,
                        chat_history_context=history_context,
                    )
                )

            if tool_result and tool_result.get("response"):
                response = cast(str, tool_result["response"])
                tool_used = cast(str, tool_result.get("tool_used", "none"))
                usage_metrics = cast(dict[str, int], tool_result.get("usage", {}))
            else:
                prompt = _build_chat_prompt(payload.query, history_context)
                raw_response = invoke_with_retries(lambda: llm.invoke(prompt))
                response = cast(str, getattr(raw_response, "content", raw_response))
                usage_metrics = extract_usage_metrics(raw_response)

        return _finalize_chat_payload(
            payload=payload,
            history_context=history_context,
            response=response,
            usage_metrics=usage_metrics,
            started=started,
            tool_used=tool_used,
            use_rag=use_rag,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log_event("chat_failed", error_type=type(exc).__name__)
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")


@app.post("/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest):
    """Streams assistant text as newline-delimited JSON events."""
    _enforce_rate_limit(request, "chat")
    _validate_chat_payload(payload)

    api_key = _resolve_api_key(payload.provider, payload.api_key)
    if not api_key:
        raise HTTPException(status_code=400, detail=f"API key for {payload.provider} is missing.")

    llm = get_llm(payload.provider, payload.model, api_key, payload.is_nvidia_key)
    history_context = _build_history_context(payload.chat_history)
    use_rag = _resolve_use_rag(payload)

    def event_stream():
        started = time.perf_counter()
        tool_used = "none"
        usage_metrics: dict[str, int] = {}
        response_parts: list[str] = []

        try:
            if use_rag:
                prompt, _sources = build_rag_prompt(
                    user_question=payload.query,
                    vector_db_path=payload.vector_db_path or "",
                    chat_history_context=history_context,
                )
                if prompt is None:
                    fallback = "I couldn't find relevant information for that in your uploaded documents."
                    for delta in _chunk_text_for_stream(fallback):
                        response_parts.append(delta)
                        yield _serialize_stream_event({"type": "chunk", "delta": delta})
                else:
                    stream_fn = getattr(llm, "stream", None)
                    if callable(stream_fn):
                        last_chunk = None
                        for chunk in stream_fn(prompt):
                            last_chunk = chunk
                            delta = _extract_stream_text(chunk)
                            if not delta:
                                continue
                            response_parts.append(delta)
                            yield _serialize_stream_event({"type": "chunk", "delta": delta})
                        usage_metrics = extract_usage_metrics(last_chunk)
                    else:
                        raw_response = invoke_with_retries(lambda: llm.invoke(prompt))
                        usage_metrics = extract_usage_metrics(raw_response)
                        text = cast(str, getattr(raw_response, "content", raw_response))
                        for delta in _chunk_text_for_stream(text):
                            response_parts.append(delta)
                            yield _serialize_stream_event({"type": "chunk", "delta": delta})
            else:
                prepared_tool_run = None
                if payload.enable_tools:
                    prepared_tool_run = invoke_with_retries(
                        lambda: prepare_agent_tool_run(
                            payload.query,
                            llm,
                            chat_history_context=history_context,
                        )
                    )

                if prepared_tool_run:
                    tool_used = cast(str, prepared_tool_run.get("tool_used", "none"))
                    direct_response = cast(Optional[str], prepared_tool_run.get("direct_response"))
                    if direct_response is not None:
                        for delta in _chunk_text_for_stream(direct_response):
                            response_parts.append(delta)
                            yield _serialize_stream_event({"type": "chunk", "delta": delta})
                    else:
                        synthesis_prompt = cast(str, prepared_tool_run["synthesis_prompt"])
                        source_urls = cast(list[str], prepared_tool_run.get("source_urls", []))
                        stream_fn = getattr(llm, "stream", None)
                        if callable(stream_fn):
                            last_chunk = None
                            for chunk in stream_fn(synthesis_prompt):
                                last_chunk = chunk
                                delta = _extract_stream_text(chunk)
                                if not delta:
                                    continue
                                response_parts.append(delta)
                                yield _serialize_stream_event({"type": "chunk", "delta": delta})
                            usage_metrics = extract_usage_metrics(last_chunk)
                        else:
                            raw_response = invoke_with_retries(lambda: llm.invoke(synthesis_prompt))
                            usage_metrics = extract_usage_metrics(raw_response)
                            text = cast(str, getattr(raw_response, "content", raw_response))
                            for delta in _chunk_text_for_stream(text):
                                response_parts.append(delta)
                                yield _serialize_stream_event({"type": "chunk", "delta": delta})

                        streamed_response = "".join(response_parts).strip()
                        if tool_used == "web_search" and source_urls and "sources:" not in streamed_response.lower():
                            trailing_sources = f"\n\nSources: {', '.join(source_urls[:3])}"
                            response_parts.append(trailing_sources)
                            yield _serialize_stream_event({"type": "chunk", "delta": trailing_sources})
                else:
                    prompt = _build_chat_prompt(payload.query, history_context)
                    stream_fn = getattr(llm, "stream", None)
                    if callable(stream_fn):
                        last_chunk = None
                        for chunk in stream_fn(prompt):
                            last_chunk = chunk
                            delta = _extract_stream_text(chunk)
                            if not delta:
                                continue
                            response_parts.append(delta)
                            yield _serialize_stream_event({"type": "chunk", "delta": delta})
                        usage_metrics = extract_usage_metrics(last_chunk)
                    else:
                        raw_response = invoke_with_retries(lambda: llm.invoke(prompt))
                        usage_metrics = extract_usage_metrics(raw_response)
                        text = cast(str, getattr(raw_response, "content", raw_response))
                        for delta in _chunk_text_for_stream(text):
                            response_parts.append(delta)
                            yield _serialize_stream_event({"type": "chunk", "delta": delta})

            response_text = "".join(response_parts).strip()
            final_payload = _finalize_chat_payload(
                payload=payload,
                history_context=history_context,
                response=response_text,
                usage_metrics=usage_metrics,
                started=started,
                tool_used=tool_used,
                use_rag=use_rag,
            )
            yield _serialize_stream_event(
                {
                    "type": "done",
                    "route_used": final_payload["route_used"],
                    "metrics": final_payload["metrics"],
                }
            )
        except Exception as exc:
            log_event("chat_failed", error_type=type(exc).__name__)
            yield _serialize_stream_event({"type": "error", "message": f"Chat failed: {exc}"})

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/metrics/summary")
def metrics_summary(request: Request):
    _authorize_observability(request)
    return metrics_store.summary()


@app.get("/metrics/events")
def metrics_events(request: Request, limit: int = 50):
    _authorize_observability(request)
    safe_limit = max(1, min(limit, 200))
    return {"events": metrics_store.events(limit=safe_limit)}
