import shutil
from pathlib import Path

from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


def allow_all(*_args, **_kwargs):
    return True, 0.0


def test_chat_rejects_query_too_long(monkeypatch):
    monkeypatch.setattr(api.rate_limiter, "is_allowed", allow_all)
    payload = {
        "query": "x" * (api.MAX_QUERY_CHARS + 1),
        "provider": "Groq",
        "model": "llama-3.3-70b-versatile",
        "api_key": "dummy",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 400
    assert "exceeds" in response.json()["detail"].lower()


def test_chat_rate_limited(monkeypatch):
    monkeypatch.setattr(api.rate_limiter, "is_allowed", lambda *_args, **_kwargs: (False, 10))
    payload = {
        "query": "hello",
        "provider": "Groq",
        "model": "llama-3.3-70b-versatile",
        "api_key": "dummy",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 429


def test_chat_success_returns_metrics(monkeypatch):
    monkeypatch.setattr(api.rate_limiter, "is_allowed", allow_all)

    class DummyResponse:
        content = "hello from model"

    class DummyLLM:
        def invoke(self, _query):
            return DummyResponse()

    monkeypatch.setattr(api, "get_llm", lambda *args, **kwargs: DummyLLM())

    payload = {
        "query": "hello",
        "provider": "Groq",
        "model": "llama-3.3-70b-versatile",
        "api_key": "dummy",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "response" in body
    assert "metrics" in body
    assert "estimated_input_tokens" in body["metrics"]
    assert body["route_used"] == "chat"


def test_upload_rejects_non_pdf(monkeypatch):
    monkeypatch.setattr(api.rate_limiter, "is_allowed", allow_all)
    files = [("files", ("notes.txt", b"bad", "text/plain"))]
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "only pdf" in response.json()["detail"].lower()


def test_upload_rejects_too_many_files(monkeypatch):
    monkeypatch.setattr(api.rate_limiter, "is_allowed", allow_all)
    files = [("files", (f"doc{i}.pdf", b"%PDF-1.4\n", "application/pdf")) for i in range(api.MAX_UPLOAD_FILES + 1)]
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "upload limit exceeded" in response.json()["detail"].lower()


def test_upload_success(monkeypatch):
    monkeypatch.setattr(api.rate_limiter, "is_allowed", allow_all)
    monkeypatch.setattr(api, "process_documents_to_chroma_db", lambda _files: "/tmp/vector_db_test")

    files = [("files", ("paper.pdf", b"%PDF-1.4\nmock", "application/pdf"))]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    assert response.json()["vector_db_path"] == "/tmp/vector_db_test"


def test_should_use_rag_with_conversational_prefix():
    assert api.should_use_rag("so tell me about machine learning") is False


def test_should_use_rag_for_followup_when_history_is_document_context():
    history = [
        api.ChatTurn(role="assistant", content="Sources: LLM Interview Questions.pdf"),
    ]
    assert api.should_use_rag("What are the most important questions", history) is True


def test_should_use_rag_for_referential_query_without_doc_markers():
    assert api.should_use_rag("Explain its current use") is True


def test_rag_only_requires_vector_db(monkeypatch):
    monkeypatch.setattr(api.rate_limiter, "is_allowed", allow_all)
    payload = {
        "query": "what is this document about?",
        "provider": "Groq",
        "model": "llama-3.3-70b-versatile",
        "api_key": "dummy",
        "routing_mode": "rag_only",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 400
    assert "no knowledge base" in response.json()["detail"].lower()


def test_chat_only_forces_non_rag(monkeypatch):
    monkeypatch.setattr(api.rate_limiter, "is_allowed", allow_all)

    class DummyResponse:
        content = "plain chat response"

    class DummyLLM:
        def invoke(self, _query):
            return DummyResponse()

    monkeypatch.setattr(api, "get_llm", lambda *args, **kwargs: DummyLLM())
    called = {"rag": False}

    def fake_rag(*_args, **_kwargs):
        called["rag"] = True
        return "rag response"

    monkeypatch.setattr(api, "answer_question_with_agent", fake_rag)

    working_dir = Path(api.__file__).resolve().parent
    test_vector_dir = working_dir / "tests_tmp_vector_db"
    test_vector_dir.mkdir(exist_ok=True)
    try:
        payload = {
            "query": "this document summary please",
            "provider": "Groq",
            "model": "llama-3.3-70b-versatile",
            "api_key": "dummy",
            "vector_db_path": str(test_vector_dir),
            "routing_mode": "chat_only",
        }
        response = client.post("/chat", json=payload)
        assert response.status_code == 200
        assert response.json()["route_used"] == "chat"
        assert called["rag"] is False
    finally:
        shutil.rmtree(test_vector_dir, ignore_errors=True)


def test_chat_uses_history_context(monkeypatch):
    monkeypatch.setattr(api.rate_limiter, "is_allowed", allow_all)
    captured = {"prompt": ""}

    class DummyResponse:
        content = "continuity ok"

    class DummyLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return DummyResponse()

    monkeypatch.setattr(api, "get_llm", lambda *args, **kwargs: DummyLLM())

    payload = {
        "query": "The last topic",
        "provider": "Groq",
        "model": "llama-3.3-70b-versatile",
        "api_key": "dummy",
        "routing_mode": "chat_only",
        "chat_history": [
            {"role": "user", "content": "Explain deep learning"},
            {"role": "assistant", "content": "Deep learning uses neural networks."},
        ],
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    assert "Conversation history" in captured["prompt"]
    assert "Explain deep learning" in captured["prompt"]


def test_rag_receives_history_context(monkeypatch):
    monkeypatch.setattr(api.rate_limiter, "is_allowed", allow_all)
    captured = {"history": ""}

    class DummyLLM:
        pass

    monkeypatch.setattr(api, "get_llm", lambda *args, **kwargs: DummyLLM())

    def fake_rag(_query, _llm, _vector_db_path, chat_history_context=""):
        captured["history"] = chat_history_context
        return "rag continuity ok"

    monkeypatch.setattr(api, "answer_question_with_agent", fake_rag)

    working_dir = Path(api.__file__).resolve().parent
    test_vector_dir = working_dir / "tests_tmp_vector_db_memory"
    test_vector_dir.mkdir(exist_ok=True)
    try:
        payload = {
            "query": "What about that topic?",
            "provider": "Groq",
            "model": "llama-3.3-70b-versatile",
            "api_key": "dummy",
            "routing_mode": "rag_only",
            "vector_db_path": str(test_vector_dir),
            "chat_history": [
                {"role": "user", "content": "Tell me about Transformers."},
                {"role": "assistant", "content": "Transformers use self-attention."},
            ],
        }
        response = client.post("/chat", json=payload)
        assert response.status_code == 200
        assert "Tell me about Transformers." in captured["history"]
    finally:
        shutil.rmtree(test_vector_dir, ignore_errors=True)


def test_auto_mode_prefers_rag_for_followup_with_doc_history(monkeypatch):
    monkeypatch.setattr(api.rate_limiter, "is_allowed", allow_all)
    called = {"rag": False}

    class DummyLLM:
        def invoke(self, _query):
            class DummyResponse:
                content = "chat response"

            return DummyResponse()

    monkeypatch.setattr(api, "get_llm", lambda *args, **kwargs: DummyLLM())

    def fake_rag(*_args, **_kwargs):
        called["rag"] = True
        return "rag response"

    monkeypatch.setattr(api, "answer_question_with_agent", fake_rag)

    working_dir = Path(api.__file__).resolve().parent
    test_vector_dir = working_dir / "tests_tmp_vector_db_auto_followup"
    test_vector_dir.mkdir(exist_ok=True)
    try:
        payload = {
            "query": "What are the most important questions",
            "provider": "Groq",
            "model": "llama-3.3-70b-versatile",
            "api_key": "dummy",
            "vector_db_path": str(test_vector_dir),
            "routing_mode": "auto",
            "chat_history": [
                {"role": "assistant", "content": "Sources: LLM Interview Questions.pdf"},
            ],
        }
        response = client.post("/chat", json=payload)
        assert response.status_code == 200
        assert response.json()["route_used"] == "rag"
        assert called["rag"] is True
    finally:
        shutil.rmtree(test_vector_dir, ignore_errors=True)


def test_chat_can_return_tool_agent_route(monkeypatch):
    monkeypatch.setattr(api.rate_limiter, "is_allowed", allow_all)

    class DummyLLM:
        def invoke(self, _query):
            class DummyResponse:
                content = "unused"

            return DummyResponse()

    monkeypatch.setattr(api, "get_llm", lambda *args, **kwargs: DummyLLM())
    monkeypatch.setattr(
        api,
        "run_agent_with_tools",
        lambda *_args, **_kwargs: {"response": "2 + 2 is 4.", "tool_used": "calculator"},
    )

    payload = {
        "query": "calculate 2+2",
        "provider": "Groq",
        "model": "llama-3.3-70b-versatile",
        "api_key": "dummy",
        "routing_mode": "chat_only",
        "enable_tools": True,
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["route_used"] == "chat_tools"
    assert body["metrics"]["tool_used"] == "calculator"
    assert body["response"] == "2 + 2 is 4."


def test_chat_skips_tools_when_disabled(monkeypatch):
    monkeypatch.setattr(api.rate_limiter, "is_allowed", allow_all)

    class DummyResponse:
        content = "plain chat"

    class DummyLLM:
        def invoke(self, _query):
            return DummyResponse()

    monkeypatch.setattr(api, "get_llm", lambda *args, **kwargs: DummyLLM())
    called = {"tool_agent": False}

    def fake_tool_agent(*_args, **_kwargs):
        called["tool_agent"] = True
        return {"response": "tool answer", "tool_used": "calculator"}

    monkeypatch.setattr(api, "run_agent_with_tools", fake_tool_agent)

    payload = {
        "query": "calculate 2+2",
        "provider": "Groq",
        "model": "llama-3.3-70b-versatile",
        "api_key": "dummy",
        "routing_mode": "chat_only",
        "enable_tools": False,
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["route_used"] == "chat"
    assert body["metrics"]["tool_used"] == "none"
    assert called["tool_agent"] is False
