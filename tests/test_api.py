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
    files = [
        ("files", (f"doc{i}.pdf", b"%PDF-1.4\n", "application/pdf"))
        for i in range(api.MAX_UPLOAD_FILES + 1)
    ]
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
