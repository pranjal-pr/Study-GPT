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
