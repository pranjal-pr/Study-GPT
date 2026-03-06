from observability import (
    MetricsStore,
    estimate_cost_usd,
    estimate_tokens,
    extract_usage_metrics,
    format_usd,
    has_model_pricing,
)


def test_estimate_tokens_basic():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2


def test_metrics_store_aggregates():
    store = MetricsStore(max_events=10)
    store.record_request(endpoint="/chat", method="POST", status_code=200, latency_ms=100)
    store.record_request(endpoint="/chat", method="POST", status_code=500, latency_ms=300)
    store.record_chat(
        provider="Groq",
        model="llama-3.3-70b-versatile",
        latency_ms=220,
        input_tokens=100,
        output_tokens=50,
        estimated_cost_usd=0.01,
    )
    store.record_upload(files_count=2)

    summary = store.summary()
    assert summary["requests_total"] == 2
    assert summary["errors_total"] == 1
    assert summary["chat_total"] == 1
    assert summary["upload_total"] == 1
    assert summary["estimated_cost_usd_total"] == 0.01


def test_estimate_cost_usd_default_non_negative():
    cost = estimate_cost_usd("unknown-model", input_tokens=1000, output_tokens=500)
    assert cost is None


def test_extract_usage_metrics_from_usage_metadata():
    class DummyResponse:
        usage_metadata = {"input_tokens": 123, "output_tokens": 45, "total_tokens": 168}

    usage = extract_usage_metrics(DummyResponse())
    assert usage == {"input_tokens": 123, "output_tokens": 45, "total_tokens": 168}


def test_extract_usage_metrics_from_response_metadata_token_usage():
    class DummyResponse:
        response_metadata = {"token_usage": {"prompt_tokens": 210, "completion_tokens": 34, "total_tokens": 244}}

    usage = extract_usage_metrics(DummyResponse())
    assert usage == {"input_tokens": 210, "output_tokens": 34, "total_tokens": 244}


def test_format_usd_avoids_scientific_notation():
    assert format_usd(3.413e-05) == "0.00003413"
    assert format_usd(None) == "n/a"


def test_has_model_pricing_knows_builtin_groq_and_rejects_unknown():
    assert has_model_pricing("llama-3.3-70b-versatile") is True
    assert has_model_pricing("unknown-model") is False
