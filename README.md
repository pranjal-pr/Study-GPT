---
title: ShinzoGPT
sdk: docker
app_port: 7860
---

# ShinzoGPT

[![Tests](https://github.com/shinzoxD/streamlit-genai-chatbot/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/shinzoxD/streamlit-genai-chatbot/actions/workflows/ci-tests.yml)
[![Deploy To Hugging Face Space](https://github.com/shinzoxD/streamlit-genai-chatbot/actions/workflows/deploy-to-hf-space.yml/badge.svg)](https://github.com/shinzoxD/streamlit-genai-chatbot/actions/workflows/deploy-to-hf-space.yml)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Live%20App-yellow?logo=huggingface)](https://huggingface.co/spaces/shinzobolte/ShinzoGPT)

ShinzoGPT is a document-aware AI chat app with:
- Streamlit frontend (`chatbot.py`)
- FastAPI backend (`api.py`)
- Chroma vector store + LangChain RAG (`rag_utility.py`)
- Docker + Hugging Face Spaces deployment
- GitHub Actions auto-deploy to Hugging Face Space

Live links:
- App: https://huggingface.co/spaces/shinzobolte/ShinzoGPT
- GitHub Actions: https://github.com/shinzoxD/streamlit-genai-chatbot/actions

## Architecture Diagram

```mermaid
flowchart TD
    A[User Browser] --> B[Streamlit UI<br/>chatbot.py]
    B -->|POST /upload| C[FastAPI Backend<br/>api.py]
    B -->|POST /chat| C
    B -->|GET /metrics/*| C

    C --> D[Routing Layer<br/>auto, chat_only, rag_only]
    D --> E[Direct LLM Path]
    D --> F[RAG Path]

    E --> G[Providers<br/>Groq / Gemini / Moonshot]
    F --> H[RAG Utility<br/>rag_utility.py]
    H --> I[Chroma Vector DB]
    H --> G

    C --> J[Observability<br/>structured logs + metrics]
    C --> K[Rate Limiting + Validation + Retry]
```

## Quantitative Results (Measured)

Measured on `2026-02-26` using:
- Retrieval benchmark: `evaluation/benchmark.jsonl` on `vector_db_1771979965`
- Latency run: 10 real `/chat` requests in `chat_only` mode with Groq `llama-3.3-70b-versatile`

| Metric | Value |
|---|---:|
| Retrieval hit rate@3 | **1.00** |
| Retrieval MRR@3 | **1.00** |
| Faithfulness avg | **1.00** |
| p95 chat latency | **2849.02 ms** |
| Error rate | **0.00%** |

Reproducible reports:
- [Full eval report](./evaluation/report_latest_full.json)
- [Retrieval-only report](./evaluation/report_latest_retrieval.json)
- [Runtime benchmark report](./evaluation/runtime_benchmark_latest.json)

## Flagship-Readiness Features

### 1) Evaluation Metrics
- Retrieval metrics: hit rate, MRR, precision@k
- Answer grounding proxy: faithfulness score from response/context overlap
- Keyword recall against benchmark expectations

Run evaluation:

```bash
python evaluation/evaluate_rag.py ^
  --vector-db-path vector_db_1234567890 ^
  --benchmark-file evaluation/benchmark.jsonl ^
  --top-k 3 ^
  --provider Groq ^
  --model llama-3.3-70b-versatile ^
  --api-key <YOUR_API_KEY> ^
  --out-file evaluation/report.json
```

You can also run retrieval-only eval (omit provider/model/api-key).

### 2) Testing + Reliability
- Unit/integration tests in `tests/`
- LLM invocation retry with exponential backoff
- HTTP retry + timeout strategy in Streamlit client
- Stronger backend error handling and safer responses

Run tests:

```bash
pytest -q
```

### 3) Observability
- Structured JSON logs (`request_completed`, `chat_completed`, etc.)
- Request-level latency/error telemetry
- Estimated token/cost tracking (configurable)
- Metrics endpoints:
  - `GET /metrics/summary`
  - `GET /metrics/events?limit=50`
- Streamlit includes a runtime metrics panel

Optional protection:
- Set `OBSERVABILITY_TOKEN` and call metrics endpoints with `x-observability-token` header.

### 4) Security Basics
- Query length limits (`MAX_QUERY_CHARS`)
- Upload limits (`MAX_UPLOAD_FILES`, `MAX_UPLOAD_FILE_MB`)
- File type and empty-file checks
- Rate limiting per IP for `/chat` and `/upload`
- Safer vector DB path validation
- API keys never logged

## Environment Variables

See `env_template.txt` for all settings.

Core keys:
- `GROQ_API_KEY`
- `GOOGLE_API_KEY`
- `MOONSHOT_API_KEY`

Operational controls:
- `CHAT_RATE_LIMIT_PER_MIN`
- `UPLOAD_RATE_LIMIT_PER_MIN`
- `LLM_RETRY_ATTEMPTS`
- `HTTP_CONNECT_TIMEOUT_SEC`
- `HTTP_READ_TIMEOUT_CHAT_SEC`
- `HTTP_READ_TIMEOUT_UPLOAD_SEC`

Cost estimation (optional):
- `DEFAULT_INPUT_COST_PER_1K_TOKENS`
- `DEFAULT_OUTPUT_COST_PER_1K_TOKENS`
- `MODEL_PRICING_OVERRIDES_JSON`

## Deploy (GitHub -> Hugging Face)

Workflow: `.github/workflows/deploy-to-hf-space.yml`

On each push to `main`/`master`:
1. Validates required secrets
2. Pushes exact GitHub commit to HF Space `main`

Required GitHub repository secrets:
- `HF_TOKEN`: Hugging Face token with write access to the Space repo
- `HF_SPACE_ID`: `username/space-name`

## Challenges / Tradeoffs / Results (Template)

## Failures + Tradeoffs

### Known Failure Modes
- Retrieval can return the right source file but still miss a specific detail chunk.
- In-memory rate limiting resets on restart and is not shared across replicas.
- Faithfulness score is heuristic and can overestimate factual grounding.
- Provider/network timeouts can still happen under external API instability.

### Mitigations
- Use top-k retrieval + source display + route controls (`chat_only` / `rag_only`) for debugging.
- Add retries with exponential backoff for model calls and HTTP client requests.
- Enforce strict input/file limits and payload validation to reduce bad requests.
- Expose runtime metrics and structured logs to detect latency/error regressions quickly.

### Tradeoffs
- Simple observability and rate limiting keep deployment lightweight but are less robust than distributed stacks.
- Cost telemetry is estimate-based unless provider-side token usage metadata is available.
- Current benchmark is small and domain-focused; broader generalization needs a larger test set.

## Challenges / Tradeoffs / Results (Template)

### Challenges
- Balancing rich custom UI with Streamlit rerun behavior.
- Keeping RAG robust while avoiding brittle answer hallucinations.
- Making observability useful without paid monitoring tools.

### Tradeoffs
- In-memory rate limiting is simple but not distributed.
- Faithfulness scoring is heuristic; stronger eval can use LLM judges/human review.
- Cost tracking is estimate-based unless provider token usage metadata is available.

### Results (Fill with your measurements)
- Retrieval hit rate@3: `__`
- MRR@3: `__`
- Faithfulness avg: `__`
- p95 chat latency: `__ ms`
- Error rate: `__ %`
