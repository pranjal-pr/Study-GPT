---
title: ShinzoGPT
sdk: docker
app_port: 7860
---

# ShinzoGPT

[![Tests](https://github.com/shinzoxD/streamlit-genai-chatbot/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/shinzoxD/streamlit-genai-chatbot/actions/workflows/ci-tests.yml)
[![Deploy To Hugging Face Space](https://github.com/shinzoxD/streamlit-genai-chatbot/actions/workflows/deploy-to-hf-space.yml/badge.svg)](https://github.com/shinzoxD/streamlit-genai-chatbot/actions/workflows/deploy-to-hf-space.yml)
[![Secret Scan](https://github.com/shinzoxD/streamlit-genai-chatbot/actions/workflows/secret-scan.yml/badge.svg)](https://github.com/shinzoxD/streamlit-genai-chatbot/actions/workflows/secret-scan.yml)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Live%20App-yellow?logo=huggingface)](https://huggingface.co/spaces/shinzobolte/ShinzoGPT)
[![Version](https://img.shields.io/badge/version-v1.0.0-black)](https://github.com/shinzoxD/streamlit-genai-chatbot/tree/v1.0.0)

ShinzoGPT is a document-aware AI chat app with:
- Streamlit frontend (`chatbot.py`)
- FastAPI backend (`api.py`)
- Chroma vector store + LangChain RAG (`rag_utility.py`)
- Agent tools for non-RAG chat (`agent_tools.py`): web search + safe calculator
- Docker + Hugging Face Spaces deployment
- GitHub Actions auto-deploy to Hugging Face Space

Website: https://huggingface.co/spaces/shinzobolte/ShinzoGPT

Live links:
- App: https://huggingface.co/spaces/shinzobolte/ShinzoGPT
- GitHub Actions: https://github.com/shinzoxD/streamlit-genai-chatbot/actions

## Resume-Ready Bullets

- Built and deployed an end-to-end RAG chatbot (Streamlit + FastAPI + Chroma) on Hugging Face Spaces with GitHub Actions CD.
- Implemented multi-provider model routing, conversational memory, and agent-style tool usage (web search + calculator).
- Added observability, reliability, and security controls: structured logs, runtime metrics, retries/timeouts, rate limiting, and automated secret scanning.
- Benchmarked all configured models with reproducible evaluation tooling, including optional RAGAS scoring (faithfulness + answer relevancy).
- Productionized project quality gates with CI checks (`ruff`, `black --check`, `mypy`, `pytest`) and release versioning (`v1.0.0`).

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
    E --> L[Tool Agent Layer<br/>web search + calculator]

    E --> G[Providers<br/>Groq / Moonshot]
    L --> G
    F --> H[RAG Utility<br/>rag_utility.py]
    H --> I[Chroma Vector DB]
    H --> G

    C --> J[Observability<br/>structured logs + metrics]
    C --> K[Rate Limiting + Validation + Retry]
```

Static architecture image (for platforms without Mermaid support):

![ShinzoGPT Architecture](./assets/architecture-diagram.svg)

## Quantitative Results (Measured)

Measured at `2026-02-27T00:05:24Z` using:
- Retrieval benchmark: `evaluation/benchmark.jsonl` on `vector_db_1771979965`
- Runtime benchmark: 5 real `/chat` requests per model in `chat_only` mode
- Model matrix tested: all models currently exposed by this app configuration

| Provider | Model | Status | Hit@3 | MRR@3 | Faithfulness | p95 Latency (ms) | Error Rate |
|---|---|---|---:|---:|---:|---:|---:|
| Groq | llama-3.3-70b-versatile | ok | 1.00 | 1.00 | 1.00 | 1698.38 | 0.00% |
| Groq | llama-3.1-8b-instant | ok | 1.00 | 1.00 | 0.80 | 891.13 | 0.00% |
| Moonshot Kimi | moonshot-v1-8k | not run | - | - | - | - | - |
| Moonshot Kimi | moonshotai/kimi-k2-thinking | ok | 1.00 | 1.00 | 1.00 | 16093.45 | 0.00% |

Recommended default for recruiter walkthrough: **Groq `llama-3.3-70b-versatile`** (best quality-speed balance in this run).

Moonshot note for this snapshot:
- `moonshot-v1-8k` was not included in the benchmark run.

Reproducible reports:
- [Model matrix report](./evaluation/model_matrix_latest.json)
- [Full eval report (single-model baseline)](./evaluation/report_latest_full.json)
- [Retrieval-only report (single-model baseline)](./evaluation/report_latest_retrieval.json)
- [Runtime benchmark report (single-model baseline)](./evaluation/runtime_benchmark_latest.json)

## Flagship-Readiness Features

### 1) Evaluation Metrics
- Retrieval metrics: hit rate, MRR, precision@k
- Answer grounding proxy: lexical faithfulness score from response/context overlap
- Keyword recall against benchmark expectations
- Optional RAGAS metrics: `faithfulness`, `answer_relevancy`

Run evaluation:

```bash
pip install -r evaluation/requirements-eval.txt

python evaluation/evaluate_rag.py ^
  --vector-db-path vector_db_1234567890 ^
  --benchmark-file evaluation/benchmark.jsonl ^
  --top-k 3 ^
  --provider Groq ^
  --model llama-3.3-70b-versatile ^
  --api-key <YOUR_API_KEY> ^
  --use-ragas ^
  --out-file evaluation/report.json
```

You can also run retrieval-only eval (omit provider/model/api-key).

Run full model-matrix benchmark (all configured UI models):

```bash
python evaluation/benchmark_model_matrix.py
```

This writes `evaluation/model_matrix_latest.json` with per-model metrics and failure reasons.
To include RAGAS for each model:

```bash
python evaluation/benchmark_model_matrix.py --use-ragas
```

### 2) Testing + Reliability
- Unit/integration tests in `tests/`
- LLM invocation retry with exponential backoff
- HTTP retry + timeout strategy in Streamlit client
- Stronger backend error handling and safer responses

Run tests:

```bash
pytest -q
```

### Current Verification Snapshot (2026-02-28)
- `ruff check .` passed
- `black --check .` passed
- `mypy .` passed
- `pytest -q` passed (`27 passed`)
- FastAPI smoke checks passed:
  - `GET /health -> 200`
  - `/chat` validation behavior for missing key -> `400` with clear message
  - `/upload` non-PDF validation -> `400` with clear message
- Streamlit headless startup smoke passed (`200` response)

Limitations of this local verification run:
- True end-to-end provider completion check is external-key dependent; local Moonshot key returned `401 Invalid Authentication` in smoke test.
- Docker build was not validated on this machine (Docker CLI unavailable locally).

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
- `MOONSHOT_API_KEY`

Operational controls:
- `CHAT_RATE_LIMIT_PER_MIN`
- `UPLOAD_RATE_LIMIT_PER_MIN`
- `LLM_RETRY_ATTEMPTS`
- `HTTP_CONNECT_TIMEOUT_SEC`
- `HTTP_READ_TIMEOUT_CHAT_SEC`
- `HTTP_READ_TIMEOUT_UPLOAD_SEC`
- `ENABLE_TOOLS`
- `AGENT_PLANNING_ENABLED`
- `ENABLE_WEB_SEARCH`
- `WEB_SEARCH_TIMEOUT_SEC`
- `WEB_SEARCH_MAX_RESULTS`
- `WEB_SEARCH_CANDIDATE_FACTOR`
- `WEB_SEARCH_MIN_RELEVANCE`
- `MAX_CALC_EXPRESSION_CHARS`

Cost estimation (optional):
- `DEFAULT_INPUT_COST_PER_1K_TOKENS`
- `DEFAULT_OUTPUT_COST_PER_1K_TOKENS`
- `USE_GLOBAL_DEFAULT_MODEL_PRICING`
- `MODEL_PRICING_OVERRIDES_JSON`

Notes:
- The backend prefers provider-reported token usage metadata when available, and only falls back to heuristic token estimates otherwise.
- Per-model pricing is preferred. Leave global defaults at `0` unless you explicitly want a single fallback price applied to unknown models.

## Deploy (GitHub -> Hugging Face)

Workflow: `.github/workflows/deploy-to-hf-space.yml`

On each push to `main`/`master`:
1. Validates required secrets
2. Pushes exact GitHub commit to HF Space `main`

Required GitHub repository secrets:
- `HF_TOKEN`: Hugging Face token with write access to the Space repo
- `HF_SPACE_ID`: `username/space-name`

## Failures + Tradeoffs

### Known Failure Modes
- Retrieval can return the right source file but still miss a specific detail chunk.
- In-memory rate limiting resets on restart and is not shared across replicas.
- Faithfulness score is heuristic and can overestimate factual grounding.
- Provider/network timeouts can still happen under external API instability.
- Web search quality depends on external search API availability and returned snippets.
- Invalid provider credentials return upstream auth errors (for example, `401 Invalid Authentication`).

### Mitigations
- Use top-k retrieval + source display + route controls (`chat_only` / `rag_only`) for debugging.
- Add retries with exponential backoff for model calls and HTTP client requests.
- Enforce strict input/file limits and payload validation to reduce bad requests.
- Expose runtime metrics and structured logs to detect latency/error regressions quickly.

### Tradeoffs
- Simple observability and rate limiting keep deployment lightweight but are less robust than distributed stacks.
- Cost telemetry is estimate-based unless provider-side token usage metadata is available.
- Current benchmark is small and domain-focused; broader generalization needs a larger test set.
