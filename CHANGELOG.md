# Changelog

## v1.0.0 - 2026-02-26

### Highlights
- Production-ready Streamlit + FastAPI RAG app deployed to Hugging Face Space.
- Model routing (`auto`, `chat_only`, `rag_only`) with conversation-memory support.
- Black/grey UI redesign with ShinzoGPT branding and architecture docs.

### Reliability and quality
- Added retries, timeouts, and request validation.
- Added observability metrics endpoints and runtime dashboard.
- Added CI test workflow and quality gates (ruff, black, mypy, pytest).
- Added automated secret scanning workflow (Gitleaks).

### Evaluation and benchmarking
- Added retrieval and faithfulness benchmark tooling.
- Added full model-matrix benchmark report with per-model latency/error metrics.
- Published reproducible reports in `evaluation/`.

### Documentation
- Added measured results table in README with explicit model statuses.
- Added architecture diagrams (Mermaid + static fallback).
- Added security notes and incident-response checklist.
