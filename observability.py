import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List

LOGGER_NAME = "shinzogpt"
DEFAULT_INPUT_COST_PER_1K = float(os.getenv("DEFAULT_INPUT_COST_PER_1K_TOKENS", "0.00059"))
DEFAULT_OUTPUT_COST_PER_1K = float(os.getenv("DEFAULT_OUTPUT_COST_PER_1K_TOKENS", "0.00079"))
_MODEL_PRICING_OVERRIDES_ENV = os.getenv(
    "MODEL_PRICING_OVERRIDES_JSON",
    (
        '{"llama-3.3-70b-versatile":{"input_per_1k":0.00059,"output_per_1k":0.00079},'
        '"llama-3.1-8b-instant":{"input_per_1k":0.00005,"output_per_1k":0.00008}}'
    ),
)


def _parse_model_pricing_overrides() -> Dict[str, Dict[str, float]]:
    try:
        parsed = json.loads(_MODEL_PRICING_OVERRIDES_ENV)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {}


MODEL_PRICING_OVERRIDES = _parse_model_pricing_overrides()


def setup_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logging()


def log_event(event: str, **fields: Any) -> None:
    payload = {"ts": round(time.time(), 3), "event": event, **fields}
    try:
        logger.info(json.dumps(payload, ensure_ascii=True, default=str))
    except Exception:
        logger.info(f"{event} {fields}")


def estimate_tokens(text: str) -> int:
    clean_text = text or ""
    # Rough heuristic: ~4 chars per token for English-like text.
    return max(1, (len(clean_text) + 3) // 4)


def _resolve_model_pricing(model: str) -> tuple[float, float]:
    override = MODEL_PRICING_OVERRIDES.get(model, {})
    input_cost = float(override.get("input_per_1k", DEFAULT_INPUT_COST_PER_1K))
    output_cost = float(override.get("output_per_1k", DEFAULT_OUTPUT_COST_PER_1K))
    return input_cost, output_cost


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    input_per_1k, output_per_1k = _resolve_model_pricing(model)
    total = (input_tokens / 1000.0) * input_per_1k + (output_tokens / 1000.0) * output_per_1k
    return round(total, 8)


class MetricsStore:
    def __init__(self, max_events: int = 300):
        self._lock = threading.Lock()
        self._requests_total = 0
        self._errors_total = 0
        self._latency_sum_ms = 0.0
        self._endpoint_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "errors": 0.0, "latency_sum_ms": 0.0}
        )
        self._chat_total = 0
        self._upload_total = 0
        self._estimated_input_tokens_total = 0
        self._estimated_output_tokens_total = 0
        self._estimated_cost_usd_total = 0.0
        self._events: deque[Dict[str, Any]] = deque(maxlen=max_events)

    def record_request(self, endpoint: str, method: str, status_code: int, latency_ms: float) -> None:
        key = f"{method.upper()} {endpoint}"
        with self._lock:
            self._requests_total += 1
            self._latency_sum_ms += latency_ms
            self._endpoint_stats[key]["count"] += 1
            self._endpoint_stats[key]["latency_sum_ms"] += latency_ms
            if status_code >= 400:
                self._errors_total += 1
                self._endpoint_stats[key]["errors"] += 1

            self._events.appendleft(
                {
                    "ts": round(time.time(), 3),
                    "kind": "request",
                    "endpoint": endpoint,
                    "method": method.upper(),
                    "status_code": status_code,
                    "latency_ms": round(latency_ms, 2),
                }
            )

    def record_chat(
        self,
        provider: str,
        model: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
    ) -> None:
        with self._lock:
            self._chat_total += 1
            self._estimated_input_tokens_total += input_tokens
            self._estimated_output_tokens_total += output_tokens
            self._estimated_cost_usd_total += estimated_cost_usd
            self._events.appendleft(
                {
                    "ts": round(time.time(), 3),
                    "kind": "chat",
                    "provider": provider,
                    "model": model,
                    "latency_ms": round(latency_ms, 2),
                    "estimated_input_tokens": input_tokens,
                    "estimated_output_tokens": output_tokens,
                    "estimated_cost_usd": round(estimated_cost_usd, 8),
                }
            )

    def record_upload(self, files_count: int) -> None:
        with self._lock:
            self._upload_total += 1
            self._events.appendleft(
                {
                    "ts": round(time.time(), 3),
                    "kind": "upload",
                    "files_count": files_count,
                }
            )

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            avg_latency = self._latency_sum_ms / self._requests_total if self._requests_total else 0.0
            endpoints = {}
            for endpoint, stats in self._endpoint_stats.items():
                endpoint_avg = stats["latency_sum_ms"] / stats["count"] if stats["count"] else 0.0
                endpoints[endpoint] = {
                    "count": int(stats["count"]),
                    "errors": int(stats["errors"]),
                    "avg_latency_ms": round(endpoint_avg, 2),
                }

            return {
                "requests_total": self._requests_total,
                "errors_total": self._errors_total,
                "avg_request_latency_ms": round(avg_latency, 2),
                "chat_total": self._chat_total,
                "upload_total": self._upload_total,
                "estimated_input_tokens_total": self._estimated_input_tokens_total,
                "estimated_output_tokens_total": self._estimated_output_tokens_total,
                "estimated_cost_usd_total": round(self._estimated_cost_usd_total, 8),
                "endpoint_stats": endpoints,
            }

    def events(self, limit: int = 50) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(limit, len(self._events)))
        with self._lock:
            return list(self._events)[:safe_limit]


metrics_store = MetricsStore(max_events=int(os.getenv("OBSERVABILITY_MAX_EVENTS", "300")))
