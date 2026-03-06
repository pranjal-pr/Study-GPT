# ruff: noqa: E402

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from api import app
from evaluation.evaluate_rag import (
    compute_ragas_scores,
    evaluate_retrieval,
    faithfulness_score,
    keyword_recall,
    load_benchmark,
    make_llm,
)
from rag_utility import answer_question_with_agent, get_context_and_sources

ENV_PATH = ROOT / ".env"
BENCHMARK_PATH = ROOT / "evaluation" / "benchmark.jsonl"
OUTPUT_PATH = ROOT / "evaluation" / "model_matrix_latest.json"


def parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def pick_latest_vector_db() -> Path:
    candidates = [p for p in ROOT.glob("vector_db_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError("No vector_db_* directory found. Upload/process docs first.")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def build_model_matrix(env_values: Dict[str, str]) -> List[Dict[str, Any]]:
    matrix = [
        {"provider": "Groq", "model": "llama-3.3-70b-versatile", "env_key": "GROQ_API_KEY", "is_nvidia_key": False},
        {"provider": "Groq", "model": "llama-3.1-8b-instant", "env_key": "GROQ_API_KEY", "is_nvidia_key": False},
    ]

    moonshot_key = env_values.get("MOONSHOT_API_KEY", "")
    if moonshot_key.startswith("nvapi-"):
        matrix.extend(
            [
                {
                    "provider": "Moonshot Kimi",
                    "model": "moonshotai/kimi-k2-thinking",
                    "env_key": "MOONSHOT_API_KEY",
                    "is_nvidia_key": True,
                },
            ]
        )
    else:
        matrix.extend(
            [
                {
                    "provider": "Moonshot Kimi",
                    "model": "moonshot-v1-8k",
                    "env_key": "MOONSHOT_API_KEY",
                    "is_nvidia_key": False,
                },
                {
                    "provider": "Moonshot Kimi",
                    "model": "moonshot-v1-32k",
                    "env_key": "MOONSHOT_API_KEY",
                    "is_nvidia_key": False,
                },
            ]
        )
    return matrix


def run_retrieval_and_faithfulness(
    provider: str,
    model: str,
    api_key: str,
    is_nvidia_key: bool,
    vector_db_path: str,
    benchmark_rows: List[Dict[str, Any]],
    top_k: int,
    use_ragas: bool,
) -> Dict[str, Any]:
    llm = make_llm(provider, model, api_key, is_nvidia_key)

    retrieval_hits: List[float] = []
    retrieval_mrr: List[float] = []
    retrieval_precision: List[float] = []
    faithfulness_values: List[float] = []
    keyword_values: List[float] = []
    ragas_questions: List[str] = []
    ragas_answers: List[str] = []
    ragas_contexts: List[List[str]] = []

    for item in benchmark_rows:
        question = item["question"]
        expected_sources = item.get("expected_sources", [])
        required_keywords = item.get("required_keywords", [])

        retrieval = evaluate_retrieval(question, vector_db_path, expected_sources, top_k)
        retrieval_hits.append(1.0 if retrieval["retrieval_hit"] else 0.0)
        retrieval_mrr.append(retrieval["mrr"])
        retrieval_precision.append(retrieval["precision_at_k"])

        answer = answer_question_with_agent(question, llm, vector_db_path) or ""
        context, _ = get_context_and_sources(vector_db_path, question, k=top_k)
        faithfulness_values.append(faithfulness_score(answer, context))
        keyword_values.append(keyword_recall(answer, required_keywords))
        if use_ragas:
            context_chunks = [chunk for chunk in context.split("\n\n---\n\n") if chunk.strip()]
            if not context_chunks:
                context_chunks = [context] if context else [""]
            ragas_questions.append(question)
            ragas_answers.append(answer)
            ragas_contexts.append(context_chunks)

    ragas_faithfulness_avg = None
    ragas_answer_relevancy_avg = None
    if use_ragas and ragas_questions:
        ragas_summary = compute_ragas_scores(ragas_questions, ragas_answers, ragas_contexts, llm)
        ragas_faithfulness_avg = ragas_summary["faithfulness_avg"]
        ragas_answer_relevancy_avg = ragas_summary["answer_relevancy_avg"]

    return {
        "samples": len(benchmark_rows),
        "retrieval_hit_rate": round(mean(retrieval_hits), 4) if retrieval_hits else 0.0,
        "retrieval_mrr": round(mean(retrieval_mrr), 4) if retrieval_mrr else 0.0,
        "retrieval_precision_at_k": round(mean(retrieval_precision), 4) if retrieval_precision else 0.0,
        "faithfulness_avg": round(mean(faithfulness_values), 4) if faithfulness_values else None,
        "keyword_recall_avg": round(mean(keyword_values), 4) if keyword_values else None,
        "ragas_faithfulness_avg": ragas_faithfulness_avg,
        "ragas_answer_relevancy_avg": ragas_answer_relevancy_avg,
    }


def run_runtime_benchmark(
    provider: str,
    model: str,
    api_key: str,
    is_nvidia_key: bool,
    prompts: List[str],
) -> Dict[str, Any]:
    client = TestClient(app)
    latencies: List[float] = []
    failures = 0
    records: List[Dict[str, Any]] = []

    for prompt in prompts:
        payload = {
            "query": prompt,
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "routing_mode": "chat_only",
            "is_nvidia_key": is_nvidia_key,
            "chat_history": [],
        }

        started = time.perf_counter()
        response = client.post("/chat", json=payload)
        wall_ms = (time.perf_counter() - started) * 1000

        if response.status_code == 200:
            data = response.json()
            latency_ms = float(data.get("metrics", {}).get("latency_ms", wall_ms))
            latencies.append(latency_ms)
            records.append({"status": 200, "latency_ms": round(latency_ms, 2), "wall_ms": round(wall_ms, 2)})
        else:
            failures += 1
            records.append(
                {
                    "status": response.status_code,
                    "latency_ms": None,
                    "wall_ms": round(wall_ms, 2),
                    "detail": (response.text or "")[:250],
                }
            )

    error_rate = (failures / len(prompts)) if prompts else 0.0
    if latencies:
        sorted_latencies = sorted(latencies)
        p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
        p95_ms = sorted_latencies[p95_index]
        avg_ms = mean(sorted_latencies)
    else:
        p95_ms = 0.0
        avg_ms = 0.0

    return {
        "samples": len(prompts),
        "successes": len(prompts) - failures,
        "failures": failures,
        "error_rate": round(error_rate, 4),
        "avg_latency_ms": round(avg_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "records": records,
    }


def to_markdown_table(rows: List[Dict[str, Any]]) -> str:
    header = (
        "| Provider | Model | Status | Hit@3 | MRR@3 | Faithfulness | RAGAS AnsRel | p95 Latency (ms) | Error Rate |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        if row["status"] == "ok":
            eval_summary = row.get("evaluation", {})
            run_summary = row.get("runtime", {})
            lines.append(
                "| {provider} | {model} | {status} | {hit:.2f} | {mrr:.2f} | {faith:.2f} | {ansrel:.2f} | {p95:.2f} | {err:.2%} |".format(
                    provider=row["provider"],
                    model=row["model"],
                    status=row["status"],
                    hit=eval_summary.get("retrieval_hit_rate", 0.0),
                    mrr=eval_summary.get("retrieval_mrr", 0.0),
                    faith=eval_summary.get("faithfulness_avg", 0.0) or 0.0,
                    ansrel=eval_summary.get("ragas_answer_relevancy_avg", 0.0) or 0.0,
                    p95=run_summary.get("p95_latency_ms", 0.0),
                    err=run_summary.get("error_rate", 0.0),
                )
            )
        else:
            lines.append(
                "| {provider} | {model} | {status} | - | - | - | - | - | - |".format(
                    provider=row["provider"],
                    model=row["model"],
                    status=row["status"],
                )
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark all configured models for retrieval/latency metrics.")
    parser.add_argument("--use-ragas", action="store_true", help="Also compute RAGAS metrics per model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_values = parse_env_file(ENV_PATH)
    vector_db_path = str(pick_latest_vector_db())
    benchmark_rows = load_benchmark(str(BENCHMARK_PATH))
    model_matrix = build_model_matrix(env_values)

    prompts = [
        "Define machine learning in one sentence.",
        "What is overfitting? Keep it short.",
        "What is precision vs recall in one sentence?",
        "What is gradient descent in one sentence?",
        "What is regularization in one sentence?",
    ]

    results: List[Dict[str, Any]] = []

    for item in model_matrix:
        provider = item["provider"]
        model = item["model"]
        env_key = item["env_key"]
        is_nvidia_key = item["is_nvidia_key"]
        api_key = env_values.get(env_key) or os.getenv(env_key, "")

        row: Dict[str, Any] = {
            "provider": provider,
            "model": model,
            "env_key": env_key,
            "is_nvidia_key": is_nvidia_key,
            "status": "ok",
            "error": "",
        }

        if not api_key:
            row["status"] = "skipped_missing_key"
            row["error"] = f"Missing {env_key}"
            results.append(row)
            continue

        try:
            row["evaluation"] = run_retrieval_and_faithfulness(
                provider=provider,
                model=model,
                api_key=api_key,
                is_nvidia_key=is_nvidia_key,
                vector_db_path=vector_db_path,
                benchmark_rows=benchmark_rows,
                top_k=3,
                use_ragas=args.use_ragas,
            )
            row["runtime"] = run_runtime_benchmark(
                provider=provider,
                model=model,
                api_key=api_key,
                is_nvidia_key=is_nvidia_key,
                prompts=prompts,
            )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"

        results.append(row)

    output = {
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_file": str(BENCHMARK_PATH.relative_to(ROOT)),
        "vector_db_path": Path(vector_db_path).name,
        "runtime_prompt_count_per_model": len(prompts),
        "ragas_enabled": args.use_ragas,
        "results": results,
        "markdown_table": to_markdown_table(results),
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
