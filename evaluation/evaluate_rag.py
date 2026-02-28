# ruff: noqa: E402

import argparse
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, cast

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from rag_utility import _get_relevant_docs, answer_question_with_agent, get_context_and_sources, get_embedding_model

try:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy as ragas_answer_relevancy_metric
    from ragas.metrics import faithfulness as ragas_faithfulness_metric

    RAGAS_AVAILABLE = True
except Exception:
    RAGAS_AVAILABLE = False


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {idx}: {exc}") from exc
    if not rows:
        raise ValueError("Benchmark is empty.")
    return rows


def make_llm(provider: str, model: str, api_key: str, is_nvidia_key: bool):
    if provider == "Groq":
        return ChatGroq(model=model, api_key=cast(Any, api_key))
    if provider == "Moonshot Kimi":
        base_url = "https://integrate.api.nvidia.com/v1" if is_nvidia_key else "https://api.moonshot.cn/v1"
        return ChatOpenAI(model=model, api_key=cast(Any, api_key), base_url=base_url)
    raise ValueError(f"Unsupported provider: {provider}")


def normalize_source_name(name: str) -> str:
    return os.path.basename((name or "").strip()).lower()


def evaluate_retrieval(query: str, vector_db_path: str, expected_sources: List[str], top_k: int):
    docs = _get_relevant_docs(vector_db_path, query, k=top_k)
    got_sources = [normalize_source_name(doc.metadata.get("source", "unknown")) for doc in docs]
    expected = [normalize_source_name(source) for source in expected_sources]

    hit = any(source in got_sources for source in expected)
    mrr = 0.0
    for rank, source in enumerate(got_sources, start=1):
        if source in expected:
            mrr = 1.0 / rank
            break

    precision_at_k = 0.0
    if got_sources:
        matches = sum(1 for source in got_sources if source in expected)
        precision_at_k = matches / len(got_sources)

    return {
        "retrieved_sources": got_sources,
        "retrieval_hit": hit,
        "mrr": round(mrr, 4),
        "precision_at_k": round(precision_at_k, 4),
    }


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def faithfulness_score(answer: str, context: str) -> float:
    if not answer or not context:
        return 0.0

    context_tokens = set(tokenize(context))
    if not context_tokens:
        return 0.0

    answer = re.sub(r"(?im)^sources:\s*.*$", "", answer).strip()
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    checked = 0
    supported = 0
    for sentence in sentences:
        sentence_tokens = tokenize(sentence)
        if len(sentence_tokens) < 4:
            continue
        checked += 1
        overlap = sum(1 for token in sentence_tokens if token in context_tokens)
        if (overlap / max(len(sentence_tokens), 1)) >= 0.45:
            supported += 1

    if checked == 0:
        return 0.0
    return round(supported / checked, 4)


def keyword_recall(answer: str, required_keywords: List[str]) -> float:
    if not required_keywords:
        return 0.0
    answer_lower = (answer or "").lower()
    hits = sum(1 for keyword in required_keywords if keyword.lower() in answer_lower)
    return round(hits / len(required_keywords), 4)


def compute_ragas_scores(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    llm: Any,
) -> Dict[str, Any]:
    if not RAGAS_AVAILABLE:
        raise RuntimeError("RAGAS dependencies are not installed. Add ragas and datasets to requirements.")

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
    )

    ragas_result = ragas_evaluate(
        dataset=dataset,
        metrics=[ragas_faithfulness_metric, ragas_answer_relevancy_metric],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(get_embedding_model()),
    )

    records = ragas_result.to_pandas().to_dict(orient="records")
    faithfulness_vals = [float(row.get("faithfulness", 0.0) or 0.0) for row in records]
    answer_relevancy_vals = [float(row.get("answer_relevancy", 0.0) or 0.0) for row in records]

    return {
        "faithfulness_avg": round(mean(faithfulness_vals), 4) if faithfulness_vals else None,
        "answer_relevancy_avg": round(mean(answer_relevancy_vals), 4) if answer_relevancy_vals else None,
        "rows": [
            {
                "ragas_faithfulness": round(float(row.get("faithfulness", 0.0) or 0.0), 4),
                "ragas_answer_relevancy": round(float(row.get("answer_relevancy", 0.0) or 0.0), 4),
            }
            for row in records
        ],
    }


def run_evaluation(args):
    benchmark = load_benchmark(args.benchmark_file)
    llm = None
    if args.provider and args.model and args.api_key:
        llm = make_llm(args.provider, args.model, args.api_key, args.is_nvidia_key)
    if args.use_ragas and not llm:
        raise ValueError("--use-ragas requires --provider, --model, and --api-key.")

    rows = []
    retrieval_hits = []
    retrieval_mrr = []
    retrieval_precision = []
    faithfulness_values = []
    keyword_values = []
    ragas_questions: List[str] = []
    ragas_answers: List[str] = []
    ragas_contexts: List[List[str]] = []

    for item in benchmark:
        question = item["question"]
        expected_sources = item.get("expected_sources", [])
        required_keywords = item.get("required_keywords", [])

        retrieval = evaluate_retrieval(question, args.vector_db_path, expected_sources, args.top_k)

        record: Dict[str, Any] = {
            "id": item.get("id", question[:32]),
            "question": question,
            **retrieval,
        }

        retrieval_hits.append(1.0 if retrieval["retrieval_hit"] else 0.0)
        retrieval_mrr.append(retrieval["mrr"])
        retrieval_precision.append(retrieval["precision_at_k"])

        if llm:
            answer = answer_question_with_agent(question, llm, args.vector_db_path) or ""
            context, _ = get_context_and_sources(args.vector_db_path, question, k=args.top_k)
            faith = faithfulness_score(answer, context)
            kw = keyword_recall(answer, required_keywords)
            record["answer"] = answer
            record["faithfulness"] = faith
            record["keyword_recall"] = kw
            faithfulness_values.append(faith)
            keyword_values.append(kw)

            if args.use_ragas:
                context_chunks = [chunk for chunk in context.split("\n\n---\n\n") if chunk.strip()]
                if not context_chunks:
                    context_chunks = [context] if context else [""]
                ragas_questions.append(question)
                ragas_answers.append(answer)
                ragas_contexts.append(context_chunks)

        rows.append(record)

    summary = {
        "samples": len(rows),
        "retrieval_hit_rate": round(mean(retrieval_hits), 4) if retrieval_hits else 0.0,
        "retrieval_mrr": round(mean(retrieval_mrr), 4) if retrieval_mrr else 0.0,
        "retrieval_precision_at_k": round(mean(retrieval_precision), 4) if retrieval_precision else 0.0,
        "faithfulness_avg": round(mean(faithfulness_values), 4) if faithfulness_values else None,
        "keyword_recall_avg": round(mean(keyword_values), 4) if keyword_values else None,
        "ragas_faithfulness_avg": None,
        "ragas_answer_relevancy_avg": None,
    }

    ragas_summary: Optional[Dict[str, Any]] = None
    if args.use_ragas and ragas_questions:
        ragas_summary = compute_ragas_scores(ragas_questions, ragas_answers, ragas_contexts, llm)
        summary["ragas_faithfulness_avg"] = ragas_summary["faithfulness_avg"]
        summary["ragas_answer_relevancy_avg"] = ragas_summary["answer_relevancy_avg"]

        for row, ragas_row in zip(rows, ragas_summary["rows"]):
            row.update(ragas_row)

    output = {"summary": summary, "rows": rows}
    if ragas_summary:
        output["ragas"] = {
            "faithfulness_avg": ragas_summary["faithfulness_avg"],
            "answer_relevancy_avg": ragas_summary["answer_relevancy_avg"],
        }

    print(json.dumps(summary, indent=2))
    if args.out_file:
        with open(args.out_file, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
        print(f"\nSaved detailed report to: {args.out_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality, faithfulness, and optional RAGAS.")
    parser.add_argument("--vector-db-path", required=True, help="Path to a generated Chroma vector DB directory.")
    parser.add_argument("--benchmark-file", default="evaluation/benchmark.jsonl", help="JSONL benchmark file.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K docs for retrieval evaluation.")
    parser.add_argument("--provider", default="", help="Provider to run answer generation (optional).")
    parser.add_argument("--model", default="", help="Model to run answer generation (optional).")
    parser.add_argument("--api-key", default="", help="API key for provider (optional).")
    parser.add_argument("--is-nvidia-key", action="store_true", help="Use NVIDIA endpoint for Moonshot models.")
    parser.add_argument("--use-ragas", action="store_true", help="Compute RAGAS faithfulness and answer relevancy.")
    parser.add_argument("--out-file", default="", help="Optional JSON report output path.")
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
