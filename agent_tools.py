import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import requests

try:
    from duckduckgo_search import DDGS

    DDGS_AVAILABLE = True
except Exception:
    DDGS_AVAILABLE = False

WEB_SEARCH_TIMEOUT_SEC = float(os.getenv("WEB_SEARCH_TIMEOUT_SEC", "8"))
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "1").strip().lower() not in {"0", "false", "off"}
AGENT_PLANNING_ENABLED = os.getenv("AGENT_PLANNING_ENABLED", "1").strip().lower() not in {"0", "false", "off"}
MAX_CALC_EXPRESSION_CHARS = int(os.getenv("MAX_CALC_EXPRESSION_CHARS", "120"))
WEB_SEARCH_CANDIDATE_FACTOR = int(os.getenv("WEB_SEARCH_CANDIDATE_FACTOR", "3"))

SEARCH_HINTS = (
    "search",
    "look up",
    "lookup",
    "find online",
    "on the web",
    "on internet",
    "internet",
    "latest",
    "today",
    "current",
    "recent",
    "news",
)

MATH_HINTS = (
    "calculate",
    "calc",
    "compute",
    "solve",
    "what is",
    "evaluate",
)

_ALLOWED_BINARY_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_ALLOWED_UNARY_OPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}
SEARCH_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "what",
    "which",
    "when",
    "where",
    "who",
    "whom",
    "about",
    "latest",
    "current",
    "best",
    "model",
    "models",
    "tell",
    "me",
}
GENERIC_WEB_QUERIES = {
    "search",
    "search web",
    "web search",
    "search the web",
    "look up",
    "lookup",
    "web",
    "internet",
}


@dataclass
class AgentAction:
    tool: str
    tool_input: str
    reason: str


def _extract_math_expression(query: str) -> str:
    text = (query or "").strip()
    if not text:
        return ""

    candidate = text
    lower_text = text.lower()
    prefix_patterns = (
        r"^(?:what is|calculate|calc|compute|solve|evaluate)\s*",
        r"^(?:please\s+)?",
    )
    for pattern in prefix_patterns:
        candidate = re.sub(pattern, "", candidate, flags=re.IGNORECASE).strip()

    # Capture arithmetic-only segments from mixed natural-language prompts.
    if not re.fullmatch(r"[\d\s\+\-\*\/\%\(\)\.\^]+", candidate):
        segment = re.search(r"[\d\.\s\+\-\*\/\%\(\)\^]{3,}", lower_text)
        candidate = segment.group(0).strip() if segment else ""

    candidate = candidate.replace("^", "**")
    if len(candidate) > MAX_CALC_EXPRESSION_CHARS:
        return ""
    return candidate


def _safe_eval_math(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPS:
        left = _safe_eval_math(node.left)
        right = _safe_eval_math(node.right)
        return _ALLOWED_BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        val = _safe_eval_math(node.operand)
        return _ALLOWED_UNARY_OPS[type(node.op)](val)
    raise ValueError("Unsupported expression.")


def run_calculator_tool(expression: str) -> str:
    expr = _extract_math_expression(expression)
    if not expr:
        return "Calculator could not find a valid arithmetic expression."

    try:
        parsed = ast.parse(expr, mode="eval")
        value = _safe_eval_math(parsed)
        if abs(value - int(value)) < 1e-10:
            return str(int(value))
        return f"{value:.8f}".rstrip("0").rstrip(".")
    except Exception:
        return "Calculator could not evaluate that expression safely."


def _flatten_related_topics(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in items:
        text = item.get("Text")
        url = item.get("FirstURL", "")
        if text:
            rows.append({"text": str(text), "url": str(url)})
            continue
        subtopics = item.get("Topics")
        if isinstance(subtopics, list):
            for sub in subtopics:
                if isinstance(sub, dict) and sub.get("Text"):
                    rows.append({"text": str(sub["Text"]), "url": str(sub.get("FirstURL", ""))})
    return rows


def _tokenize(text: str) -> list[str]:
    normalized = (text or "").lower().replace("open ai", "openai")
    return re.findall(r"[a-z0-9]+", normalized)


def _query_terms(query: str) -> list[str]:
    terms = []
    for token in _tokenize(query):
        if len(token) <= 2:
            continue
        if token in SEARCH_STOPWORDS:
            continue
        terms.append(token)
    return terms


def _relevance_score(query_terms: list[str], title: str, snippet: str, url: str) -> float:
    if not query_terms:
        return 0.0
    haystack = " ".join([title or "", snippet or "", url or ""]).lower()
    hits = sum(1 for term in query_terms if term in haystack)
    return hits / len(query_terms)


def _is_generic_web_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip().lower())
    return q in GENERIC_WEB_QUERIES


def _extract_last_user_query(chat_history_context: str) -> str:
    if not chat_history_context:
        return ""
    candidates = re.findall(r"(?im)^user:\s*(.+)$", chat_history_context)
    if not candidates:
        return ""
    return candidates[-1].strip()


def _resolve_web_query(tool_input: str, user_query: str, chat_history_context: str) -> str:
    query = (tool_input or "").strip() or (user_query or "").strip()
    if not _is_generic_web_query(query):
        return query

    last_user = _extract_last_user_query(chat_history_context)
    if last_user and not _is_generic_web_query(last_user):
        return last_user
    return query


def _search_via_ddgs(cleaned_query: str) -> list[dict[str, str]]:
    if not DDGS_AVAILABLE:
        return []

    try:
        with DDGS() as ddgs:
            raw_rows = list(
                ddgs.text(
                    cleaned_query,
                    max_results=max(WEB_SEARCH_MAX_RESULTS * WEB_SEARCH_CANDIDATE_FACTOR, WEB_SEARCH_MAX_RESULTS),
                )
            )
    except Exception:
        return []

    rows: list[dict[str, str]] = []
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        snippet = str(item.get("body", "")).strip()
        url = str(item.get("href", "")).strip()
        if not (title or snippet or url):
            continue
        rows.append({"title": title, "snippet": snippet, "url": url})
    return rows


def _search_via_instant_api(cleaned_query: str) -> list[dict[str, str]]:
    params = {
        "q": cleaned_query,
        "format": "json",
        "no_redirect": "1",
        "skip_disambig": "1",
        "no_html": "1",
    }

    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params=params,
            timeout=WEB_SEARCH_TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    rows: list[dict[str, str]] = []
    answer = (payload.get("Answer") or "").strip()
    abstract = (payload.get("AbstractText") or "").strip()
    abstract_url = (payload.get("AbstractURL") or "").strip()
    if answer:
        rows.append({"title": "Answer", "snippet": answer, "url": abstract_url})
    if abstract:
        rows.append({"title": "Summary", "snippet": abstract, "url": abstract_url})

    for row in _flatten_related_topics(payload.get("RelatedTopics", [])):
        rows.append({"title": "Related", "snippet": row["text"], "url": row["url"]})
    return rows


def _rank_search_results(query: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    query_terms = _query_terms(query)
    scored_rows: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        url = (row.get("url") or "").strip()
        if "/c/" in url and "duckduckgo.com" in url:
            continue
        score = _relevance_score(
            query_terms=query_terms,
            title=row.get("title", ""),
            snippet=row.get("snippet", ""),
            url=url,
        )
        scored_rows.append((score, row))

    scored_rows.sort(key=lambda item: item[0], reverse=True)
    if query_terms:
        scored_rows = [item for item in scored_rows if item[0] >= 0.2]

    deduped: list[dict[str, str]] = []
    seen = set()
    for _, row in scored_rows:
        url = (row.get("url") or "").strip()
        if not url:
            continue
        parsed = urlparse(url)
        key = f"{parsed.netloc}{parsed.path}".lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= WEB_SEARCH_MAX_RESULTS:
            break
    return deduped


def run_web_search_tool(query: str) -> tuple[str, list[str]]:
    if not ENABLE_WEB_SEARCH:
        return ("Web search is disabled by configuration.", [])

    cleaned = (query or "").strip()
    if not cleaned:
        return ("Web search requires a non-empty query.", [])

    ddgs_rows = _search_via_ddgs(cleaned)
    fallback_rows = _search_via_instant_api(cleaned) if not ddgs_rows else []
    ranked_rows = _rank_search_results(cleaned, ddgs_rows or fallback_rows)
    if not ranked_rows and ddgs_rows and fallback_rows:
        ranked_rows = _rank_search_results(cleaned, fallback_rows)

    if not ranked_rows:
        return ("No high-confidence web results were returned for that query.", [])

    lines = []
    source_urls = []
    for index, row in enumerate(ranked_rows, start=1):
        title = row.get("title", "").strip() or "Result"
        snippet = re.sub(r"\s+", " ", row.get("snippet", "").strip())
        snippet = snippet[:260]
        lines.append(f"{index}. {title}: {snippet}")
        source_urls.append(row.get("url", "").strip())

    return ("\n".join(lines), source_urls)


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None

    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def _heuristic_action(query: str) -> AgentAction:
    q = (query or "").strip()
    lower_q = q.lower()

    expression = _extract_math_expression(q)
    if expression and (any(hint in lower_q for hint in MATH_HINTS) or re.fullmatch(r"[\d\.\s\+\-\*\/\%\(\)\^]+", q)):
        return AgentAction(tool="calculator", tool_input=expression, reason="Detected arithmetic expression.")

    if any(hint in lower_q for hint in SEARCH_HINTS):
        cleaned = re.sub(r"^(search|look up|lookup)\s+", "", q, flags=re.IGNORECASE).strip()
        return AgentAction(tool="web_search", tool_input=cleaned or q, reason="Detected web-search intent.")

    return AgentAction(tool="none", tool_input="", reason="No clear tool intent.")


def _llm_planned_action(query: str, llm_instance, chat_history_context: str = "") -> AgentAction:
    if not AGENT_PLANNING_ENABLED:
        return AgentAction(tool="none", tool_input="", reason="Planning disabled.")

    planner_prompt = (
        "You are a tool router.\n"
        "Choose exactly one tool for the user query.\n"
        "Allowed tools: none, calculator, web_search.\n"
        "Use calculator only for arithmetic. Use web_search for latest/current/news/internet lookup.\n"
        "Return strict JSON only with keys: tool, tool_input, reason.\n\n"
        f"Conversation history:\n{chat_history_context or '[none]'}\n\n"
        f"User query: {query}"
    )

    try:
        raw = llm_instance.invoke(planner_prompt)
        text = getattr(raw, "content", str(raw))
        data = _extract_json_object(text) or {}
        tool = str(data.get("tool", "none")).strip().lower()
        tool_input = str(data.get("tool_input", "")).strip()
        reason = str(data.get("reason", "")).strip() or "LLM-selected tool."
        if tool not in {"none", "calculator", "web_search"}:
            return AgentAction(tool="none", tool_input="", reason="Planner returned unsupported tool.")
        if tool == "calculator":
            tool_input = tool_input or _extract_math_expression(query)
        if tool == "web_search":
            tool_input = tool_input or query
        return AgentAction(tool=tool, tool_input=tool_input, reason=reason)
    except Exception:
        return AgentAction(tool="none", tool_input="", reason="Planner failed.")


def choose_agent_action(query: str, llm_instance, chat_history_context: str = "") -> AgentAction:
    heuristic = _heuristic_action(query)
    if heuristic.tool != "none":
        return heuristic

    planned = _llm_planned_action(query, llm_instance, chat_history_context=chat_history_context)
    if planned.tool != "none":
        return planned

    return heuristic


def run_agent_with_tools(query: str, llm_instance, chat_history_context: str = "") -> Optional[dict[str, Any]]:
    action = choose_agent_action(query, llm_instance, chat_history_context=chat_history_context)
    if action.tool == "none":
        return None

    tool_result = ""
    source_urls: list[str] = []
    if action.tool == "calculator":
        tool_result = run_calculator_tool(action.tool_input)
    elif action.tool == "web_search":
        resolved_web_query = _resolve_web_query(action.tool_input, query, chat_history_context)
        tool_result, source_urls = run_web_search_tool(resolved_web_query)
    else:
        return None

    synthesis_prompt = (
        "You are a helpful assistant.\n"
        "Use the tool output below to answer the user's question accurately.\n"
        "If the tool output says it failed or has no results, be transparent.\n"
        "Keep the answer concise and practical.\n\n"
        f"Conversation history:\n{chat_history_context or '[none]'}\n\n"
        f"User question:\n{query}\n\n"
        f"Tool used: {action.tool}\n"
        f"Tool output:\n{tool_result}\n"
    )

    final = llm_instance.invoke(synthesis_prompt)
    response = str(getattr(final, "content", final)).strip()

    if action.tool == "web_search" and source_urls and "sources:" not in response.lower():
        source_list = ", ".join(source_urls[:3])
        response = f"{response}\n\nSources: {source_list}"

    return {
        "response": response,
        "tool_used": action.tool,
        "tool_input": action.tool_input,
        "tool_reason": action.reason,
    }
