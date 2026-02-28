import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import requests

WEB_SEARCH_TIMEOUT_SEC = float(os.getenv("WEB_SEARCH_TIMEOUT_SEC", "8"))
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "1").strip().lower() not in {"0", "false", "off"}
AGENT_PLANNING_ENABLED = os.getenv("AGENT_PLANNING_ENABLED", "1").strip().lower() not in {"0", "false", "off"}
MAX_CALC_EXPRESSION_CHARS = int(os.getenv("MAX_CALC_EXPRESSION_CHARS", "120"))

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


def run_web_search_tool(query: str) -> tuple[str, list[str]]:
    if not ENABLE_WEB_SEARCH:
        return ("Web search is disabled by configuration.", [])

    cleaned = (query or "").strip()
    if not cleaned:
        return ("Web search requires a non-empty query.", [])

    params = {
        "q": cleaned,
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
    except Exception as exc:
        return (f"Web search failed: {type(exc).__name__}", [])

    lines: list[str] = []
    source_urls: list[str] = []

    answer = (payload.get("Answer") or "").strip()
    abstract = (payload.get("AbstractText") or "").strip()
    abstract_url = (payload.get("AbstractURL") or "").strip()

    if answer:
        lines.append(f"Answer: {answer}")
    if abstract:
        lines.append(f"Summary: {abstract}")
    if abstract_url:
        source_urls.append(abstract_url)

    related = _flatten_related_topics(payload.get("RelatedTopics", []))
    for row in related[:WEB_SEARCH_MAX_RESULTS]:
        lines.append(f"- {row['text']}")
        if row["url"]:
            source_urls.append(row["url"])

    if not lines:
        return ("No high-confidence web results were returned.", [])

    deduped_sources = []
    seen = set()
    for url in source_urls:
        if url and url not in seen:
            deduped_sources.append(url)
            seen.add(url)

    return ("\n".join(lines), deduped_sources[:WEB_SEARCH_MAX_RESULTS])


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
        tool_result, source_urls = run_web_search_tool(action.tool_input)
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
