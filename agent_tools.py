import ast
import html
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, cast
from urllib.parse import parse_qs, unquote, urlparse
from xml.etree import ElementTree as ET

import requests

from observability import extract_usage_metrics

try:
    from ddgs import DDGS

    DDGS_AVAILABLE = True
except Exception:
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
WEB_SEARCH_MIN_RELEVANCE = float(os.getenv("WEB_SEARCH_MIN_RELEVANCE", "0.25"))
OPEN_METEO_GEOCODING_URL = os.getenv("OPEN_METEO_GEOCODING_URL", "https://geocoding-api.open-meteo.com/v1/search")
OPEN_METEO_FORECAST_URL = os.getenv("OPEN_METEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast")
GOOGLE_NEWS_TOP_RSS_URL = os.getenv("GOOGLE_NEWS_TOP_RSS_URL", "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en")
GOOGLE_NEWS_SEARCH_RSS_URL = os.getenv(
    "GOOGLE_NEWS_SEARCH_RSS_URL",
    "https://news.google.com/rss/search",
)
BBC_WORLD_RSS_URL = os.getenv("BBC_WORLD_RSS_URL", "https://feeds.bbci.co.uk/news/world/rss.xml")
TWELVE_DATA_PRICE_URL = os.getenv("TWELVE_DATA_PRICE_URL", "https://api.twelvedata.com/price")
COINGECKO_SIMPLE_PRICE_URL = os.getenv(
    "COINGECKO_SIMPLE_PRICE_URL",
    "https://api.coingecko.com/api/v3/simple/price",
)
COINBASE_SPOT_PRICE_URL = os.getenv("COINBASE_SPOT_PRICE_URL", "https://api.coinbase.com/v2/prices/{symbol}-USD/spot")
YAHOO_SEARCH_URL = os.getenv("YAHOO_SEARCH_URL", "https://search.yahoo.com/search")
JINA_MIRROR_PREFIX = os.getenv("JINA_MIRROR_PREFIX", "https://r.jina.ai/http://")
DEFAULT_HTTP_HEADERS = {"User-Agent": "ShinzoGPT/1.0 (+https://huggingface.co/spaces/shinzobolte/ShinzoGPT)"}
SEARCH_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

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
EXPLICIT_WEB_HINTS = (
    "search",
    "look up",
    "lookup",
    "find online",
    "on the web",
    "on internet",
    "internet",
    "web",
)
REFERENTIAL_TOKENS = {"it", "its", "this", "that", "those", "them", "previous", "last", "earlier", "same"}

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
TRUSTED_DOMAIN_BOOST = {
    "openai.com": 0.35,
    "help.openai.com": 0.35,
    "platform.openai.com": 0.35,
    "docs.anthropic.com": 0.2,
    "ai.google.dev": 0.2,
    "blog.google": 0.25,
    "deepmind.google": 0.25,
    "developers.googleblog.com": 0.2,
    "huggingface.co": 0.15,
    "arxiv.org": 0.15,
    "github.com": 0.1,
    "developer.nvidia.com": 0.1,
}
LOW_QUALITY_DOMAIN_PENALTY = {
    "hinative.com": -0.45,
    "quora.com": -0.35,
    "reddit.com": -0.15,
    "medium.com": -0.1,
    "pinterest.com": -0.4,
}
RECENCY_HINTS = {"latest", "current", "today", "recent", "newest", "news"}
TOOL_CONTROL_PATTERNS = (
    r"^(?:please\s+)?(?:use|try|enable)\s+(?:the\s+)?(?:web\s+search(?:\s+tools?)?|search(?:\s+tools?)?|tools?|tool)\b.*$",
    r"^(?:please\s+)?(?:search|look up|lookup)\s+(?:the\s+web\s+)?(?:for\s+)?(?:it|that|this)\s*$",
)
TIME_INTENT_PATTERNS = (
    r"\bcurrent\s+time\b",
    r"\btime\s+(?:in|at|for|of)\b",
    r"\bwhat(?:'s| is)\s+the\s+(?:current\s+)?time\b",
)
WEATHER_INTENT_PATTERNS = (
    r"\bcurrent\s+weather\b",
    r"\bweather\s+(?:in|at|for|of)\b",
    r"\bforecast\s+(?:in|at|for|of)\b",
    r"\btemperature\s+(?:in|at|for|of)\b",
    r"\bwhat(?:'s| is)\s+the\s+(?:current\s+)?weather\b",
)
TIME_LOCATION_PATTERNS = (
    r"\bcurrent\s+time\s+(?:in|at|for|of)\s+(?P<location>.+)$",
    r"\btime\s+(?:in|at|for|of)\s+(?P<location>.+)$",
    r"^(?P<location>.+?)\s+time$",
)
WEATHER_LOCATION_PATTERNS = (
    r"\b(?:current\s+)?weather\s+(?:in|at|for|of)\s+(?P<location>.+)$",
    r"\bforecast\s+(?:in|at|for|of)\s+(?P<location>.+)$",
    r"\btemperature\s+(?:in|at|for|of)\s+(?P<location>.+)$",
    r"^(?P<location>.+?)\s+weather$",
)
NEWS_INTENT_PATTERNS = (
    r"\bnews\b",
    r"\bheadline(?:s)?\b",
    r"\btop stories\b",
    r"\bbreaking news\b",
    r"\bworld news\b",
    r"\bglobal news\b",
)
ASSET_PRICE_HINTS = (
    "price",
    "worth",
    "trading",
    "traded",
    "stock",
    "share",
    "shares",
    "usd",
    "dollar",
    "dollars",
)
LOCATION_NOISE_TOKENS = {
    "ask",
    "can",
    "city",
    "current",
    "forecast",
    "give",
    "hello",
    "hey",
    "hi",
    "how",
    "me",
    "please",
    "show",
    "tell",
    "temperature",
    "time",
    "weather",
    "what",
    "whats",
    "where",
    "you",
}
WEATHER_CODE_LABELS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}
STOCK_ALIASES = {
    "aapl": ("AAPL", "Apple"),
    "apple": ("AAPL", "Apple"),
    "msft": ("MSFT", "Microsoft"),
    "microsoft": ("MSFT", "Microsoft"),
    "googl": ("GOOGL", "Alphabet"),
    "google": ("GOOGL", "Alphabet"),
    "alphabet": ("GOOGL", "Alphabet"),
    "amzn": ("AMZN", "Amazon"),
    "amazon": ("AMZN", "Amazon"),
    "meta": ("META", "Meta"),
    "meta platforms": ("META", "Meta"),
    "nvda": ("NVDA", "Nvidia"),
    "nvidia": ("NVDA", "Nvidia"),
    "tsla": ("TSLA", "Tesla"),
    "tesla": ("TSLA", "Tesla"),
}
CRYPTO_ALIASES = {
    "bitcoin": ("bitcoin", "BTC", "Bitcoin"),
    "btc": ("bitcoin", "BTC", "Bitcoin"),
    "ethereum": ("ethereum", "ETH", "Ethereum"),
    "eth": ("ethereum", "ETH", "Ethereum"),
    "solana": ("solana", "SOL", "Solana"),
    "sol": ("solana", "SOL", "Solana"),
    "dogecoin": ("dogecoin", "DOGE", "Dogecoin"),
    "doge": ("dogecoin", "DOGE", "Dogecoin"),
    "xrp": ("ripple", "XRP", "XRP"),
    "ripple": ("ripple", "XRP", "XRP"),
}


@dataclass
class AgentAction:
    tool: str
    tool_input: str
    reason: str


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _clean_location_text(text: str) -> str:
    cleaned = _normalize_space(text)
    cleaned = re.split(r"[?!]", cleaned, maxsplit=1)[0]
    cleaned = re.split(r"\b(?:what(?:'s| is)|how|tell me|give me)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    cleaned = re.sub(r"\b(?:right now|now|today|currently|current|please|pls)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[?.!,;:]+$", "", cleaned)
    return _normalize_space(cleaned).strip(" -")


def _is_plausible_location_text(text: str) -> bool:
    tokens = _tokenize(text)
    if not tokens:
        return False
    return not any(token in LOCATION_NOISE_TOKENS for token in tokens)


def _extract_location_from_patterns(query: str, patterns: tuple[str, ...]) -> str:
    text = _normalize_space(query)
    if not text:
        return ""

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        location = _clean_location_text(match.group("location"))
        if location and _is_plausible_location_text(location):
            return location
    return ""


def _extract_time_location(query: str) -> str:
    return _extract_location_from_patterns(query, TIME_LOCATION_PATTERNS)


def _extract_weather_location(query: str) -> str:
    return _extract_location_from_patterns(query, WEATHER_LOCATION_PATTERNS)


def _has_time_intent(query: str) -> bool:
    normalized = _normalize_space(query).lower()
    return any(re.search(pattern, normalized) for pattern in TIME_INTENT_PATTERNS)


def _has_weather_intent(query: str) -> bool:
    normalized = _normalize_space(query).lower()
    return any(re.search(pattern, normalized) for pattern in WEATHER_INTENT_PATTERNS)


def _resolve_location_input(query_or_location: str, extractor) -> str:
    location = extractor(query_or_location)
    if location:
        return location

    cleaned = _clean_location_text(query_or_location)
    if _is_plausible_location_text(cleaned):
        return cleaned
    return ""


def _has_news_intent(query: str) -> bool:
    normalized = _normalize_space(query).lower()
    return any(re.search(pattern, normalized) for pattern in NEWS_INTENT_PATTERNS)


def _extract_stock_match(query: str) -> Optional[tuple[str, str]]:
    normalized = _normalize_space(query).lower()
    for alias, match in STOCK_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return match
    return None


def _extract_crypto_match(query: str) -> Optional[tuple[str, str, str]]:
    normalized = _normalize_space(query).lower()
    for alias, match in CRYPTO_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return match
    return None


def _has_asset_price_intent(query: str) -> bool:
    normalized = _normalize_space(query).lower()
    if not any(hint in normalized for hint in ASSET_PRICE_HINTS):
        return False
    return _extract_stock_match(query) is not None or _extract_crypto_match(query) is not None


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


def _format_location_name(place: dict[str, Any]) -> str:
    parts = []
    for key in ("name", "admin1", "country"):
        value = str(place.get(key, "")).strip()
        if value and value not in parts:
            parts.append(value)
    return ", ".join(parts)


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.1f}".rstrip("0").rstrip(".")


def _geocode_location(location: str) -> Optional[dict[str, Any]]:
    cleaned = _clean_location_text(location)
    if not cleaned:
        return None

    def _fetch_geocode_results(name: str) -> list[dict[str, Any]]:
        try:
            response = requests.get(
                OPEN_METEO_GEOCODING_URL,
                params={
                    "name": name,
                    "count": 5,
                    "language": "en",
                    "format": "json",
                },
                headers=DEFAULT_HTTP_HEADERS,
                timeout=WEB_SEARCH_TIMEOUT_SEC,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []
        results = payload.get("results") or []
        return [result for result in results if isinstance(result, dict)]

    results = _fetch_geocode_results(cleaned)
    if not results and "," in cleaned:
        results = _fetch_geocode_results(cleaned.split(",", 1)[0].strip())
    if not results:
        return None

    normalized_cleaned = cleaned.lower()
    best = max(
        results,
        key=lambda row: (
            str(row.get("name", "")).strip().lower() == normalized_cleaned,
            row.get("population") or 0,
        ),
    )
    if not isinstance(best, dict):
        return None
    return best


def _fetch_open_meteo_current(place: dict[str, Any], current_fields: list[str]) -> Optional[dict[str, Any]]:
    latitude = place.get("latitude")
    longitude = place.get("longitude")
    if latitude is None or longitude is None:
        return None

    try:
        response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": ",".join(current_fields),
                "forecast_days": 1,
                "timezone": "auto",
            },
            timeout=WEB_SEARCH_TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    if not isinstance(payload, dict) or not isinstance(payload.get("current"), dict):
        return None
    return payload


def run_current_time_tool(query_or_location: str) -> str:
    location = _resolve_location_input(query_or_location, _extract_time_location)
    if not location:
        return "Time lookup requires a city or location."

    place = _geocode_location(location)
    if not place:
        return f"I couldn't find a reliable location match for '{location}'."

    payload = _fetch_open_meteo_current(place, ["is_day"])
    if not payload:
        return f"I couldn't retrieve the current local time for {_format_location_name(place)}."

    current_time = str(payload.get("current", {}).get("time", "")).strip()
    if not current_time:
        return f"I couldn't retrieve the current local time for {_format_location_name(place)}."

    try:
        parsed_time = datetime.fromisoformat(current_time)
        formatted_time = parsed_time.strftime("%I:%M %p").lstrip("0")
        formatted_date = parsed_time.strftime("%B %d, %Y")
    except ValueError:
        formatted_time = current_time
        formatted_date = ""

    date_suffix = f" on {formatted_date}" if formatted_date else ""
    timezone = str(payload.get("timezone") or place.get("timezone") or "").strip()
    timezone_suffix = f" ({timezone})" if timezone else ""
    return (
        f"Current local time in {_format_location_name(place)}: "
        f"{formatted_time}{date_suffix}{timezone_suffix}. Source: Open-Meteo."
    )


def _weather_code_label(code: Any) -> str:
    try:
        return WEATHER_CODE_LABELS[int(code)]
    except (TypeError, ValueError, KeyError):
        return "Current conditions unavailable"


def run_weather_tool(query_or_location: str) -> str:
    location = _resolve_location_input(query_or_location, _extract_weather_location)
    if not location:
        return "Weather lookup requires a city or location."

    place = _geocode_location(location)
    if not place:
        return f"I couldn't find a reliable location match for '{location}'."

    payload = _fetch_open_meteo_current(
        place,
        [
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "weather_code",
            "wind_speed_10m",
        ],
    )
    if not payload:
        return f"I couldn't retrieve the current weather for {_format_location_name(place)}."

    current = payload.get("current", {})
    units = payload.get("current_units", {})

    summary_parts = [_weather_code_label(current.get("weather_code"))]

    temperature = current.get("temperature_2m")
    if temperature is not None:
        summary_parts.append(f"{_format_number(temperature)}{units.get('temperature_2m', '°C')}")

    apparent = current.get("apparent_temperature")
    if apparent is not None:
        summary_parts.append(f"feels like {_format_number(apparent)}{units.get('apparent_temperature', '°C')}")

    humidity = current.get("relative_humidity_2m")
    if humidity is not None:
        summary_parts.append(f"humidity {_format_number(humidity)}{units.get('relative_humidity_2m', '%')}")

    wind_speed = current.get("wind_speed_10m")
    if wind_speed is not None:
        summary_parts.append(f"wind {_format_number(wind_speed)}{units.get('wind_speed_10m', 'km/h')}")

    return f"Current weather in {_format_location_name(place)}: {', '.join(summary_parts)}. Source: Open-Meteo."


def _parse_rss_items(xml_text: str, max_items: int = 5) -> list[dict[str, str]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    items: list[dict[str, str]] = []
    for item in root.findall(".//item"):
        title = _normalize_space(item.findtext("title", default=""))
        link = _normalize_space(item.findtext("link", default=""))
        pub_date = _normalize_space(item.findtext("pubDate", default=""))
        if not title or not link:
            continue
        items.append({"title": title, "url": link, "pub_date": pub_date})
        if len(items) >= max_items:
            break
    return items


def run_news_tool(query: str) -> tuple[str, list[str]]:
    normalized = _normalize_space(query)
    lower_query = normalized.lower()
    if not normalized:
        return ("News lookup requires a non-empty query.", [])

    url = GOOGLE_NEWS_TOP_RSS_URL
    params: dict[str, str] | None = None
    if any(token in lower_query for token in ("world", "global")):
        url = BBC_WORLD_RSS_URL
    elif not any(token in lower_query for token in ("headline", "headlines", "top stories", "breaking news")):
        url = GOOGLE_NEWS_SEARCH_RSS_URL
        params = {"q": normalized, "hl": "en-US", "gl": "US", "ceid": "US:en"}

    try:
        response = requests.get(url, params=params, headers=DEFAULT_HTTP_HEADERS, timeout=WEB_SEARCH_TIMEOUT_SEC)
        response.raise_for_status()
    except Exception:
        return ("I couldn't retrieve current news headlines right now.", [])

    items = _parse_rss_items(response.text, max_items=WEB_SEARCH_MAX_RESULTS)
    if not items:
        return ("I couldn't parse any current news headlines right now.", [])

    lines = []
    source_urls = []
    for index, item in enumerate(items, start=1):
        date_suffix = f" ({item['pub_date']})" if item.get("pub_date") else ""
        lines.append(f"{index}. {item['title']}{date_suffix}")
        source_urls.append(item["url"])

    source_name = "BBC World RSS" if url == BBC_WORLD_RSS_URL else "Google News RSS"
    return (f"Top news headlines from {source_name}:\n" + "\n".join(lines), source_urls)


def _fetch_stock_price(symbol: str) -> Optional[str]:
    try:
        response = requests.get(
            TWELVE_DATA_PRICE_URL,
            params={"symbol": symbol, "apikey": "demo"},
            headers=DEFAULT_HTTP_HEADERS,
            timeout=WEB_SEARCH_TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = response.json()
        price = payload.get("price")
        if price:
            return f"{_format_number(price)} USD (Twelve Data demo)"
    except Exception:
        pass

    try:
        response = requests.get(
            f"https://stooq.com/q/l/?s={symbol.lower()}.us&i=d",
            headers=DEFAULT_HTTP_HEADERS,
            timeout=WEB_SEARCH_TIMEOUT_SEC,
        )
        response.raise_for_status()
        row = response.text.strip().split(",")
        if len(row) >= 7 and row[6] not in {"N/D", ""}:
            date_part = row[1].strip()
            time_part = row[2].strip()
            timestamp = f" as of {date_part} {time_part} UTC" if date_part and time_part else ""
            return f"{_format_number(row[6])} USD{timestamp} (Stooq)"
    except Exception:
        return None
    return None


def _fetch_crypto_price(coin_id: str, symbol: str) -> Optional[str]:
    try:
        response = requests.get(
            COINGECKO_SIMPLE_PRICE_URL,
            params={"ids": coin_id, "vs_currencies": "usd"},
            headers=DEFAULT_HTTP_HEADERS,
            timeout=WEB_SEARCH_TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = response.json()
        usd_price = payload.get(coin_id, {}).get("usd")
        if usd_price is not None:
            return f"{_format_number(usd_price)} USD (CoinGecko)"
    except Exception:
        pass

    try:
        response = requests.get(
            COINBASE_SPOT_PRICE_URL.format(symbol=symbol),
            headers=DEFAULT_HTTP_HEADERS,
            timeout=WEB_SEARCH_TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = response.json()
        amount = payload.get("data", {}).get("amount")
        if amount:
            return f"{_format_number(amount)} USD (Coinbase spot)"
    except Exception:
        return None
    return None


def run_asset_price_tool(query: str) -> str:
    crypto_match = _extract_crypto_match(query)
    if crypto_match:
        coin_id, symbol, label = crypto_match
        price = _fetch_crypto_price(coin_id=coin_id, symbol=symbol)
        if price:
            return f"Current {label} price: {price}. Source: {price.split('(', 1)[-1].rstrip(')')}."
        return f"I couldn't retrieve the current {label} price."

    stock_match = _extract_stock_match(query)
    if stock_match:
        symbol, label = stock_match
        price = _fetch_stock_price(symbol)
        if price:
            return f"Latest {label} ({symbol}) price: {price}. Source: {price.split('(', 1)[-1].rstrip(')')}."
        return f"I couldn't retrieve the latest {label} ({symbol}) price."

    return "Price lookup requires a supported stock or cryptocurrency."


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


def _domain_quality_score(url: str) -> float:
    domain = urlparse(url or "").netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    score = 0.0

    for trusted_domain, boost in TRUSTED_DOMAIN_BOOST.items():
        if domain == trusted_domain or domain.endswith(f".{trusted_domain}"):
            score += boost

    for low_domain, penalty in LOW_QUALITY_DOMAIN_PENALTY.items():
        if domain == low_domain or domain.endswith(f".{low_domain}"):
            score += penalty
    return score


def _relevance_score(query_terms: list[str], title: str, snippet: str, url: str) -> float:
    if not query_terms:
        return 0.0
    haystack = " ".join([title or "", snippet or "", url or ""]).lower()
    hits = sum(1 for term in query_terms if term in haystack)
    lexical = hits / len(query_terms)
    return lexical + _domain_quality_score(url)


def _is_generic_web_query(query: str) -> bool:
    q = _normalize_space(query).lower()
    return q in GENERIC_WEB_QUERIES or _is_tool_control_query(q)


def _is_referential_query(query: str) -> bool:
    tokens = _tokenize(query)
    return any(token in REFERENTIAL_TOKENS for token in tokens)


def _extract_user_queries(chat_history_context: str) -> list[str]:
    if not chat_history_context:
        return []
    return [
        candidate.strip() for candidate in re.findall(r"(?im)^user:\s*(.+)$", chat_history_context) if candidate.strip()
    ]


def _extract_last_user_query(chat_history_context: str) -> str:
    candidates = _extract_user_queries(chat_history_context)
    if not candidates:
        return ""
    return candidates[-1]


def _is_tool_control_query(query: str) -> bool:
    normalized = _normalize_space(query).lower()
    if not normalized:
        return False
    if normalized in GENERIC_WEB_QUERIES:
        return True
    return any(re.match(pattern, normalized) for pattern in TOOL_CONTROL_PATTERNS)


def _extract_last_substantive_user_query(chat_history_context: str) -> str:
    for candidate in reversed(_extract_user_queries(chat_history_context)):
        if not _is_tool_control_query(candidate):
            return candidate
    return ""


def _resolve_tool_target_query(query: str, chat_history_context: str) -> str:
    normalized_query = _normalize_space(query)
    if not _is_tool_control_query(normalized_query):
        return normalized_query

    last_user = _extract_last_substantive_user_query(chat_history_context)
    if last_user:
        return last_user
    return normalized_query


def _resolve_web_query(tool_input: str, user_query: str, chat_history_context: str) -> str:
    query = (tool_input or "").strip() or (user_query or "").strip()
    if not _is_generic_web_query(query):
        return query

    last_user = _extract_last_substantive_user_query(chat_history_context)
    if last_user and not _is_generic_web_query(last_user):
        return last_user
    return query


def _query_candidates(cleaned_query: str) -> list[str]:
    candidates = [cleaned_query]
    lower_query = cleaned_query.lower()
    time_location = _extract_time_location(cleaned_query)
    weather_location = _extract_weather_location(cleaned_query)

    if time_location:
        candidates.extend(
            [
                f"current time in {time_location}",
                f"{time_location} local time",
                f"site:timeanddate.com {time_location} time",
            ]
        )
    if weather_location:
        candidates.extend(
            [
                f"current weather in {weather_location}",
                f"{weather_location} weather forecast",
                f"site:weather.com {weather_location} weather",
            ]
        )
    if _has_news_intent(cleaned_query):
        candidates.extend(
            [
                "top world headlines",
                "global breaking news",
                "latest world news",
            ]
        )
    if "best model" in lower_query or "current model" in lower_query or "latest model" in lower_query:
        if "openai" in lower_query or "open ai" in lower_query:
            candidates.extend(
                [
                    "site:platform.openai.com/docs/models OpenAI models",
                    "site:help.openai.com OpenAI best model",
                    "site:openai.com OpenAI latest model",
                ]
            )
        if "gemini" in lower_query or "google" in lower_query:
            candidates.extend(
                [
                    "site:ai.google.dev gemini models",
                    "site:blog.google gemini latest model",
                    "site:deepmind.google gemini model",
                ]
            )
    if "openai" in lower_query:
        candidates.append(f"site:openai.com {cleaned_query}")
        candidates.append(f"site:help.openai.com {cleaned_query}")

    deduped: list[str] = []
    seen = set()
    for candidate in candidates:
        normalized = candidate.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def _search_via_ddgs(cleaned_query: str) -> list[dict[str, str]]:
    if not DDGS_AVAILABLE:
        return []

    rows: list[dict[str, str]] = []
    query_candidates = _query_candidates(cleaned_query)
    max_results = max(WEB_SEARCH_MAX_RESULTS * WEB_SEARCH_CANDIDATE_FACTOR, WEB_SEARCH_MAX_RESULTS)
    use_news = any(hint in cleaned_query.lower() for hint in RECENCY_HINTS) or _has_news_intent(cleaned_query)

    try:
        with DDGS() as ddgs:
            for candidate in query_candidates:
                raw_rows = list(ddgs.text(candidate, max_results=max_results))
                for item in raw_rows:
                    if not isinstance(item, dict):
                        continue
                    title = str(item.get("title", "")).strip()
                    snippet = str(item.get("body", "")).strip()
                    url = str(item.get("href", "")).strip()
                    if not (title or snippet or url):
                        continue
                    rows.append({"title": title, "snippet": snippet, "url": url})

            if use_news:
                for candidate in query_candidates:
                    raw_news = list(ddgs.news(candidate, max_results=max_results))
                    for item in raw_news:
                        if not isinstance(item, dict):
                            continue
                        title = str(item.get("title", "")).strip()
                        snippet = str(item.get("body", "")).strip()
                        url = str(item.get("url", "")).strip()
                        if not (title or snippet or url):
                            continue
                        rows.append({"title": title, "snippet": snippet, "url": url})
    except Exception:
        return []

    return rows


def _decode_ddg_html_url(raw_url: str) -> str:
    unescaped = html.unescape(raw_url or "").strip()
    if unescaped.startswith("//"):
        unescaped = f"https:{unescaped}"
    parsed = urlparse(unescaped)
    if "duckduckgo.com" not in parsed.netloc:
        return unescaped
    encoded = parse_qs(parsed.query).get("uddg", [""])[0]
    return unquote(encoded) if encoded else unescaped


def _search_via_html_ddg(cleaned_query: str) -> list[dict[str, str]]:
    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": cleaned_query},
            headers=SEARCH_HTTP_HEADERS,
            timeout=WEB_SEARCH_TIMEOUT_SEC,
        )
        response.raise_for_status()
        page = response.text
    except Exception:
        return []

    links = re.findall(r'result__a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, flags=re.IGNORECASE | re.DOTALL)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page, flags=re.IGNORECASE | re.DOTALL)

    rows: list[dict[str, str]] = []
    for index, (href, title_html) in enumerate(links):
        title = _normalize_space(re.sub(r"<.*?>", "", html.unescape(title_html)))
        snippet_html = snippets[index] if index < len(snippets) else ""
        snippet = _normalize_space(re.sub(r"<.*?>", "", html.unescape(snippet_html)))
        url = _decode_ddg_html_url(href)
        if not (title or snippet or url):
            continue
        rows.append({"title": title, "snippet": snippet, "url": url})
        if len(rows) >= max(WEB_SEARCH_MAX_RESULTS * WEB_SEARCH_CANDIDATE_FACTOR, WEB_SEARCH_MAX_RESULTS):
            break
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


def _decode_yahoo_redirect_url(raw_url: str) -> str:
    unescaped = html.unescape(raw_url or "").strip()
    if not unescaped:
        return ""

    parsed = urlparse(unescaped)
    if "search.yahoo.com" not in parsed.netloc and "r.search.yahoo.com" not in parsed.netloc:
        return unescaped

    ru_match = re.search(r"/RU=([^/]+)/", parsed.path, flags=re.IGNORECASE)
    if ru_match:
        return unquote(ru_match.group(1))

    ru = parse_qs(parsed.query).get("RU", [""])[0]
    return unquote(ru) if ru else unescaped


def _search_via_yahoo_html(cleaned_query: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    query_candidates = _query_candidates(cleaned_query)
    max_results = max(WEB_SEARCH_MAX_RESULTS * WEB_SEARCH_CANDIDATE_FACTOR, WEB_SEARCH_MAX_RESULTS)

    result_pattern = re.compile(r'<li><div class="dd lst algo.*?</li>', flags=re.IGNORECASE | re.DOTALL)
    title_pattern = re.compile(
        r'<a[^>]+href="([^"]+)"[^>]*>.*?<h3[^>]*>.*?<span[^>]*>(.*?)</span>.*?</h3>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<div class="compText[^>]*>\s*<p[^>]*>(.*?)</p>',
        flags=re.IGNORECASE | re.DOTALL,
    )

    for candidate in query_candidates:
        try:
            response = requests.get(
                YAHOO_SEARCH_URL,
                params={"p": candidate},
                headers=SEARCH_HTTP_HEADERS,
                timeout=WEB_SEARCH_TIMEOUT_SEC,
            )
            response.raise_for_status()
            page = response.text
        except Exception:
            continue

        for block in result_pattern.findall(page):
            title_match = title_pattern.search(block)
            if not title_match:
                continue

            url = _decode_yahoo_redirect_url(title_match.group(1))
            if not url or "search.yahoo.com/search" in url:
                continue

            title = _normalize_space(re.sub(r"<.*?>", "", html.unescape(title_match.group(2))))
            snippet_match = snippet_pattern.search(block)
            snippet = ""
            if snippet_match:
                snippet = _normalize_space(re.sub(r"<.*?>", "", html.unescape(snippet_match.group(1))))

            if not (title or snippet or url):
                continue

            rows.append({"title": title, "snippet": snippet, "url": url})
            if len(rows) >= max_results:
                return rows

    return rows


def _parse_markdown_link_line(line: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"\[([^\]]+)\]\((https?://[^\)]+)\)", line.strip())
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _search_via_jina_mirror(cleaned_query: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    query_candidates = [cleaned_query]
    for candidate in _query_candidates(cleaned_query):
        if candidate.strip().lower() == cleaned_query.lower():
            continue
        query_candidates.append(candidate)
        if len(query_candidates) >= 3:
            break
    max_results = max(WEB_SEARCH_MAX_RESULTS * WEB_SEARCH_CANDIDATE_FACTOR, WEB_SEARCH_MAX_RESULTS)

    for candidate in query_candidates:
        mirrored_url = (
            f"{JINA_MIRROR_PREFIX}https://html.duckduckgo.com/html/?q={requests.utils.quote(candidate, safe='')}"
        )
        try:
            response = requests.get(
                mirrored_url,
                headers=SEARCH_HTTP_HEADERS,
                timeout=max(WEB_SEARCH_TIMEOUT_SEC, 20),
            )
            response.raise_for_status()
            markdown = response.text
        except Exception:
            continue

        content = markdown.split("Markdown Content:\n", 1)[-1]
        lines = [line.strip() for line in content.splitlines()]

        for index, line in enumerate(lines):
            parsed_link = _parse_markdown_link_line(line)
            if not parsed_link:
                continue

            title, raw_url = parsed_link
            if not title or title.lower() == "duckduckgo":
                continue

            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            if not re.fullmatch(r"-{10,}", next_line):
                continue

            url = _decode_ddg_html_url(raw_url)
            if not url:
                continue
            parsed_url = urlparse(url)
            lower_path = parsed_url.path.lower()
            if lower_path.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico")):
                continue

            snippet = ""
            for look_ahead in lines[index + 2 : index + 8]:
                parsed_snippet = _parse_markdown_link_line(look_ahead)
                if not parsed_snippet:
                    continue
                snippet_text, snippet_url = parsed_snippet
                if snippet_text.startswith("!"):
                    continue
                if _decode_ddg_html_url(snippet_url) != url:
                    continue
                snippet = _normalize_space(snippet_text.replace("**", ""))
                break

            rows.append({"title": _normalize_space(title), "snippet": snippet, "url": url})
            if len(rows) >= max_results:
                return rows

        if rows:
            return rows

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
        scored_rows = [item for item in scored_rows if item[0] >= WEB_SEARCH_MIN_RELEVANCE]

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
    html_rows = _search_via_html_ddg(cleaned) if not ddgs_rows else []
    jina_rows = _search_via_jina_mirror(cleaned) if not (ddgs_rows or html_rows) else []
    yahoo_rows = _search_via_yahoo_html(cleaned) if not (ddgs_rows or html_rows or jina_rows) else []
    fallback_rows = _search_via_instant_api(cleaned) if not (ddgs_rows or html_rows or jina_rows or yahoo_rows) else []
    candidate_rows = ddgs_rows or html_rows or jina_rows or yahoo_rows or fallback_rows
    ranked_rows = _rank_search_results(cleaned, candidate_rows)
    if not ranked_rows and ddgs_rows and html_rows:
        ranked_rows = _rank_search_results(cleaned, html_rows)
    if not ranked_rows and candidate_rows:
        ranked_rows = candidate_rows[:WEB_SEARCH_MAX_RESULTS]

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

    if _has_time_intent(q):
        return AgentAction(
            tool="current_time",
            tool_input=_extract_time_location(q),
            reason="Detected current-time lookup.",
        )

    if _has_weather_intent(q):
        return AgentAction(
            tool="weather",
            tool_input=_extract_weather_location(q),
            reason="Detected weather lookup.",
        )

    if _has_asset_price_intent(q):
        return AgentAction(tool="asset_price", tool_input=q, reason="Detected asset-price lookup.")

    if _has_news_intent(q):
        return AgentAction(tool="news", tool_input=q, reason="Detected news lookup.")

    has_explicit_web_intent = any(hint in lower_q for hint in EXPLICIT_WEB_HINTS)
    has_recency_hint = any(hint in lower_q for hint in RECENCY_HINTS)

    # Avoid accidental web-search trigger on referential follow-ups like:
    # "explain its current use" (should stay in conversation/RAG context).
    if has_explicit_web_intent or (has_recency_hint and not _is_referential_query(q)):
        cleaned = re.sub(r"^(search|look up|lookup)\s+", "", q, flags=re.IGNORECASE).strip()
        return AgentAction(tool="web_search", tool_input=cleaned or q, reason="Detected web-search intent.")

    return AgentAction(tool="none", tool_input="", reason="No clear tool intent.")


def _llm_planned_action(query: str, llm_instance, chat_history_context: str = "") -> AgentAction:
    if not AGENT_PLANNING_ENABLED:
        return AgentAction(tool="none", tool_input="", reason="Planning disabled.")

    planner_prompt = (
        "You are a tool router.\n"
        "Choose exactly one tool for the user query.\n"
        "Allowed tools: none, calculator, current_time, weather, asset_price, news, web_search.\n"
        "Use calculator only for arithmetic.\n"
        "Use current_time for local time queries tied to a city or place.\n"
        "Use weather for current weather queries tied to a city or place.\n"
        "Use asset_price for current stock or cryptocurrency price questions.\n"
        "Use news for current headlines or news lookups.\n"
        "Use web_search for latest/current/news/internet lookup when the dedicated tools do not fit.\n"
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
        if tool not in {"none", "calculator", "current_time", "weather", "asset_price", "news", "web_search"}:
            return AgentAction(tool="none", tool_input="", reason="Planner returned unsupported tool.")
        if tool == "calculator":
            tool_input = tool_input or _extract_math_expression(query)
        if tool == "current_time":
            tool_input = tool_input or _extract_time_location(query)
        if tool == "weather":
            tool_input = tool_input or _extract_weather_location(query)
        if tool in {"asset_price", "news"}:
            tool_input = tool_input or query
        if tool == "web_search":
            tool_input = tool_input or query
        return AgentAction(tool=tool, tool_input=tool_input, reason=reason)
    except Exception:
        return AgentAction(tool="none", tool_input="", reason="Planner failed.")


def choose_agent_action(query: str, llm_instance, chat_history_context: str = "") -> AgentAction:
    effective_query = _resolve_tool_target_query(query, chat_history_context)

    if _is_referential_query(effective_query) and not any(
        hint in (effective_query or "").lower() for hint in EXPLICIT_WEB_HINTS
    ):
        return AgentAction(tool="none", tool_input="", reason="Referential query; avoid web tool over-trigger.")

    heuristic = _heuristic_action(effective_query)
    if heuristic.tool != "none":
        return heuristic

    planned = _llm_planned_action(effective_query, llm_instance, chat_history_context=chat_history_context)
    if planned.tool != "none":
        return planned

    return heuristic


def prepare_agent_tool_run(query: str, llm_instance, chat_history_context: str = "") -> Optional[dict[str, Any]]:
    resolved_query = _resolve_tool_target_query(query, chat_history_context)
    action = choose_agent_action(resolved_query, llm_instance, chat_history_context=chat_history_context)
    if action.tool == "none":
        return None

    tool_result = ""
    source_urls: list[str] = []
    if action.tool == "calculator":
        tool_result = run_calculator_tool(action.tool_input)
    elif action.tool == "current_time":
        tool_result = run_current_time_tool(action.tool_input or resolved_query)
    elif action.tool == "weather":
        tool_result = run_weather_tool(action.tool_input or resolved_query)
    elif action.tool == "asset_price":
        tool_result = run_asset_price_tool(action.tool_input or resolved_query)
    elif action.tool == "news":
        tool_result, source_urls = run_news_tool(action.tool_input or resolved_query)
    elif action.tool == "web_search":
        resolved_web_query = _resolve_web_query(action.tool_input, resolved_query, chat_history_context)
        tool_result, source_urls = run_web_search_tool(resolved_web_query)
    else:
        return None

    if action.tool in {"current_time", "weather", "asset_price", "news"}:
        response = tool_result
        if source_urls and "sources:" not in response.lower():
            response = f"{response}\n\nSources: {', '.join(source_urls[:3])}"
        return {
            "direct_response": response,
            "tool_used": action.tool,
            "tool_input": action.tool_input,
            "tool_reason": action.reason,
            "usage": {},
            "resolved_query": resolved_query,
            "source_urls": source_urls,
        }

    synthesis_prompt = (
        "You are a helpful assistant.\n"
        "The latest user message may ask you to use tools on the prior question.\n"
        "Use the tool output below to answer the user's question accurately.\n"
        "If the tool output says it failed or has no results, be transparent.\n"
        "Keep the answer concise and practical.\n\n"
        f"Conversation history:\n{chat_history_context or '[none]'}\n\n"
        f"Latest user message:\n{query}\n\n"
        f"Resolved question to answer:\n{resolved_query}\n\n"
        f"Tool used: {action.tool}\n"
        f"Tool output:\n{tool_result}\n"
    )

    return {
        "synthesis_prompt": synthesis_prompt,
        "tool_result": tool_result,
        "tool_used": action.tool,
        "tool_input": action.tool_input,
        "tool_reason": action.reason,
        "resolved_query": resolved_query,
        "source_urls": source_urls,
    }


def run_agent_with_tools(query: str, llm_instance, chat_history_context: str = "") -> Optional[dict[str, Any]]:
    prepared = prepare_agent_tool_run(query, llm_instance, chat_history_context=chat_history_context)
    if not prepared:
        return None

    direct_response = cast(Optional[str], prepared.get("direct_response"))
    if direct_response is not None:
        return {
            "response": direct_response,
            "tool_used": prepared["tool_used"],
            "tool_input": prepared["tool_input"],
            "tool_reason": prepared["tool_reason"],
            "usage": prepared.get("usage", {}),
        }

    synthesis_prompt = cast(str, prepared["synthesis_prompt"])
    final = llm_instance.invoke(synthesis_prompt)
    response = str(getattr(final, "content", final)).strip()

    tool_used = cast(str, prepared["tool_used"])
    source_urls = cast(list[str], prepared.get("source_urls", []))
    if tool_used == "web_search" and source_urls and "sources:" not in response.lower():
        source_list = ", ".join(source_urls[:3])
        response = f"{response}\n\nSources: {source_list}"

    return {
        "response": response,
        "tool_used": tool_used,
        "tool_input": prepared["tool_input"],
        "tool_reason": prepared["tool_reason"],
        "usage": extract_usage_metrics(final),
    }
