import agent_tools


def test_calculator_tool_handles_basic_expression():
    assert agent_tools.run_calculator_tool("calculate 2 + 2 * 3") == "8"


def test_choose_agent_action_detects_web_search_intent():
    class DummyLLM:
        def invoke(self, _prompt):
            class DummyResponse:
                content = '{"tool":"none","tool_input":"","reason":"none"}'

            return DummyResponse()

    action = agent_tools.choose_agent_action("search latest OpenAI API docs", DummyLLM())
    assert action.tool == "web_search"


def test_choose_agent_action_detects_calculator_intent():
    class DummyLLM:
        def invoke(self, _prompt):
            class DummyResponse:
                content = '{"tool":"none","tool_input":"","reason":"none"}'

            return DummyResponse()

    action = agent_tools.choose_agent_action("what is 12*7", DummyLLM())
    assert action.tool == "calculator"


def test_choose_agent_action_detects_current_time_intent():
    class DummyLLM:
        def invoke(self, _prompt):
            class DummyResponse:
                content = '{"tool":"none","tool_input":"","reason":"none"}'

            return DummyResponse()

    action = agent_tools.choose_agent_action("what is the current time in jaipur", DummyLLM())
    assert action.tool == "current_time"
    assert action.tool_input.lower() == "jaipur"


def test_choose_agent_action_detects_weather_intent():
    class DummyLLM:
        def invoke(self, _prompt):
            class DummyResponse:
                content = '{"tool":"none","tool_input":"","reason":"none"}'

            return DummyResponse()

    action = agent_tools.choose_agent_action("whats the current weather in jaipur", DummyLLM())
    assert action.tool == "weather"
    assert action.tool_input.lower() == "jaipur"


def test_choose_agent_action_detects_weather_of_intent():
    class DummyLLM:
        def invoke(self, _prompt):
            class DummyResponse:
                content = '{"tool":"none","tool_input":"","reason":"none"}'

            return DummyResponse()

    action = agent_tools.choose_agent_action("what is the current weather of jaipur", DummyLLM())
    assert action.tool == "weather"
    assert action.tool_input.lower() == "jaipur"


def test_choose_agent_action_detects_news_intent():
    class DummyLLM:
        def invoke(self, _prompt):
            class DummyResponse:
                content = '{"tool":"none","tool_input":"","reason":"none"}'

            return DummyResponse()

    action = agent_tools.choose_agent_action("what are the top global headlines right now?", DummyLLM())
    assert action.tool == "news"


def test_choose_agent_action_detects_asset_price_intent():
    class DummyLLM:
        def invoke(self, _prompt):
            class DummyResponse:
                content = '{"tool":"none","tool_input":"","reason":"none"}'

            return DummyResponse()

    stock_action = agent_tools.choose_agent_action("What is the current trading price of Apple stock?", DummyLLM())
    crypto_action = agent_tools.choose_agent_action("How much is one bitcoin worth in US dollars today?", DummyLLM())

    assert stock_action.tool == "asset_price"
    assert crypto_action.tool == "asset_price"


def test_choose_agent_action_current_time_without_location_keeps_empty_tool_input():
    class DummyLLM:
        def invoke(self, _prompt):
            class DummyResponse:
                content = '{"tool":"none","tool_input":"","reason":"none"}'

            return DummyResponse()

    action = agent_tools.choose_agent_action("hello whats the current time", DummyLLM())
    assert action.tool == "current_time"
    assert action.tool_input == ""


def test_choose_agent_action_keeps_real_location_with_leading_article():
    class DummyLLM:
        def invoke(self, _prompt):
            class DummyResponse:
                content = '{"tool":"none","tool_input":"","reason":"none"}'

            return DummyResponse()

    action = agent_tools.choose_agent_action("what is the current weather in the hague", DummyLLM())
    assert action.tool == "weather"
    assert action.tool_input.lower() == "the hague"


def test_relevance_scoring_prefers_matching_result():
    terms = agent_tools._query_terms("best openai llm model")
    good = agent_tools._relevance_score(
        query_terms=terms,
        title="OpenAI model comparison",
        snippet="Latest OpenAI LLM model details and benchmarks",
        url="https://openai.com/research",
    )
    bad = agent_tools._relevance_score(
        query_terms=terms,
        title="World Wide Web",
        snippet="History of the internet protocol suite.",
        url="https://en.wikipedia.org/wiki/World_Wide_Web",
    )
    assert good > bad


def test_domain_quality_penalizes_low_quality_sources():
    assert agent_tools._domain_quality_score("https://openai.com/blog") > 0
    assert agent_tools._domain_quality_score("https://hinative.com/questions/1") < 0


def test_run_agent_with_tools_uses_last_user_context_for_generic_web_query(monkeypatch):
    captured = {"query": ""}

    def fake_search(query: str):
        captured["query"] = query
        return ("1. Result: test snippet", ["https://example.com"])

    monkeypatch.setattr(agent_tools, "run_web_search_tool", fake_search)
    monkeypatch.setattr(
        agent_tools,
        "choose_agent_action",
        lambda *_args, **_kwargs: agent_tools.AgentAction(
            tool="web_search",
            tool_input="search web",
            reason="test",
        ),
    )

    class DummyLLM:
        def invoke(self, _prompt):
            class DummyResponse:
                content = "Here are better results."

            return DummyResponse()

    response = agent_tools.run_agent_with_tools(
        query="search web",
        llm_instance=DummyLLM(),
        chat_history_context=(
            "User: what is current best llm model from openai\n" "Assistant: Let me check web results."
        ),
    )

    assert captured["query"] == "what is current best llm model from openai"
    assert response is not None
    assert response["tool_used"] == "web_search"


def test_resolve_tool_target_query_uses_last_substantive_user_query():
    resolved = agent_tools._resolve_tool_target_query(
        "use web search tools",
        "User: what is the current time in jaipur\nAssistant: I could not find it.",
    )
    assert resolved == "what is the current time in jaipur"


def test_run_current_time_tool_formats_open_meteo_response(monkeypatch):
    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params=None, timeout=None, headers=None):
        assert timeout == agent_tools.WEB_SEARCH_TIMEOUT_SEC
        if "geocoding-api" in url:
            assert params["name"] == "jaipur"
            return DummyResponse(
                {
                    "results": [
                        {
                            "name": "Jaipur",
                            "admin1": "Rajasthan",
                            "country": "India",
                            "latitude": 26.91,
                            "longitude": 75.79,
                            "timezone": "Asia/Kolkata",
                            "population": 3000000,
                        }
                    ]
                }
            )
        return DummyResponse(
            {
                "timezone": "Asia/Kolkata",
                "current": {"time": "2026-03-06T14:05", "is_day": 1},
            }
        )

    monkeypatch.setattr(agent_tools.requests, "get", fake_get)

    result = agent_tools.run_current_time_tool("what is the current time in jaipur")

    assert "Jaipur, Rajasthan, India" in result
    assert "2:05 PM" in result
    assert "Asia/Kolkata" in result


def test_run_current_time_tool_without_location_requests_location():
    result = agent_tools.run_current_time_tool("hello whats the current time")
    assert result == "Time lookup requires a city or location."


def test_run_weather_tool_formats_open_meteo_response(monkeypatch):
    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params=None, timeout=None, headers=None):
        assert timeout == agent_tools.WEB_SEARCH_TIMEOUT_SEC
        if "geocoding-api" in url:
            return DummyResponse(
                {
                    "results": [
                        {
                            "name": "Jaipur",
                            "admin1": "Rajasthan",
                            "country": "India",
                            "latitude": 26.91,
                            "longitude": 75.79,
                            "timezone": "Asia/Kolkata",
                            "population": 3000000,
                        }
                    ]
                }
            )
        return DummyResponse(
            {
                "current": {
                    "temperature_2m": 31.2,
                    "apparent_temperature": 30.0,
                    "relative_humidity_2m": 19,
                    "weather_code": 0,
                    "wind_speed_10m": 8.4,
                },
                "current_units": {
                    "temperature_2m": "°C",
                    "apparent_temperature": "°C",
                    "relative_humidity_2m": "%",
                    "wind_speed_10m": "km/h",
                },
            }
        )

    monkeypatch.setattr(agent_tools.requests, "get", fake_get)

    result = agent_tools.run_weather_tool("whats the current weather in jaipur")

    assert "Jaipur, Rajasthan, India" in result
    assert "Clear sky" in result
    assert "31.2°C" in result
    assert "humidity 19%" in result


def test_run_weather_tool_without_location_requests_location():
    result = agent_tools.run_weather_tool("what is the current weather")
    assert result == "Weather lookup requires a city or location."


def test_run_weather_tool_handles_richer_location_prompt(monkeypatch):
    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params=None, timeout=None, headers=None):
        if "geocoding-api" in url:
            assert params["name"] == "Jaipur, Rajasthan"
            return DummyResponse(
                {
                    "results": [
                        {
                            "name": "Jaipur",
                            "admin1": "Rajasthan",
                            "country": "India",
                            "latitude": 26.91,
                            "longitude": 75.79,
                            "timezone": "Asia/Kolkata",
                            "population": 3000000,
                        }
                    ]
                }
            )
        return DummyResponse(
            {
                "current": {
                    "temperature_2m": 33.2,
                    "apparent_temperature": 33.3,
                    "relative_humidity_2m": 20,
                    "weather_code": 0,
                    "wind_speed_10m": 6.9,
                },
                "current_units": {
                    "temperature_2m": "Â°C",
                    "apparent_temperature": "Â°C",
                    "relative_humidity_2m": "%",
                    "wind_speed_10m": "km/h",
                },
            }
        )

    monkeypatch.setattr(agent_tools.requests, "get", fake_get)

    result = agent_tools.run_weather_tool(
        "What is the current weather in Jaipur, Rajasthan right now? What is the temperature?"
    )

    assert "Jaipur, Rajasthan, India" in result
    assert "33.2Â°C" in result


def test_run_news_tool_formats_rss(monkeypatch):
    class DummyResponse:
        text = (
            "<?xml version='1.0'?>"
            "<rss><channel>"
            "<item><title>Headline One</title><link>https://example.com/1</link><pubDate>Fri, 06 Mar 2026 10:00:00 GMT</pubDate></item>"
            "<item><title>Headline Two</title><link>https://example.com/2</link><pubDate>Fri, 06 Mar 2026 09:00:00 GMT</pubDate></item>"
            "</channel></rss>"
        )

        def raise_for_status(self):
            return None

    monkeypatch.setattr(agent_tools.requests, "get", lambda *args, **kwargs: DummyResponse())

    summary, urls = agent_tools.run_news_tool("what are the top global headlines right now?")

    assert "Top news headlines" in summary
    assert "Headline One" in summary
    assert urls == ["https://example.com/1", "https://example.com/2"]


def test_run_asset_price_tool_formats_stock_quote(monkeypatch):
    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params=None, timeout=None, headers=None):
        assert params == {"symbol": "AAPL", "apikey": "demo"}
        return DummyResponse({"price": "260.26001"})

    monkeypatch.setattr(agent_tools.requests, "get", fake_get)

    result = agent_tools.run_asset_price_tool("What is the current trading price of Apple stock?")

    assert "Apple (AAPL)" in result
    assert "260.3 USD" in result


def test_run_asset_price_tool_formats_crypto_quote(monkeypatch):
    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params=None, timeout=None, headers=None):
        assert params == {"ids": "bitcoin", "vs_currencies": "usd"}
        return DummyResponse({"bitcoin": {"usd": 70510}})

    monkeypatch.setattr(agent_tools.requests, "get", fake_get)

    result = agent_tools.run_asset_price_tool("How much is one bitcoin worth in US dollars today?")

    assert "Bitcoin" in result
    assert "70510 USD" in result


def test_run_agent_with_tools_reuses_previous_question_for_tool_command(monkeypatch):
    captured = {"location": ""}

    def fake_time_tool(location: str):
        captured["location"] = location
        return "Current local time in Jaipur, Rajasthan, India: 2:05 PM on March 06, 2026 (Asia/Kolkata). Source: Open-Meteo."

    monkeypatch.setattr(agent_tools, "run_current_time_tool", fake_time_tool)

    class DummyLLM:
        def invoke(self, _prompt):
            class DummyResponse:
                content = "It is 2:05 PM in Jaipur."

            return DummyResponse()

    response = agent_tools.run_agent_with_tools(
        query="use web search tools",
        llm_instance=DummyLLM(),
        chat_history_context="User: what is the current time in jaipur\nAssistant: I could not find it.",
    )

    assert captured["location"].lower() == "jaipur"
    assert response is not None
    assert response["tool_used"] == "current_time"


def test_decode_yahoo_redirect_url_extracts_target():
    raw_url = (
        "https://r.search.yahoo.com/_ylt=test/RV=2/RE=1773994210/RO=10/"
        "RU=https%3a%2f%2fdevelopers.openai.com%2fapi%2fdocs/RK=2/RS=test-"
    )

    assert agent_tools._decode_yahoo_redirect_url(raw_url) == "https://developers.openai.com/api/docs"


def test_run_web_search_tool_falls_back_to_jina_mirror(monkeypatch):
    monkeypatch.setattr(agent_tools, "_search_via_ddgs", lambda _query: [])
    monkeypatch.setattr(agent_tools, "_search_via_html_ddg", lambda _query: [])
    monkeypatch.setattr(
        agent_tools,
        "_search_via_jina_mirror",
        lambda _query: [
            {
                "title": "OpenAI API Platform Documentation",
                "snippet": "Explore guides, API docs, and examples for the OpenAI API.",
                "url": "https://developers.openai.com/api/docs",
            }
        ],
    )
    monkeypatch.setattr(agent_tools, "_search_via_yahoo_html", lambda _query: [])
    monkeypatch.setattr(agent_tools, "_search_via_instant_api", lambda _query: [])

    summary, urls = agent_tools.run_web_search_tool("latest OpenAI API docs")

    assert "OpenAI API Platform Documentation" in summary
    assert urls == ["https://developers.openai.com/api/docs"]


def test_search_via_jina_mirror_parses_duckduckgo_markdown(monkeypatch):
    class DummyResponse:
        text = (
            "Title: latest OpenAI API docs at DuckDuckGo\n\n"
            "Markdown Content:\n"
            "latest OpenAI API docs at DuckDuckGo\n"
            "===============\n"
            "[OpenAI API Platform Documentation](https://duckduckgo.com/l/?uddg=https%3A%2F%2Fdevelopers.openai.com%2Fapi%2Fdocs)\n"
            "-------------------------\n"
            "[Explore guides, **API** docs, and examples for the **OpenAI** API.](https://duckduckgo.com/l/?uddg=https%3A%2F%2Fdevelopers.openai.com%2Fapi%2Fdocs)\n"
        )

        def raise_for_status(self):
            return None

    monkeypatch.setattr(agent_tools.requests, "get", lambda *args, **kwargs: DummyResponse())
    monkeypatch.setattr(agent_tools, "_query_candidates", lambda _query: ["latest OpenAI API docs"])

    rows = agent_tools._search_via_jina_mirror("latest OpenAI API docs")

    assert rows == [
        {
            "title": "OpenAI API Platform Documentation",
            "snippet": "Explore guides, API docs, and examples for the OpenAI API.",
            "url": "https://developers.openai.com/api/docs",
        }
    ]


def test_query_candidates_include_openai_official_sites():
    candidates = agent_tools._query_candidates("latest best model from openai")
    assert any("site:openai.com" in candidate for candidate in candidates)


def test_heuristic_action_does_not_web_search_on_referential_current_query():
    action = agent_tools._heuristic_action("explain its current use")
    assert action.tool == "none"
