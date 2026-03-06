import agent_tools


def test_calculator_tool_handles_basic_expression():
    assert agent_tools.run_calculator_tool("calculate 2 + 2 * 3") == "8"


def test_choose_agent_action_detects_web_search_intent():
    class DummyLLM:
        def invoke(self, _prompt):
            class DummyResponse:
                content = '{"tool":"none","tool_input":"","reason":"none"}'

            return DummyResponse()

    action = agent_tools.choose_agent_action("search latest ai news", DummyLLM())
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

    def fake_get(url, params=None, timeout=None):
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

    def fake_get(url, params=None, timeout=None):
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


def test_query_candidates_include_openai_official_sites():
    candidates = agent_tools._query_candidates("latest best model from openai")
    assert any("site:openai.com" in candidate for candidate in candidates)


def test_heuristic_action_does_not_web_search_on_referential_current_query():
    action = agent_tools._heuristic_action("explain its current use")
    assert action.tool == "none"
