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


def test_query_candidates_include_openai_official_sites():
    candidates = agent_tools._query_candidates("latest best model from openai")
    assert any("site:openai.com" in candidate for candidate in candidates)


def test_heuristic_action_does_not_web_search_on_referential_current_query():
    action = agent_tools._heuristic_action("explain its current use")
    assert action.tool == "none"
