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
