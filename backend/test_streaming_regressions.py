"""Regression tests for the streaming-path bugs.

These pin the failure modes from the debugging write-up so they can't silently
return. All deterministic — the LLM is mocked, no AWS/Bedrock calls.
"""
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app


class TestClassifierContentShapes:
    """The root-cause bug: classify_intent must parse the label whether Bedrock
    returns content as a plain string (invoke path) or a list of content blocks
    (streaming path)."""

    def _run(self, content):
        with patch.object(app, "llm") as m:
            m.invoke.return_value = AIMessage(content=content)
            return app.classify_intent({"messages": [HumanMessage(content="q")]})

    def test_list_content_product(self):
        # The streaming shape that caused the original bug.
        assert self._run([{"type": "text", "text": "product_advice"}])["intent"] == "product_advice"

    def test_string_content_product(self):
        assert self._run("product_advice")["intent"] == "product_advice"

    def test_list_content_returns(self):
        r = self._run([{"type": "text", "text": "returns"}])
        assert r["intent"] == "returns"
        assert r.get("active_flow") == "returns"

    def test_string_content_general(self):
        assert self._run("general")["intent"] == "general"

    def test_active_returns_flow_short_circuits_without_llm(self):
        with patch.object(app, "llm") as m:
            r = app.classify_intent({
                "messages": [HumanMessage(content="anything")],
                "active_flow": "returns",
                "refund_applied": False,
            })
            assert r["intent"] == "returns"
            m.invoke.assert_not_called()


class TestToText:
    def test_string(self):
        assert app._to_text("hello") == "hello"

    def test_list_of_blocks(self):
        assert app._to_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"

    def test_block_without_type_key(self):
        assert app._to_text([{"text": "x"}]) == "x"


class TestNoToolMessages:
    """The toolConfig-warning fix: no-tool model calls must not receive tool
    blocks, and BOTH halves of each exchange must be removed to stay balanced."""

    def test_strips_both_halves(self):
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[{"name": "lookup_order", "args": {"order_number": "X"}, "id": "1"}]),
            ToolMessage(content="result", tool_call_id="1"),
            AIMessage(content="here is your answer"),
        ]
        out = app._no_tool_messages(msgs)
        assert [type(m).__name__ for m in out] == ["HumanMessage", "AIMessage"]
        assert out[-1].content == "here is your answer"
        assert all(not getattr(m, "tool_calls", None) for m in out)
        assert all(not isinstance(m, ToolMessage) for m in out)


class TestSSEEncoder:
    """The SSE display bugs: encoder must be newline-safe and preserve a token's
    leading space."""

    def test_single_line_frame(self):
        frame = app.sse("token", "a b")
        assert "data: a b\n" in frame
        assert frame.endswith("\n\n")

    def test_multiline_becomes_multiple_data_lines(self):
        frame = app.sse("token", "line1\nline2")
        assert "data: line1\n" in frame
        assert "data: line2\n" in frame

    def test_leading_space_preserved(self):
        # Token " sure" must survive so words don't run together on the client.
        assert "data:  sure\n" in app.sse("token", " sure")
