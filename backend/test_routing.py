"""Unit tests for routing functions: route_returns and route_intent."""

import pytest
from langchain_core.messages import AIMessage

from app import route_returns, route_intent


class TestRouteIntent:
    """Test route_intent maps intent labels to correct node names."""

    def test_product_advice(self):
        state = {"intent": "product_advice"}
        assert route_intent(state) == "product_advisor"

    def test_returns(self):
        state = {"intent": "returns"}
        assert route_intent(state) == "returns_agent"

    def test_general(self):
        state = {"intent": "general"}
        assert route_intent(state) == "general_response"

    def test_unknown_intent_defaults_to_general(self):
        state = {"intent": "something_unexpected"}
        assert route_intent(state) == "general_response"

    def test_missing_intent_defaults_to_general(self):
        state = {}
        assert route_intent(state) == "general_response"


class TestRouteReturns:
    """Test route_returns for the three routing cases."""

    def test_request_refund_routes_to_human_approval(self):
        msg = AIMessage(
            content="",
            tool_calls=[{
                "name": "request_refund",
                "args": {"order_number": "LUMA-1001", "amount": 30.0, "reason": "Unsatisfied"},
                "id": "call_123",
            }],
        )
        state = {"messages": [msg]}
        assert route_returns(state) == "human_approval"

    def test_normal_tool_calls_route_to_returns_tools(self):
        msg = AIMessage(
            content="",
            tool_calls=[{
                "name": "lookup_order",
                "args": {"order_number": "LUMA-1001"},
                "id": "call_456",
            }],
        )
        state = {"messages": [msg]}
        assert route_returns(state) == "returns_tools"

    def test_multiple_normal_tool_calls_route_to_returns_tools(self):
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "lookup_order", "args": {"order_number": "LUMA-1001"}, "id": "call_1"},
                {"name": "check_return_policy", "args": {"order_number": "LUMA-1001"}, "id": "call_2"},
            ],
        )
        state = {"messages": [msg]}
        assert route_returns(state) == "returns_tools"

    def test_plain_reply_routes_to_end(self):
        msg = AIMessage(content="What is your order number?")
        state = {"messages": [msg]}
        assert route_returns(state) == "__end__"

    def test_request_refund_among_other_calls_routes_to_human_approval(self):
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "check_return_policy", "args": {"order_number": "LUMA-1001"}, "id": "call_a"},
                {"name": "request_refund", "args": {"order_number": "LUMA-1001", "amount": 30.0, "reason": "Damaged"}, "id": "call_b"},
            ],
        )
        state = {"messages": [msg]}
        assert route_returns(state) == "human_approval"
