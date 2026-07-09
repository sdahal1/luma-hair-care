"""End-to-end style integration tests for the routed graph (LLM mocked).

These cover the demo flows that the other test modules don't already verify:
  - Req 1.4: per-thread memory continuity across turns AND isolation between
    different thread_ids.
  - Req 4.2: an ineligible order is explained and does NOT proceed to approval
    (the returns agent never emits request_refund / never reaches human_approval).
  - Req 1.2 / no-leak: the streaming filter only forwards user-facing nodes, so
    classifier and tool-node text can't leak to the client.

The LLM calls are patched at the module level (like test_hitl.py) so the graph
runs deterministically without AWS Bedrock:
  - `app.llm`                 — classify_intent / general_response
  - `app.returns_agent_llm`   — returns_agent
  - `app.product_advisor_llm` — product_advisor
"""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app
import store
from app import graph


def make_ai(content="", tool_calls=None):
    return AIMessage(content=content, tool_calls=tool_calls or [])


@pytest.fixture(autouse=True)
def reset_order_state():
    """Reset LUMA-1001's refund status before and after each test for isolation."""
    store._data["orders"]["LUMA-1001"]["refund_status"] = None
    yield
    store._data["orders"]["LUMA-1001"]["refund_status"] = None


# ---------------------------------------------------------------------------
# Req 1.4 — memory continuity on one thread_id, isolation across thread_ids
# ---------------------------------------------------------------------------

def test_memory_continuity_across_turns_same_thread():
    """Req 1.4: a follow-up on the same thread_id retains prior turns."""
    config = {"configurable": {"thread_id": "mem-continuity"}}

    with patch.object(app, "llm") as mock_llm, \
         patch.object(app, "product_advisor_llm") as mock_advisor:
        mock_llm.invoke.return_value = make_ai("product_advice")
        mock_advisor.invoke.side_effect = [
            make_ai("Hydra Wash Shampoo is great for dry hair."),
            make_ai("Yes, it is sulfate-free."),
        ]

        graph.invoke({"messages": [("user", "Best shampoo for dry hair?")]}, config=config)
        graph.invoke({"messages": [("user", "Is it sulfate-free?")]}, config=config)

    msgs = graph.get_state(config).values["messages"]
    human_texts = [m.content for m in msgs if isinstance(m, HumanMessage)]
    # Both user turns persisted in the same thread's history.
    assert "Best shampoo for dry hair?" in human_texts
    assert "Is it sulfate-free?" in human_texts
    assert msgs[-1].content == "Yes, it is sulfate-free."


def test_memory_isolation_across_threads():
    """Req 1.4: a different thread_id does not see another thread's history."""
    cfg_a = {"configurable": {"thread_id": "iso-a"}}
    cfg_b = {"configurable": {"thread_id": "iso-b"}}

    with patch.object(app, "llm") as mock_llm, \
         patch.object(app, "product_advisor_llm") as mock_advisor:
        mock_llm.invoke.return_value = make_ai("product_advice")
        mock_advisor.invoke.side_effect = [
            make_ai("Reply for thread A."),
            make_ai("Reply for thread B."),
        ]

        graph.invoke({"messages": [("user", "Question from thread A")]}, config=cfg_a)
        graph.invoke({"messages": [("user", "Question from thread B")]}, config=cfg_b)

    a_humans = [m.content for m in graph.get_state(cfg_a).values["messages"]
                if isinstance(m, HumanMessage)]
    b_humans = [m.content for m in graph.get_state(cfg_b).values["messages"]
                if isinstance(m, HumanMessage)]

    assert a_humans == ["Question from thread A"]
    assert b_humans == ["Question from thread B"]
    # Cross-thread leakage must not occur.
    assert "Question from thread B" not in a_humans
    assert "Question from thread A" not in b_humans


# ---------------------------------------------------------------------------
# Req 4.2 — ineligible order is explained, no refund request / no approval
# ---------------------------------------------------------------------------

def test_ineligible_order_does_not_reach_approval():
    """Req 4.2: an out-of-window order is explained and never hits human_approval.

    The returns agent checks the policy (a normal tool call executed by the
    returns_tools ToolNode against the real store), then gives a plain reply
    explaining ineligibility — it must NOT emit request_refund.
    """
    config = {"configurable": {"thread_id": "ineligible"}}
    policy_call = make_ai(content="", tool_calls=[{
        "name": "check_return_policy",
        "args": {"order_number": "LUMA-1002"},
        "id": "toolu_policy_1",
    }])
    explanation = make_ai(
        "Order LUMA-1002 is outside the 30-day return window, so I'm unable to "
        "process a refund."
    )

    with patch.object(app, "llm") as mock_llm, \
         patch.object(app, "returns_agent_llm") as mock_returns:
        mock_llm.invoke.return_value = make_ai("returns")
        mock_returns.invoke.side_effect = [policy_call, explanation]

        graph.invoke(
            {"messages": [("user", "I want a refund for order LUMA-1002")]},
            config=config,
        )

    state = graph.get_state(config)
    # Ran to completion — never paused for approval.
    assert state.next == ()
    # No refund was applied to the ineligible order.
    assert store.refund_status("LUMA-1002") != "refunded"
    # The final assistant turn is the plain explanation, not a tool signal.
    final = state.values["messages"][-1]
    assert "30-day return window" in final.content
    assert not (getattr(final, "tool_calls", None) or [])
    # request_refund was never produced anywhere in the transcript.
    for m in state.values["messages"]:
        for call in (getattr(m, "tool_calls", None) or []):
            assert call["name"] != "request_refund"


# ---------------------------------------------------------------------------
# Req 1.2 / no-leak — only user-facing nodes are streamed to the client
# ---------------------------------------------------------------------------

def test_internal_nodes_excluded_from_streaming():
    """The streaming filter only forwards specialist replies, never internal
    nodes (classifier, tool nodes, approval plumbing), so routing labels and
    raw retrieved chunks can't leak to the client."""
    internal_nodes = {
        "classify_intent",
        "product_tools",
        "returns_tools",
        "human_approval",
        "apply_refund",
        "deny_refund",
    }
    assert app.USER_FACING_NODES == {"product_advisor", "returns_agent", "general_response"}
    assert internal_nodes.isdisjoint(app.USER_FACING_NODES)
