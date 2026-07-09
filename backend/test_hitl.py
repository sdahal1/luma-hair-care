"""Integration tests for the human-in-the-loop (HITL) refund flow.

Drives the compiled graph (`app.graph`) end-to-end with the LLM calls mocked
so the tests run without AWS Bedrock. Each LLM the graph nodes use is patched
at the module level:
  - `app.llm`                 — used by `classify_intent` / `general_response`
  - `app.returns_agent_llm`   — used by `returns_agent`
  - `app.product_advisor_llm` — used by `product_advisor`

The mocks return scripted `AIMessage`s (including a `request_refund` tool call)
so we can reproduce the multi-turn refund conversation deterministically.

Validates Requirements:
  - 5.1 graph pauses via interrupt() before any refund side effect
  - 5.4 approve → resume, refund applied to mock data, confirmation
  - 5.5 deny  → resume, refund status unchanged, denial communicated
  - 5.6 apply_refund is idempotent (no double refund on replay)
  - 5.7 no decision → refund not finalized (the paused state mutates nothing)
  - 1.1 a product question still routes to product_advisor
"""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

import app
import store
from app import graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ai(content="", tool_calls=None):
    """Build an AIMessage, optionally carrying tool calls."""
    return AIMessage(content=content, tool_calls=tool_calls or [])


def refund_tool_call(order_number="LUMA-1001", amount=30.0,
                     reason="Product caused irritation", call_id="toolu_test_1"):
    """An AIMessage that emits a request_refund signal call (drives the gate)."""
    return make_ai(content="", tool_calls=[{
        "name": "request_refund",
        "args": {"order_number": order_number, "amount": amount, "reason": reason},
        "id": call_id,
    }])


def drive_to_gate(thread_id, mock_llm, mock_returns, confirmation="Your refund is being processed."):
    """Configure the mocks and invoke the graph so it pauses at human_approval.

    `mock_returns.invoke` is scripted with two responses: the request_refund
    signal call (before the interrupt) and a plain confirmation reply (after
    the refund node loops back into returns_agent on resume).
    Returns the run config.
    """
    config = {"configurable": {"thread_id": thread_id}}
    mock_llm.invoke.return_value = make_ai("returns")
    mock_returns.invoke.side_effect = [refund_tool_call(), make_ai(confirmation)]
    graph.invoke(
        {"messages": [("user", "I'd like a refund for order LUMA-1001")]},
        config=config,
    )
    return config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_order_state():
    """Reset LUMA-1001's refund status before and after each test for isolation."""
    store._data["orders"]["LUMA-1001"]["refund_status"] = None
    yield
    store._data["orders"]["LUMA-1001"]["refund_status"] = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_graph_pauses_at_human_approval():
    """Req 5.1 / 5.7: the graph pauses at human_approval before applying anything."""
    with patch.object(app, "llm") as mock_llm, \
         patch.object(app, "returns_agent_llm") as mock_returns:
        config = drive_to_gate("hitl-pause", mock_llm, mock_returns)

        state = graph.get_state(config)
        # The interrupt pauses execution with human_approval queued as next.
        assert "human_approval" in state.next
        # Req 5.7: nothing finalized while awaiting a decision.
        assert store.refund_status("LUMA-1001") is None


def test_resume_approve_applies_refund():
    """Req 5.4: approving resumes the graph and mutates the refund status."""
    with patch.object(app, "llm") as mock_llm, \
         patch.object(app, "returns_agent_llm") as mock_returns:
        config = drive_to_gate("hitl-approve", mock_llm, mock_returns,
                               confirmation="Refund approved and processed!")

        # Sanity: paused and not yet applied.
        assert "human_approval" in graph.get_state(config).next
        assert store.refund_status("LUMA-1001") is None

        graph.invoke(
            Command(resume={"approved": True, "reviewer": "manager-demo"}),
            config=config,
        )

        # Data mutated and the flow has run to completion.
        assert store.refund_status("LUMA-1001") == "refunded"
        final = graph.get_state(config)
        assert final.next == ()
        # The pending_refund decision was recorded as approved.
        assert final.values["pending_refund"]["approved"] is True
        assert final.values["refund_applied"] is True


def test_resume_deny_keeps_data_unchanged():
    """Req 5.5: denying resumes the graph without changing the refund status."""
    with patch.object(app, "llm") as mock_llm, \
         patch.object(app, "returns_agent_llm") as mock_returns:
        config = drive_to_gate("hitl-deny", mock_llm, mock_returns,
                               confirmation="Sorry, the refund was denied.")

        assert "human_approval" in graph.get_state(config).next
        assert store.refund_status("LUMA-1001") is None

        graph.invoke(
            Command(resume={"approved": False, "reason": "Outside policy"}),
            config=config,
        )

        # No mutation — the order is still unrefunded.
        assert store.refund_status("LUMA-1001") is None
        final = graph.get_state(config)
        assert final.next == ()
        assert final.values["pending_refund"]["approved"] is False
        assert final.values["pending_refund"]["decision_reason"] == "Outside policy"


def test_refund_not_double_applied():
    """Req 5.6: apply_refund is idempotent — a replay does not refund twice."""
    pending = {
        "order_number": "LUMA-1001",
        "amount": 30.0,
        "tool_call_id": "toolu_test_1",
    }

    # Wrap the real store.apply_refund so we can count how often it actually runs.
    with patch.object(store, "apply_refund", wraps=store.apply_refund) as spy:
        # First application: refund_applied not yet set → applies the refund.
        first = app.apply_refund({"pending_refund": pending, "refund_applied": False})
        assert store.refund_status("LUMA-1001") == "refunded"
        assert first["refund_applied"] is True
        assert spy.call_count == 1

        # Replay with refund_applied already True → guard skips the mutation.
        app.apply_refund({"pending_refund": pending, "refund_applied": True})
        assert spy.call_count == 1  # not called a second time
        assert store.refund_status("LUMA-1001") == "refunded"

        # Even without the flag, the data-layer guard prevents a second refund
        # because the order is already "refunded".
        app.apply_refund({"pending_refund": pending, "refund_applied": False})
        assert spy.call_count == 1
        assert store.refund_status("LUMA-1001") == "refunded"


def test_product_question_routes_to_product_advisor():
    """Req 1.1: a product question still routes to the product_advisor branch."""
    advisor_reply = "Our Hydra Wash Shampoo is sulfate-free and great for dry hair."
    config = {"configurable": {"thread_id": "product-route"}}

    with patch.object(app, "llm") as mock_llm, \
         patch.object(app, "product_advisor_llm") as mock_advisor:
        mock_llm.invoke.return_value = make_ai("product_advice")
        mock_advisor.invoke.return_value = make_ai(advisor_reply)

        graph.invoke(
            {"messages": [("user", "Which shampoo is best for dry hair?")]},
            config=config,
        )

        state = graph.get_state(config)
        assert state.values["intent"] == "product_advice"
        # The product_advisor's reply is the final assistant message.
        assert state.values["messages"][-1].content == advisor_reply
        # The product advisor LLM was consulted; the returns branch was not entered.
        mock_advisor.invoke.assert_called_once()
        assert state.values.get("active_flow") != "returns"
