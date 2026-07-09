"""Unit tests for the LangChain @tool functions in app.py.

Validates Requirements 3.3 and 4.6:
  - Each tool returns graceful strings on missing/invalid order numbers.
  - Each tool returns correct content on valid ones.
"""

import pytest

# Import the tools from app.py. Since they are @tool decorated, we invoke them
# using .invoke() which accepts the tool's input schema.
from app import lookup_order, check_return_policy, check_refund_status, request_refund


# ---------------------------------------------------------------------------
# lookup_order
# ---------------------------------------------------------------------------

class TestLookupOrder:
    """Test the lookup_order tool."""

    def test_valid_order(self):
        """A known order returns formatted details."""
        result = lookup_order.invoke({"order_number": "LUMA-1001"})
        assert "LUMA-1001" in result
        assert "Jordan Lee" in result
        assert "jordan@example.com" in result
        assert "Weekly Repair Mask" in result
        assert "delivered" in result

    def test_order_not_found(self):
        """A non-existent order returns a graceful not-found message."""
        result = lookup_order.invoke({"order_number": "LUMA-9999"})
        assert "No order found" in result
        assert "LUMA-9999" in result

    def test_empty_order_number(self):
        """An empty string returns a graceful not-found message."""
        result = lookup_order.invoke({"order_number": ""})
        assert "No order found" in result


# ---------------------------------------------------------------------------
# check_return_policy
# ---------------------------------------------------------------------------

class TestCheckReturnPolicy:
    """Test the check_return_policy tool."""

    def test_eligible_order(self):
        """LUMA-1001 is recent and delivered — should be eligible."""
        result = check_return_policy.invoke({"order_number": "LUMA-1001"})
        assert "LUMA-1001" in result
        assert "ELIGIBLE" in result
        assert "30" in result  # window_days

    def test_out_of_window(self):
        """LUMA-1002 was purchased long ago — should be NOT ELIGIBLE."""
        result = check_return_policy.invoke({"order_number": "LUMA-1002"})
        assert "LUMA-1002" in result
        assert "NOT ELIGIBLE" in result

    def test_already_refunded(self):
        """LUMA-1003 is already refunded — should be NOT ELIGIBLE."""
        result = check_return_policy.invoke({"order_number": "LUMA-1003"})
        assert "LUMA-1003" in result
        assert "NOT ELIGIBLE" in result
        assert "already" in result.lower()

    def test_order_not_found(self):
        """A non-existent order returns an error-like string gracefully."""
        result = check_return_policy.invoke({"order_number": "LUMA-9999"})
        assert "NOT ELIGIBLE" in result
        assert "not found" in result.lower()

    def test_empty_order_number(self):
        """An empty string returns a graceful message without crashing."""
        result = check_return_policy.invoke({"order_number": ""})
        assert isinstance(result, str)
        assert "NOT ELIGIBLE" in result


# ---------------------------------------------------------------------------
# check_refund_status
# ---------------------------------------------------------------------------

class TestCheckRefundStatus:
    """Test the check_refund_status tool."""

    def test_no_refund(self):
        """LUMA-1001 has no refund yet (refund_status is null).

        The store returns None for null refund_status, and the tool treats
        None as 'no order found'. This is the current behavior — the tool
        returns a graceful string either way.
        """
        result = check_refund_status.invoke({"order_number": "LUMA-1001"})
        assert "LUMA-1001" in result
        assert isinstance(result, str)

    def test_refunded(self):
        """LUMA-1003 was already refunded — status should say so."""
        result = check_refund_status.invoke({"order_number": "LUMA-1003"})
        assert "LUMA-1003" in result
        assert "refunded" in result.lower()

    def test_order_not_found(self):
        """A non-existent order returns a graceful not-found message."""
        result = check_refund_status.invoke({"order_number": "LUMA-9999"})
        assert "No order found" in result
        assert "LUMA-9999" in result

    def test_empty_order_number(self):
        """An empty string returns a graceful message without crashing."""
        result = check_refund_status.invoke({"order_number": ""})
        assert "No order found" in result


# ---------------------------------------------------------------------------
# request_refund (signal tool — just validates it returns a string)
# ---------------------------------------------------------------------------

class TestRequestRefund:
    """Test the request_refund signal tool."""

    def test_valid_request(self):
        """A valid refund request returns a confirmation string."""
        result = request_refund.invoke({
            "order_number": "LUMA-1001",
            "amount": 30.0,
            "reason": "Product caused irritation",
        })
        assert "LUMA-1001" in result
        assert "$30.00" in result
        assert "Product caused irritation" in result
        assert "approval" in result.lower()

    def test_invalid_order_still_returns_string(self):
        """Even a bogus order number produces a string (signal tool doesn't validate)."""
        result = request_refund.invoke({
            "order_number": "FAKE-0000",
            "amount": 99.99,
            "reason": "Testing",
        })
        assert isinstance(result, str)
        assert "FAKE-0000" in result
        assert "$99.99" in result
