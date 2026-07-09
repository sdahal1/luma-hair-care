"""Unit tests for the mock data store (store.py)."""

import pytest
from datetime import date, timedelta
from unittest.mock import patch

import store


class TestGetOrder:
    """Test order lookup (found / not found)."""

    def test_found(self):
        order = store.get_order("LUMA-1001")
        assert order is not None
        assert order["customer_name"] == "Jordan Lee"
        assert order["email"] == "jordan@example.com"
        assert order["status"] == "delivered"

    def test_not_found(self):
        order = store.get_order("LUMA-9999")
        assert order is None

    def test_not_found_empty_string(self):
        order = store.get_order("")
        assert order is None


class TestReturnPolicy:
    """Test eligibility: in-window vs out-of-window vs already refunded."""

    def test_eligible_in_window(self):
        # Derive fake_today from LUMA-1001's actual purchase date + 14 days so
        # this test stays robust if the mock purchase date changes.
        purchased = date.fromisoformat(store.get_order("LUMA-1001")["purchased"])
        fake_today = purchased + timedelta(days=14)
        with patch("store.date") as mock_date:
            mock_date.today.return_value = fake_today
            mock_date.fromisoformat = date.fromisoformat
            result = store.return_policy("LUMA-1001")

        assert result["eligible"] is True
        assert result["days_since_purchase"] == 14
        assert result["window_days"] == 30

    def test_ineligible_out_of_window(self):
        # Patch today so LUMA-1002 (purchased 2025-03-15) is way past 30 days.
        fake_today = date(2025, 7, 15)
        with patch("store.date") as mock_date:
            mock_date.today.return_value = fake_today
            mock_date.fromisoformat = date.fromisoformat
            result = store.return_policy("LUMA-1002")

        assert result["eligible"] is False
        assert "30-day return window has passed" in result["reason"]
        assert result["days_since_purchase"] > 30

    def test_ineligible_already_refunded(self):
        # LUMA-1003 is already refunded.
        result = store.return_policy("LUMA-1003")
        assert result["eligible"] is False
        assert "already been refunded" in result["reason"]

    def test_order_not_found(self):
        result = store.return_policy("LUMA-9999")
        assert result["eligible"] is False
        assert "Order not found" in result["reason"]


class TestRefundStatus:
    """Test refund status lookup."""

    def test_no_refund(self):
        assert store.refund_status("LUMA-1001") is None

    def test_refunded(self):
        assert store.refund_status("LUMA-1003") == "refunded"

    def test_order_not_found(self):
        assert store.refund_status("LUMA-9999") is None


class TestRefundAmount:
    """Test refund amount computation."""

    def test_single_item(self):
        # LUMA-1001: 1 x $30
        assert store.refund_amount("LUMA-1001") == 30.0

    def test_multiple_qty(self):
        # LUMA-1002: 2 x $32
        assert store.refund_amount("LUMA-1002") == 64.0

    def test_multiple_items(self):
        # LUMA-1003: 1x$22 + 1x$24 = $46
        assert store.refund_amount("LUMA-1003") == 46.0

    def test_order_not_found(self):
        assert store.refund_amount("LUMA-9999") == 0.0


class TestApplyRefund:
    """Test apply_refund idempotency."""

    def test_apply_refund_success(self):
        # Reset LUMA-1001 state for the test.
        store._data["orders"]["LUMA-1001"]["refund_status"] = None

        result = store.apply_refund("LUMA-1001")
        assert result["success"] is True
        assert store.get_order("LUMA-1001")["refund_status"] == "refunded"

    def test_apply_refund_idempotent(self):
        # Ensure it's already refunded from previous test (or set it).
        store._data["orders"]["LUMA-1001"]["refund_status"] = "refunded"

        result = store.apply_refund("LUMA-1001")
        assert result["success"] is True
        assert "no-op" in result["message"]
        # Status unchanged.
        assert store.get_order("LUMA-1001")["refund_status"] == "refunded"

    def test_apply_refund_order_not_found(self):
        result = store.apply_refund("LUMA-9999")
        assert result["success"] is False
        assert "not found" in result["message"].lower()

    def teardown_method(self):
        # Reset LUMA-1001 to its original state after each test.
        if "LUMA-1001" in store._data.get("orders", {}):
            store._data["orders"]["LUMA-1001"]["refund_status"] = None
