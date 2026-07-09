"""
Mock data-access layer for the Luma returns assistant.

Loads order data from a local JSON file at startup and provides
functions for order lookup, return-policy checks, refund status,
refund amount computation, and applying refunds (idempotent).

This module is intentionally thin so it can be swapped for a real
API later without changing the graph or tool layer.
"""

import json
import os
from datetime import date, datetime
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "mock_data" / "orders.json"

# In-process store loaded once at import time.
_data: dict = {}


def _load() -> None:
    """Load the JSON data file into memory."""
    global _data
    with open(_DATA_PATH, "r") as f:
        _data = json.load(f)


# Load on import so data is available immediately.
_load()

RETURN_WINDOW_DAYS = 30


def get_order(order_number: str) -> dict | None:
    """Return the order dict for the given order number, or None if not found."""
    return _data.get("orders", {}).get(order_number)


def return_policy(order_number: str) -> dict:
    """
    Check whether an order is eligible for a refund.

    Returns a dict with:
      - eligible: bool
      - reason: str explaining why
      - days_since_purchase: int
      - window_days: int (the policy window, always 30)
    """
    order = get_order(order_number)
    if order is None:
        return {
            "eligible": False,
            "reason": "Order not found.",
            "days_since_purchase": 0,
            "window_days": RETURN_WINDOW_DAYS,
        }

    purchased = date.fromisoformat(order["purchased"])
    days_since = (date.today() - purchased).days

    # Already refunded
    if order.get("refund_status") == "refunded":
        return {
            "eligible": False,
            "reason": "This order has already been refunded.",
            "days_since_purchase": days_since,
            "window_days": RETURN_WINDOW_DAYS,
        }

    # Must be delivered to be eligible
    if order.get("status") != "delivered":
        return {
            "eligible": False,
            "reason": f"Order status is '{order.get('status')}'; only delivered orders are eligible.",
            "days_since_purchase": days_since,
            "window_days": RETURN_WINDOW_DAYS,
        }

    # 30-day window
    if days_since > RETURN_WINDOW_DAYS:
        return {
            "eligible": False,
            "reason": f"The 30-day return window has passed ({days_since} days since purchase).",
            "days_since_purchase": days_since,
            "window_days": RETURN_WINDOW_DAYS,
        }

    return {
        "eligible": True,
        "reason": "Order is eligible for a refund.",
        "days_since_purchase": days_since,
        "window_days": RETURN_WINDOW_DAYS,
    }


def refund_status(order_number: str) -> str | None:
    """Return the current refund_status for an order, or None if order not found."""
    order = get_order(order_number)
    if order is None:
        return None
    return order.get("refund_status")


def refund_amount(order_number: str) -> float:
    """Compute the total refund amount (sum of price * qty for each line item)."""
    order = get_order(order_number)
    if order is None:
        return 0.0
    return sum(item["price"] * item["qty"] for item in order.get("items", []))


def apply_refund(order_number: str) -> dict:
    """
    Apply a refund to an order (set refund_status to 'refunded').

    Idempotent: if the order is already refunded, this is a no-op.
    Returns a dict with 'success' and 'message'.
    """
    order = get_order(order_number)
    if order is None:
        return {"success": False, "message": "Order not found."}

    if order.get("refund_status") == "refunded":
        # Idempotent — already refunded, no-op.
        return {"success": True, "message": "Refund already applied (no-op)."}

    # Mutate the in-memory data.
    order["refund_status"] = "refunded"
    return {"success": True, "message": f"Refund applied for order {order_number}."}
