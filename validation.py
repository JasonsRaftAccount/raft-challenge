# file: validation.py
"""
Regex-based factual validation for LLM outputs.

Purpose: Verify that LLM-parsed values match the source data.
Pydantic validates structure; this validates accuracy.

Installation:
    No additional dependencies (uses stdlib re module)
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OrderAnchors:
    """Ground truth values extracted from raw order string via regex."""

    orderId: str
    total: float
    state: str
    returned: bool
    raw_string: str


class AnchorExtractionError(Exception):
    """Raised when regex fails to extract required fields from raw order."""

    pass


def _extract(pattern: str, text: str, field: str) -> str:
    """Extract first capture group or raise with field name."""
    if match := re.search(pattern, text):
        return match.group(1)
    raise AnchorExtractionError(f"Could not extract {field}")


def extract_anchors(raw_order: str) -> OrderAnchors:
    """Extract verifiable fields from raw order string using regex."""
    return OrderAnchors(
        orderId=_extract(r"Order (\d+):", raw_order, "orderId"),
        total=float(_extract(r"Total=\$([0-9.]+)", raw_order, "total")),
        state=_extract(r"Location=[^,]+,\s*([A-Z]{2})", raw_order, "state"),
        returned=_extract(r"Returned=(Yes|No)", raw_order, "returned") == "Yes",
        raw_string=raw_order,
    )


def build_anchor_index(raw_orders: list[str]) -> dict[str, OrderAnchors]:
    """Build lookup dict of order ID → anchors for batch validation."""
    index = {}
    for raw_order in raw_orders:
        try:
            anchor = extract_anchors(raw_order)
            index[anchor.orderId] = anchor
        except AnchorExtractionError as e:
            logger.warning(f"Skipping order: {e}")
    return index


def validate_order(
    parsed: dict, anchor: OrderAnchors, tolerance: float = 0.01
) -> list[str]:
    """
    Compare parsed order against anchor. Returns list of mismatches (empty if valid).
    """
    mismatches = []

    if str(parsed.get("orderId")) != anchor.orderId:
        mismatches.append(f"orderId: {parsed.get('orderId')} != {anchor.orderId}")

    if abs(parsed.get("total", 0) - anchor.total) > tolerance:
        mismatches.append(f"total: {parsed.get('total')} != {anchor.total}")

    if parsed.get("state", "").upper() != anchor.state:
        mismatches.append(f"state: {parsed.get('state')} != {anchor.state}")

    if parsed.get("returned") != anchor.returned:
        mismatches.append(f"returned: {parsed.get('returned')} != {anchor.returned}")

    return mismatches


# TODO: artifact of the previous parsing workflow, replaced in validate_node loop, can delete later.
def validate_parsed_orders(
    parsed_orders: list[dict], anchor_index: dict[str, OrderAnchors]
) -> tuple[list[dict], list[dict]]:
    """
    Validate batch of parsed orders. Returns (valid_orders, invalid_orders).
    """
    valid, invalid = [], []

    for order in parsed_orders:
        order_id = str(order.get("orderId"))

        if order_id not in anchor_index:
            logger.warning(f"Order {order_id} not in source data (hallucination?)")
            invalid.append(order)
            continue

        mismatches = validate_order(order, anchor_index[order_id])
        if mismatches:
            logger.warning(f"Order {order_id} mismatches: {mismatches}")
            invalid.append(order)
        else:
            valid.append(order)

    return valid, invalid
