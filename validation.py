# validation.py
"""
Validation functions for the order parsing agent.
Provides structural validation (JSON parsing, Pydantic)
Semantic validation (LLM-as-judge) for parsed orders.
"""

import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from prompts import VALIDATION_PROMPT
from schemas import FailedRecord, Order

logger = logging.getLogger(__name__)


def parse_json_response(content: str) -> dict | None:
    """
    Extract JSON from LLM response, handling markdown fences.

    Args:
        content: Raw LLM response string.

    Returns:
        Parsed JSON as dict, or None if parsing fails.
    """
    text = content.strip()

    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
            logger.debug("Extracted JSON from markdown fence")

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed: %s", e)
        return None


def validate_schema(data: dict) -> tuple[list[Order], list[str]]:
    """
    Validate parsed data against Pydantic Order model.

    Args:
        data: Dict with "orders" key containing list of order dicts.

    Returns:
        Tuple of (valid Order objects, list of error messages).
    """
    valid_orders = []
    errors = []

    orders_data = data.get("orders", [])
    if not isinstance(orders_data, list):
        logger.warning("Expected 'orders' to be a list, got %s", type(orders_data))
        return [], ["'orders' field is not a list"]

    for i, order_data in enumerate(orders_data):
        try:
            order = Order.model_validate(order_data)
            valid_orders.append(order)
        except Exception as e:
            order_id = order_data.get("orderId", f"index_{i}")
            error_msg = f"Order {order_id}: {e}"
            errors.append(error_msg)
            logger.debug("Schema validation failed for order %s: %s", order_id, e)

    logger.info(
        "Schema validation: %d valid, %d errors",
        len(valid_orders),
        len(errors),
    )
    return valid_orders, errors


async def validate_batch(
    parsed_orders: list[dict],
    raw_orders: list[str],
    llm: ChatOpenAI,
    semaphore: asyncio.Semaphore,
) -> tuple[list[dict], list[FailedRecord]]:
    """
    Send parsed orders and matching raw orders to LLM for semantic validation.

    Args:
        parsed_orders: List of parsed order dicts.
        raw_orders: List of raw order strings.
        llm: Async LLM client.
        semaphore: Concurrency control semaphore.

    Returns:
        Tuple of (valid order dicts, list of FailedRecord objects).
    """
    if not parsed_orders:
        logger.debug("No parsed orders to validate")
        return [], []

    parsed_ids = {o.get("orderId") for o in parsed_orders}
    matching_raw = [
        r for r in raw_orders if any(f"Order {pid}:" in r for pid in parsed_ids)
    ]

    logger.debug(
        "Validating %d parsed orders against %d matching raw orders",
        len(parsed_orders),
        len(matching_raw),
    )

    raw_text = "\n".join(matching_raw)
    parsed_text = json.dumps(parsed_orders)

    prompt = VALIDATION_PROMPT.format(
        raw_orders=raw_text,
        parsed_orders=parsed_text,
    )

    async with semaphore:
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            result = parse_validation_response(response.content, parsed_orders)
            return result
        except Exception as e:
            logger.error("LLM validation call failed: %s", e)
            return parsed_orders, []


def parse_validation_response(
    content: str,
    parsed_orders: list[dict],
) -> tuple[list[dict], list[FailedRecord]]:
    """
    Parse LLM validation response into valid and invalid orders.

    Args:
        content: Raw LLM response string.
        parsed_orders: Original parsed orders for lookup.

    Returns:
        Tuple of (valid order dicts, list of FailedRecord objects).
    """
    data = parse_json_response(content)

    if data is None:
        logger.warning("Could not parse validation response, rejecting batch")
        failed_records = [
            FailedRecord(
                orderId=o.get("orderId"),
                rawSnippet=None,
                failureType="mismatch",
                reason="Validation response unparseable",
            )
            for o in parsed_orders
        ]
        return [], failed_records

    valid_ids = {item.get("orderId") for item in data.get("valid", [])}
    invalid_items = data.get("invalid", [])

    valid_orders = [o for o in parsed_orders if o.get("orderId") in valid_ids]
    failed_records = []

    for item in invalid_items:
        failed_records.append(
            FailedRecord(
                orderId=item.get("orderId"),
                rawSnippet=item.get("rawSnippet"),
                failureType=item.get("failureType", "mismatch"),
                reason=item.get("reason", "Unknown"),
            )
        )

    logger.info(
        "Validation result: %d valid, %d invalid",
        len(valid_orders),
        len(failed_records),
    )
    return valid_orders, failed_records
