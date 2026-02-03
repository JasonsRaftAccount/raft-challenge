# clients.py
"""
External service clients for the order parsing agent.

Provides async HTTP client for the Order API and factory for the
async LangChain LLM client.
"""

import logging

import httpx
from langchain_openai import ChatOpenAI

import config

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Raised when an API call fails."""

    pass


async def fetch_orders_async() -> list[str]:
    """
    Fetch all orders from the dummy API.

    Returns:
        List of raw order strings.

    Raises:
        APIError: If the request fails or times out.
    """
    logger.debug("Fetching orders from %s", config.DUMMY_API_URL)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{config.DUMMY_API_URL}/api/orders",
                timeout=30.0,
            )
            response.raise_for_status()
            orders = response.json().get("raw_orders", [])
            logger.info("Fetched %d orders from API", len(orders))
            return orders
    except httpx.TimeoutException as e:
        logger.error("API request timed out: %s", e)
        raise APIError(f"Request timed out: {e}")
    except httpx.HTTPStatusError as e:
        logger.error("API returned HTTP %d: %s", e.response.status_code, e)
        raise APIError(f"HTTP error {e.response.status_code}: {e}")
    except httpx.RequestError as e:
        logger.error("API request failed: %s", e)
        raise APIError(f"Request failed: {e}")


async def fetch_order_async(order_id: str) -> str | None:
    """
    Fetch a single order by ID from the dummy API.

    Args:
        order_id: The order ID to fetch.

    Returns:
        Raw order string if found, None if not found.

    Raises:
        APIError: If the request fails or times out.
    """
    logger.debug("Fetching order %s", order_id)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{config.DUMMY_API_URL}/api/order/{order_id}",
                timeout=10.0,
            )
            if response.status_code == 404:
                logger.warning("Order %s not found", order_id)
                return None
            response.raise_for_status()
            logger.debug("Fetched order %s", order_id)
            return response.json().get("raw_order")
    except httpx.TimeoutException as e:
        logger.error("Request for order %s timed out: %s", order_id, e)
        raise APIError(f"Request timed out: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(
            "Request for order %s returned HTTP %d", order_id, e.response.status_code
        )
        raise APIError(f"HTTP error {e.response.status_code}: {e}")
    except httpx.RequestError as e:
        logger.error("Request for order %s failed: %s", order_id, e)
        raise APIError(f"Request failed: {e}")


def get_async_llm() -> ChatOpenAI:
    """
    Create an async-capable LangChain LLM client.

    Returns:
        Configured ChatOpenAI instance for async operations.
    """
    logger.debug(
        "Creating LLM client: model=%s, base_url=%s, max_tokens=%d",
        config.LLM_MODEL,
        config.LLM_BASE_URL,
        config.MAX_TOKENS,
    )
    return ChatOpenAI(
        model=config.LLM_MODEL,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )
