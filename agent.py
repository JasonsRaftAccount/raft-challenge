# agent.py
"""
LangGraph agent for the order parsing agent.
Orchestrates fetch, parse, and validate stages with parallel processing and retry.
"""

import asyncio
import logging
from typing import Any, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import config
from clients import APIError, fetch_orders_async, get_async_llm
from prompts import SYSTEM_PROMPT
from schemas import (
    AgentResult,
    DeadLetterQueue,
    Order,
    QueryMeta,
    RawOrderStore,
)
from validation import (
    parse_json_response,
    validate_batch,
    validate_schema,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State flowing through the LangGraph pipeline."""

    query: str
    raw_store: Optional[dict]
    parsed_orders: list[dict]
    valid_orders: list[dict]
    dlq: dict
    error: Optional[str]


async def parse_batch(
    chunk: list[str],
    query: str,
    llm: Any,
    semaphore: asyncio.Semaphore,
    batch_index: int,
) -> tuple[list[Order], Optional[str]]:
    """
    Parse a batch of raw orders via LLM.

    Args:
        chunk: List of raw order strings.
        query: User's natural language query.
        llm: Async LLM client.
        semaphore: Concurrency control semaphore.
        batch_index: Index of this batch for logging.

    Returns:
        Tuple of (list of valid Order objects, error message if failed).
    """
    user_content = f"Query: {query}\n\nRaw orders:\n" + "\n".join(chunk)

    async with semaphore:
        logger.debug("Parsing batch %d with %d orders", batch_index, len(chunk))
        try:
            response = await llm.ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ]
            )

            data = parse_json_response(response.content)
            if data is None:
                return [], "Failed to parse LLM response as JSON"

            orders, errors = validate_schema(data)

            if errors:
                logger.warning(
                    "Batch %d had %d schema errors: %s",
                    batch_index,
                    len(errors),
                    errors[:3],
                )

            logger.info(
                "Batch %d parsed: %d valid orders",
                batch_index,
                len(orders),
            )
            return orders, None

        except Exception as e:
            logger.error("Batch %d parse failed: %s", batch_index, e)
            return [], str(e)


async def parse_batch_with_retry(
    chunk: list[str],
    query: str,
    llm: Any,
    semaphore: asyncio.Semaphore,
    batch_index: int,
    max_retries: int = config.MAX_RETRIES,
    base_delay: float = config.RETRY_BASE_DELAY,
) -> tuple[list[Order], Optional[str], int]:
    """
    Parse a batch with exponential backoff retry.

    Args:
        chunk: List of raw order strings.
        query: User's natural language query.
        llm: Async LLM client.
        semaphore: Concurrency control semaphore.
        batch_index: Index of this batch for logging.
        max_retries: Maximum retry attempts.
        base_delay: Base delay for exponential backoff.

    Returns:
        Tuple of (valid orders, error message if all retries failed, attempt count).
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        orders, error = await parse_batch(chunk, query, llm, semaphore, batch_index)

        if error is None:
            return orders, None, attempt

        last_error = error
        if attempt < max_retries:
            delay = base_delay * (2 ** (attempt - 1))  # exp backoff
            logger.warning(
                "Batch %d attempt %d failed, retrying in %.1fs: %s",
                batch_index,
                attempt,
                delay,
                error,
            )
            await asyncio.sleep(delay)

    logger.error(
        "Batch %d failed after %d attempts: %s",
        batch_index,
        max_retries,
        last_error,
    )
    return [], last_error, max_retries


async def fetch_node(state: AgentState) -> dict:
    """
    Fetch raw orders from the API.

    Args:
        state: Current agent state.

    Returns:
        Updated state with raw_store populated.
    """
    logger.info("Fetching orders for query: %s", state["query"])
    try:
        raw_orders = await fetch_orders_async()
        raw_store = RawOrderStore(orders=raw_orders)
        logger.info("Fetched %d orders", len(raw_orders))
        return {"raw_store": raw_store.model_dump()}
    except APIError as e:
        logger.error("Fetch failed: %s", e)
        return {"error": str(e)}


async def parse_node(state: AgentState) -> dict:
    """
    Parse raw orders in parallel batches.

    Args:
        state: Current agent state with raw_store.

    Returns:
        Updated state with parsed_orders and dlq.
    """
    if state.get("error"):
        return {"parsed_orders": [], "dlq": DeadLetterQueue().model_dump()}

    raw_store = RawOrderStore.model_validate(state["raw_store"])
    query = state["query"]
    dlq = DeadLetterQueue()

    llm = get_async_llm()
    semaphore = asyncio.Semaphore(config.PARSE_CONCURRENCY)

    total_batches = raw_store.total_batches(config.CHUNK_SIZE)
    logger.info(
        "Parsing %d orders in %d batches (chunk size: %d)",
        len(raw_store.orders),
        total_batches,
        config.CHUNK_SIZE,
    )

    tasks = []
    for i in range(total_batches):
        chunk = raw_store.get_batch(i, config.CHUNK_SIZE)
        tasks.append(parse_batch_with_retry(chunk, query, llm, semaphore, i))

    results = await asyncio.gather(*tasks)

    all_orders = []
    for i, (orders, error, attempts) in enumerate(results):
        if error:
            chunk = raw_store.get_batch(i, config.CHUNK_SIZE)
            dlq.add_batch_failure(
                batch_index=i,
                raw_orders=chunk,
                error=error,
                attempts=attempts,
            )
        else:
            all_orders.extend(orders)

    logger.info(
        "Parse complete: %d orders parsed, %d batches failed",
        len(all_orders),
        len(dlq.failed_batches),
    )

    return {
        "parsed_orders": [o.model_dump() for o in all_orders],
        "dlq": dlq.model_dump(),
    }


async def validate_node(state: AgentState) -> dict:
    """
    Validate parsed orders via LLM-as-judge.

    Args:
        state: Current agent state with parsed_orders.

    Returns:
        Updated state with valid_orders and dlq.
    """
    if state.get("error"):
        return {"valid_orders": []}

    parsed_orders = state["parsed_orders"]
    raw_store = RawOrderStore.model_validate(state["raw_store"])
    dlq = DeadLetterQueue.model_validate(state["dlq"])

    if not parsed_orders:
        logger.warning("No parsed orders to validate")
        return {"valid_orders": [], "dlq": dlq.model_dump()}

    llm = get_async_llm()
    semaphore = asyncio.Semaphore(config.VALIDATE_CONCURRENCY)

    total_batches = (len(parsed_orders) + config.CHUNK_SIZE - 1) // config.CHUNK_SIZE
    logger.info(
        "Validating %d orders in %d batches",
        len(parsed_orders),
        total_batches,
    )

    tasks = []
    for i in range(total_batches):
        start = i * config.CHUNK_SIZE
        end = start + config.CHUNK_SIZE
        parsed_chunk = parsed_orders[start:end]
        # Pass ALL raw orders - validate_batch filters to matching IDs
        tasks.append(validate_batch(parsed_chunk, raw_store.orders, llm, semaphore))

    results = await asyncio.gather(*tasks)

    all_valid = []
    for valid_orders, failed_records in results:
        all_valid.extend(valid_orders)
        for record in failed_records:
            dlq.failed_records.append(record)

    logger.info(
        "Validation complete: %d valid, %d failed records",
        len(all_valid),
        len(dlq.failed_records),
    )

    return {
        "valid_orders": all_valid,
        "dlq": dlq.model_dump(),
    }


def build_graph() -> StateGraph:
    """
    Construct the LangGraph state machine.

    Returns:
        Compiled StateGraph ready for execution.
    """
    builder = StateGraph(AgentState)

    builder.add_node("fetch", fetch_node)
    builder.add_node("parse", parse_node)
    builder.add_node("validate", validate_node)

    builder.add_edge(START, "fetch")
    builder.add_edge("fetch", "parse")
    builder.add_edge("parse", "validate")
    builder.add_edge("validate", END)

    return builder.compile()


_graph = build_graph()


async def run_agent_async(query: str) -> AgentResult:
    """
    Execute the agent pipeline asynchronously.

    Args:
        query: Natural language query for filtering orders.

    Returns:
        AgentResult with valid orders, failures, and metadata.
    """
    logger.info("Running agent with query: %s", query)

    initial_state: AgentState = {
        "query": query,
        "raw_store": None,
        "parsed_orders": [],
        "valid_orders": [],
        "dlq": DeadLetterQueue().model_dump(),
        "error": None,
    }

    result = await _graph.ainvoke(initial_state)

    raw_store = (
        RawOrderStore.model_validate(result["raw_store"])
        if result["raw_store"]
        else RawOrderStore()
    )
    dlq = DeadLetterQueue.model_validate(result["dlq"])

    valid_orders = [Order.model_validate(o) for o in result["valid_orders"]]

    meta = QueryMeta(
        total_raw=len(raw_store.orders),
        total_parsed=len(result["parsed_orders"]),
        total_valid=len(valid_orders),
        total_failed=dlq.total_failures,
    )

    logger.info(
        "Agent complete: %d/%d orders valid (%.1f%% success rate)",
        meta.total_valid,
        meta.total_raw,
        meta.success_rate * 100,
    )

    return AgentResult(
        raw_store=raw_store,
        valid_orders=valid_orders,
        dlq=dlq,
        meta=meta,
    )


def run_agent(query: str) -> AgentResult:
    """
    Execute the agent pipeline synchronously.

    Args:
        query: Natural language query for filtering orders.

    Returns:
        AgentResult with valid orders, failures, and metadata.
    """
    return asyncio.run(run_agent_async(query))
