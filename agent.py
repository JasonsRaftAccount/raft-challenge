# file: agent.py
"""
LangGraph Agent for parsing and filtering order data.

START --> [fetch] (Call API, get raw orders)
[fetch] --> [parse] (LLM parses + filters orders + check structure)
[parse] --> [validate] (Check LLM output against regex anchors)
[validate] --> END

With queries like, "Show me all orders from Ohio over $500"
1. Fetches raw order data from the API
2. Sends raw data + user query to LLM for parsing AND filtering
3. Validates LLM output against regex anchors (slow but high fidelity)
4. Returns structured JSON
"""

import json
import re
import logging
from typing import TypedDict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

import config
from api_client import fetch_orders, APIError
from schemas import safe_parse_response
from validation import build_anchor_index, validate_order

logger = logging.getLogger(__name__)

# SYSTEM PROMPT - Defines how LLM should parse and filter orders

SYSTEM_PROMPT = """You parse multiple raw order strings into JSON and filter based on user criteria.

INPUT: Raw order strings (e.g., "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop (4.2*), Returned=No") + a natural language query (e.g., "orders from Ohio over $500")
OUTPUT:
- Return compact JSON with no extra whitespace, newlines, or indentation.
- Example: {"orders":[{"orderId":"1001","buyer":"John Davis","city":"Columbus","state":"OH","total":742.10,"items":[{"name":"laptop","rating":4.2}],"returned":false}]}

Parsing rules:
- orderId: digits only, as string
- state: 2-letter uppercase
- total: number without $ sign
- rating: number before * symbol
- returned: "Yes" = true, "No" = false

Field inference:
- Field names or formatting in the raw data may change between requests.
- Use context to infer the correct mapping. For example, "Customer:" means the same as "Buyer=", "Amount:" means the same as "Total=".
- If a field is genuinely absent from a record, omit that entire order from the output rather than guessing.

Filtering rules:
- Apply the user's query as a filter
- Only include orders that match ALL criteria in the query
- If no orders match, return {"orders": []}
"""

# STATE - Data that flows through the graph


class AgentState(TypedDict):
    query: str  # User's natural language query
    raw_orders: list[str]  # Raw strings from API
    anchor_index: dict  # Regex ground truth for validation
    parsed_orders: list[dict]  # LLM output
    valid_orders: list[dict]  # After validation
    error: Optional[str]


# LLM


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.LLM_MODEL,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        max_retries=3,
        request_timeout=60,  # TODO: move to configs
    )


# NODES - Functions / tools


def fetch_node(state: AgentState) -> dict:
    """Fetch orders from API and build anchor index."""
    try:
        raw_orders = fetch_orders()
        anchor_index = build_anchor_index(raw_orders)

        extraction_rate = len(anchor_index) / len(raw_orders) if raw_orders else 0
        logger.info(
            f"Fetched {len(raw_orders)} orders, anchor extraction rate: {extraction_rate:.0%}"
        )

        if extraction_rate < 1.0:
            logger.warning(
                f"Anchor extraction failed for {len(raw_orders) - len(anchor_index)} orders — "
                "format may have changed"
            )

        return {
            "raw_orders": raw_orders,
            "anchor_index": anchor_index,
        }
    except APIError as e:
        return {"error": str(e)}


def parse_node(state: AgentState) -> dict:
    """Send raw orders + user query to LLM for parsing and filtering."""
    if state.get("error") or not state["raw_orders"]:
        return {"parsed_orders": []}

    # Chunk if needed for context window
    chunks = [
        state["raw_orders"][i : i + config.CHUNK_SIZE]
        for i in range(0, len(state["raw_orders"]), config.CHUNK_SIZE)
    ]

    llm = get_llm()
    all_orders = []

    for chunk in chunks:
        # User query + raw data → LLM does parsing AND filtering
        user_msg = f"Query: {state['query']}\n\nRaw orders:\n" + "\n".join(chunk)

        try:
            response = llm.invoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)]
            )
            content = response.content.strip()
            logger.info(f"Raw LLM response length: {len(content)}")
            logger.debug(f"Raw LLM response:\n{content[:1000]}")  # First 1000 chars

            # Strip markdown code blocks if present (common w/ gpt family)
            code_block_match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
            if code_block_match:
                content = code_block_match.group(1).strip()

            # Skip empty responses
            if not content.strip():
                logger.warning("Empty LLM response, skipping chunk")
                continue

            # Pydantic structure validation
            parsed = safe_parse_response(json.loads(content))
            if parsed.error:
                logger.warning(f"Pydantic validation: {parsed.error}")
                continue

            # Convert validated Orders to dicts for regex validation
            orders = [o.model_dump() for o in parsed.orders]
            all_orders.extend(orders)
        except json.JSONDecodeError as e:
            logger.warning(f"Parse error: {e}")
            logger.debug(f"Raw LLM response:\n{content[:500]}")  # Log first 500 chars
        except Exception as e:
            logger.warning(f"Unexpected error: {e}")

    return {"parsed_orders": all_orders}


def validate_node(state: AgentState) -> dict:
    """Validate LLM output against regex anchors."""
    if state.get("error") or not state["parsed_orders"]:
        return {"valid_orders": []}

    anchor_index = state["anchor_index"]
    parsed_orders = state["parsed_orders"]

    # Schema change detection: if most anchors failed, log it
    if anchor_index and len(anchor_index) < len(state["raw_orders"]) * 0.5:
        logger.warning(
            f"Anchor extraction only succeeded for {len(anchor_index)}/{len(state['raw_orders'])} orders "
            "— possible API schema change. Orders without anchors will skip regex validation."
        )

    valid = []
    skipped = []

    for order in parsed_orders:
        order_id = str(order.get("orderId"))

        if order_id in anchor_index:
            # Anchor exists — full regex validation
            mismatches = validate_order(order, anchor_index[order_id])
            if mismatches:
                logger.warning(f"Order {order_id} rejected — mismatches: {mismatches}")
                skipped.append(order)
            else:
                valid.append(order)
        else:
            # No anchor (schema change or new format) — trust Pydantic validation only
            logger.info(
                f"Order {order_id} has no anchor — accepted via Pydantic validation only"
            )
            valid.append(order)

    if skipped:
        logger.warning(f"Skipped {len(skipped)} orders with validation mismatches")

    return {"valid_orders": valid}


# GRAPH - Connect nodes with edges pretty pretty cool


def build_graph():
    """
    Build the agent graph:
    START → fetch → parse → validate → END
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


# Agent API

# build once at mod level
_graph = build_graph()


def run_agent(query: str, full_output: bool = False) -> dict:
    """
    Run the agent with a natural language query.

    Args:
        query: Natural language query, e.g., "Show me all orders from Ohio over $500"
        full_output: If True, return all fields. If False, return challenge format only.

    Returns:
        {"orders": [...]}
    """
    result = _graph.invoke(
        {
            "query": query,
            "raw_orders": [],
            "anchor_index": {},
            "parsed_orders": [],
            "valid_orders": [],
            "error": None,
        }
    )

    if result.get("error"):
        return {"orders": [], "error": result["error"]}

    valid_orders = result.get("valid_orders", [])

    if full_output:
        return {"orders": valid_orders}

    # Challenge format (only required fields)
    return {
        "orders": [
            {
                "orderId": o["orderId"],
                "buyer": o["buyer"],
                "state": o["state"],
                "total": o["total"],
            }
            for o in valid_orders
        ]
    }


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Show me all orders"
    print(json.dumps(run_agent(query), indent=2))
