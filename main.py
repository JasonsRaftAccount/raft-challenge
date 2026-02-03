# main.py
"""
CLI entry point for the order parsing agent.

Usage:
    python main.py "Show me all orders from Ohio over $500"
    python main.py --full "Show me all orders"
    python main.py  # defaults to "Show me all orders"
"""

import argparse
import json
import logging
import sys

from agent import run_agent
from utils import setup_logging


def main() -> None:
    """Parse arguments, run agent, and output results."""
    parser = argparse.ArgumentParser(
        description="Parse orders using natural language queries."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Show me all orders",
        help="Natural language query (default: 'Show me all orders')",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Return full order details instead of summary",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Query: %s", args.query)

    try:
        result = run_agent(args.query)

        if args.full:
            output = result.to_analytics_data().model_dump()
        else:
            output = result.to_query_response().model_dump()

        print(json.dumps(output, indent=2))

    except Exception as e:
        logger.error("Agent failed: %s", e)
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
