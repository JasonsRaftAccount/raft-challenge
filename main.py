# file: main.py
"""
CLI entry point for the order parsing agent.

Usage:
    python main.py "Show me all orders from Ohio over $500"
    python main.py "orders where buyer is John"
    python main.py  # defaults to "Show me all orders"
"""

import sys
import json
from agent import run_agent
from config import setup_logging

setup_logging()


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Show me all orders"
    result = run_agent(query)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
