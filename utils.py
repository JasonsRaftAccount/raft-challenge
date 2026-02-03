# utils.py
"""
Utility functions for the order parsing agent.
Provides logging configuration and shared helper.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for console and file output.

    Args:
        level: Logging level (default: INFO).
    """
    root = logging.getLogger()

    if root.handlers:
        return

    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler("agent.log")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
