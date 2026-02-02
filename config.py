# file: config.py
"""
Configuration for the order parsing agent and logging behavior.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()


# centralized logging setup (easier for scope)
def setup_logging(level=logging.DEBUG):
    """Configure logging once. Safe to call multiple times (idempotent)."""
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.FileHandler("sys.log")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


# Provider: openrouter or ollama for local
PROVIDER = os.getenv("PROVIDER", "openrouter")

# LLM Settings
if PROVIDER == "ollama":
    LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    LLM_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    LLM_API_KEY = "ollama"
else:
    LLM_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:exacto")
    LLM_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# App Settings
DUMMY_API_URL = os.getenv("DUMMY_API_URL", "http://localhost:5001")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "30"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))

# Validate required config
if PROVIDER == "openrouter" and not LLM_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is required when PROVIDER=openrouter. "
        "Set it in your .env file."
    )

if __name__ == "__main__":
    print(f"Provider: {PROVIDER}")
    print(f"Model: {LLM_MODEL}")
    print(f"Base URL: {LLM_BASE_URL}")
    # print(f"API Key set: {bool(LLM_API_KEY)}")
