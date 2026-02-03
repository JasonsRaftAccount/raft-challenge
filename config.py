"""
Configuration management for the order parsing agent.

Loads environment variables and provides validated settings to all modules.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# LLM Provider
PROVIDER = os.getenv("PROVIDER", "openrouter")

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:exacto")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Provider Selection
if PROVIDER == "openrouter":
    LLM_MODEL = OPENROUTER_MODEL
    LLM_BASE_URL = OPENROUTER_BASE_URL
    LLM_API_KEY = OPENROUTER_API_KEY
else:
    LLM_MODEL = OLLAMA_MODEL
    LLM_BASE_URL = OLLAMA_BASE_URL
    LLM_API_KEY = "ollama"

# LLM Parameters
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))

# Batch Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "25"))
PARSE_CONCURRENCY = int(os.getenv("PARSE_CONCURRENCY", "10"))
VALIDATE_CONCURRENCY = int(os.getenv("VALIDATE_CONCURRENCY", "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "10.0"))  # 10 sec retry

# API Configuration
DUMMY_API_URL = os.getenv("DUMMY_API_URL", "http://localhost:5001")

# Validation
if PROVIDER == "openrouter" and not LLM_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is required when PROVIDER=openrouter. "
        "Set it in your .env file."
    )
