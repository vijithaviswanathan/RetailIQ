import os

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://localhost/retailiq",
)

# AI provider: ollama locally, openai in the cloud
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").strip().lower()

if AI_PROVIDER not in {"ollama", "openai"}:
    raise ValueError("AI_PROVIDER must be either 'ollama' or 'openai'.")

# Ollama configuration
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

EMBEDDING_URL = f"{OLLAMA_BASE_URL}/api/embed"
CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://api.openai.com/v1",
).rstrip("/")

OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

OPENAI_CHAT_MODEL = os.getenv(
    "OPENAI_CHAT_MODEL",
    "gpt-4o-mini",
)