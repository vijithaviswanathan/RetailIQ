
import os

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://localhost/retailiq",
)

# AI provider: ollama, openai, or gemini
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").strip().lower()

if AI_PROVIDER not in {"ollama", "openai", "gemini"}:
    raise ValueError(
        "AI_PROVIDER must be 'ollama', 'openai', or 'gemini'."
    )

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

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta",
).rstrip("/")

GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001",
)

GEMINI_CHAT_MODEL = os.getenv(
    "GEMINI_CHAT_MODEL",
    "gemini-3.8-flash",
)
