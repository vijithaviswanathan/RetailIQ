
import os

# Database configuration
def get_database_url():
    # First: environment variable (local Docker / Render)
    database_url = os.getenv("DATABASE_URL", "").strip()

    # Second: Streamlit Cloud secrets
    if not database_url:
        try:
            import streamlit as st
            database_url = str(
                st.secrets.get("DATABASE_URL", "")
            ).strip()
        except Exception:
            database_url = ""

    # Local development fallback
    if not database_url:
        database_url = "postgresql+psycopg2://localhost/retailiq"

    # Normalize PostgreSQL URLs for SQLAlchemy
    if database_url.startswith("postgres://"):
        database_url = database_url.replace(
            "postgres://",
            "postgresql+psycopg2://",
            1,
        )
    elif database_url.startswith("postgresql://"):
        database_url = database_url.replace(
            "postgresql://",
            "postgresql+psycopg2://",
            1,
        )

    return database_url


DATABASE_URL = get_database_url()

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
