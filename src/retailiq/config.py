import os

# PostgreSQL connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://localhost/retailiq",
)

# Ollama connection
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

EMBEDDING_URL = f"{OLLAMA_BASE_URL}/api/embed"
CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

# AI models
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text",
)

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "llama3.2:3b",
)
