"""
config.py
---------
Central configuration for the Research RAG Application.
"""

import os
from pathlib import Path

# ── Project Paths ─────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

for _dir in (DATA_DIR, VECTORSTORE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# ── FAISS ─────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH: str = str(VECTORSTORE_DIR / "faiss_index")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE:    int = 1000
CHUNK_OVERLAP: int = 200

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS: int = 4

# ── API Providers ─────────────────────────────────────────────────────────────
API_PROVIDERS: dict = {
    "Google Gemini": {
        "models": [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
        ],
        "default": "gemini-2.5-flash-preview-05-20",
        "key_label": "Google Gemini API Key",
        "key_prefix": "AI...",
        "get_key_url": "https://aistudio.google.com/app/apikey",
    },
    "OpenAI": {
        "models": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ],
        "default": "gpt-4o-mini",
        "key_label": "OpenAI API Key",
        "key_prefix": "sk-...",
        "get_key_url": "https://platform.openai.com/api-keys",
    },
    "Anthropic Claude": {
        "models": [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229",
        ],
        "default": "claude-3-haiku-20240307",
        "key_label": "Anthropic API Key",
        "key_prefix": "sk-ant-...",
        "get_key_url": "https://console.anthropic.com/settings/keys",
    },
}

# ── Upload Validation ─────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB:    int       = 50
ALLOWED_EXTENSIONS: set[str]  = {".pdf"}

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE:    str = "📄 Research RAG Assistant"
APP_SUBTITLE: str = "Ask questions about your research papers — powered by AI"
