"""Central configuration: env vars, paths, model constants."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (override=True so .env wins over shell env)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

# --- API keys (fail fast) ---------------------------------------------------

GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")

# Ensure GOOGLE_API_KEY matches GEMINI_API_KEY so the google-genai SDK
# doesn't pick up a stale key from the shell environment.
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY


def require_keys() -> None:
    """Validate that both API keys are set. Call at CLI entry points."""
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if missing:
        raise ValueError(
            f"Missing required env vars: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your keys."
        )


# --- Paths -------------------------------------------------------------------

DATA_DIR = _PROJECT_ROOT / "data"
PAGES_DIR = DATA_DIR / "pages"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

PAGES_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Model constants ---------------------------------------------------------

EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIM = 3072
GENERATION_MODEL = "gemini-3-flash-preview"

# --- Pinecone ----------------------------------------------------------------

INDEX_NAME = "multimodal-rag"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# --- Batching ----------------------------------------------------------------

EMBEDDING_TEXT_BATCH = 10
EMBEDDING_IMAGE_BATCH = 6
EMBEDDING_SLEEP_S = 1.0
PINECONE_UPSERT_BATCH = 100

# --- Dataset -----------------------------------------------------------------

HF_DATASET = "vidore/vidore_v3_finance_en"
