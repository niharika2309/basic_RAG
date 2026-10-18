"""Shared helpers used by all the step files.

Each step file has ONE job (see its docstring). This file holds the
plumbing they share:

  - config (model names, API key) read from a .env file
  - get_embedder()  a small LOCAL model that turns text into embeddings
  - get_llm()       a client for any OpenAI-compatible LLM API
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads KEY=value lines from .env, if one exists

# Everything is configurable, so you can swap models without touching code.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L12-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

_embedder = None


def get_embedder():
    """Load the MiniLM embedding model (only once).

    MiniLM is a small, well-known embedding model. On the first run it
    downloads itself (~130 MB) from Hugging Face; after that it runs
    completely offline and costs nothing.
    """
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model '{EMBEDDING_MODEL}' "
              f"(first run downloads it, ~130 MB)...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model ready.\n")
    return _embedder


def embed(texts):
    """Turn a list of texts into an array of embeddings.

    The vectors are normalized (each has length 1), which makes
    "similarity" as easy as a dot product — you'll see why in step 3.
    """
    import numpy as np

    vectors = get_embedder().encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=len(texts) > 1,  # no progress bar for one question
    )
    return np.asarray(vectors)


_llm_client = None


def get_llm():
    """Create an OpenAI-compatible chat client (only once).

    Works with any API that speaks the OpenAI protocol — OpenAI itself,
    Groq, OpenRouter, a local Ollama server, and so on.
    """
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI

        _llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _llm_client


def require_api_key():
    """Fail with a friendly message if no LLM key is configured."""
    if not LLM_API_KEY:
        raise SystemExit(
            "No LLM API key found.\n"
            "  1) Copy .env.example to .env and add your key, or\n"
            "  2) Run with --dry-run to see the prompt without calling the LLM."
        )
