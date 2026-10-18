"""Shared helpers used by all the step files.

Each step file has ONE job (see its docstring). This file holds the
plumbing they all share:

  1. A tiny .env loader, so your API key lives in a file instead of code.
  2. Two "factories":
       - get_embedder()  a LOCAL model (MiniLM) that turns text into
                         embeddings. Free, private, runs on your CPU.
       - get_llm()       a client for any OpenAI-compatible LLM API.
"""

import os

# ---------------------------------------------------------------------------
# 1) A tiny .env loader
# ---------------------------------------------------------------------------

def load_dotenv(path=".env"):
    """Read KEY=value lines from a .env file, if one exists.

    Real environment variables always win over values in the file.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


load_dotenv()

# Everything is configurable through these environment variables, so you can
# swap models or endpoints without touching any code.
#
# "all-MiniLM-L12-v2" is a small version of the well-known MiniLM embedding
# model. ("all-MiniLM-L6-v2" is even smaller/faster, if you prefer that.)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L12-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
# Some models "reason" before answering and spend the token budget on
# thinking. 2000 is comfortable for those and harmless for the rest.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

# ---------------------------------------------------------------------------
# 2) The embedder — a small model that runs on YOUR computer
# ---------------------------------------------------------------------------

_embedder = None


def get_embedder():
    """Load the MiniLM embedding model (only once).

    MiniLM is a small, well-known embedding model. On the first run it
    downloads itself (~130 MB) from Hugging Face; after that it works
    completely offline and costs nothing.

    It turns any piece of text into a vector of 384 numbers.
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

    The vectors are normalized (each has length 1). That's what makes
    cosine similarity as easy as a dot product — you'll see why in step 3.
    """
    import numpy as np

    model = get_embedder()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,  # length-1 vectors, so cosine == dot product
        batch_size=32,
        show_progress_bar=len(texts) > 1,  # no bar for a single question
    )
    return np.asarray(vectors)


# ---------------------------------------------------------------------------
# 3) The LLM — any OpenAI-compatible API
# ---------------------------------------------------------------------------

_llm_client = None


def get_llm():
    """Create an OpenAI-compatible chat client.

    This works with any API that speaks the OpenAI protocol — OpenAI
    itself, Groq, OpenRouter, Together, a local Ollama server, and so on.
    Point LLM_BASE_URL / LLM_MODEL in your .env at whatever you use.
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
