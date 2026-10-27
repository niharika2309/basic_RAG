"""Shared helpers used by all the step files.

Each step file has ONE job (see its docstring). This file holds the
plumbing they share:

  - config (model names, API key) read from a .env file
  - get_embedder()  a small LOCAL model that turns text into embeddings
  - get_reranker()  a small LOCAL model that scores how well a chunk
                    answers a question (used by step 3's second pass)
  - get_llm()       a client for any OpenAI-compatible LLM API
"""

import math
import os

from dotenv import load_dotenv

load_dotenv()  # reads KEY=value lines from .env, if one exists

# Everything is configurable, so you can swap models without touching code.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L12-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
LLM_EVAL_MODEL = os.getenv("LLM_EVAL_MODEL", "gpt-4o-mini")
LLM_EVAL_BASE_URL = os.getenv("LLM_EVAL_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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


_reranker = None
_reranker_error = None


def get_reranker():
    """Load the small cross-encoder that re-checks search results (once).

    The embedding model compares two *fingerprints*, so the question and the
    chunk are encoded separately, never side by side. A cross-encoder reads
    them TOGETHER — "question + chunk" go in, one relevance score comes out.
    That is much more careful, and much slower: it has to re-read every
    candidate, which is why step 3 only shows it the ~20 the search found
    first, instead of the whole handbook.

    This one is the standard small, fast choice (~100 MB, from Hugging Face,
    then offline and free). If it can't load — no internet, a typo in
    RERANKER_MODEL — the reason is remembered, so one bad model name doesn't
    mean waiting out the same download for every question.
    """
    global _reranker, _reranker_error
    if _reranker is not None:
        return _reranker
    if _reranker_error is not None:
        raise _reranker_error

    from sentence_transformers import CrossEncoder

    print(f"Loading reranker model '{RERANKER_MODEL}' "
          f"(first run downloads it, ~100 MB)...")
    try:
        _reranker = CrossEncoder(RERANKER_MODEL)
    except Exception as exc:
        _reranker_error = exc
        raise
    print("Reranker ready.\n")
    return _reranker


def rerank(question, texts):
    """Score each text against the question: a 0..1 relevance per text.

    1.0 = this chunk answers the question, 0.0 = it doesn't. Unlike
    similarity, which only asks "do these mean something alike?", this one
    has actually read both together.
    """
    if not texts:
        return []
    logits = get_reranker().predict([(question, text) for text in texts])
    # The model's raw output is an unbounded score, so squash each one into
    # 0..1. (Clamping first keeps extreme values from overflowing math.exp.)
    return [1 / (1 + math.exp(-min(30.0, max(-30.0, float(x))))) for x in logits]


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

def get_llm_eval():
    """Create an OpenAI-compatible chat client (only once).

    Works with any API that speaks the OpenAI protocol — OpenAI itself,
    Groq, OpenRouter, a local Ollama server, and so on.
    """
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI

        _llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=LLM_EVAL_BASE_URL)
    return _llm_client


def require_api_key():
    """Fail with a friendly message if no LLM key is configured."""
    if not LLM_API_KEY or not OPENAI_API_KEY:
        raise SystemExit(
            "No LLM API key found.\n"
            "  1) Copy .env.example to .env and add your key, or\n"
            "  2) Run with --dry-run to see the prompt without calling the LLM."
        )
