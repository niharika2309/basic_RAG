"""STEP 3 — Vector search: find the chunks most similar to a question.

The question is turned into an embedding with the same MiniLM model,
then compared against every stored embedding using cosine similarity.

Because every vector has length 1 (normalized), cosine similarity is
just a dot product — a sum of the products of matching numbers. We
compute it in plain Python, so you can see exactly what "similar" means.

    vector_store.json + question  -->  top-k chunks        (this step)

Run:
  python step3_search.py
  python step3_search.py "How much vacation do I get after one year?"
"""

import json
import re
import sys

from helpers import embed

STORE_PATH = "vector_store.json"
DEFAULT_QUESTION = "Can I take a day off to attend my sister's wedding?"
TOP_K = 5  # how many chunks to report / retrieve


def load_store(path=STORE_PATH):
    with open(path) as f:
        return json.load(f)


def keyword_search(question, store):
    """The old-school baseline: keep chunks that contain the exact words
    of the question (ignoring very short words).

    Notice what it can't do: it only matches words, never meaning.
    """
    words = {w for w in re.findall(r"[a-z]{4,}", question.lower())}
    return [
        entry
        for entry in store
        if any(w in entry["text"].lower() for w in words)
    ]


def cosine(question_vec, entry_vec):
    """Cosine similarity for two vectors of length 1 — a dot product.

    1.0 means identical direction (very similar meaning), 0 means
    unrelated, -1 means opposite.
    """
    return sum(a * b for a, b in zip(question_vec, entry_vec))


def vector_search(question, store, top_k=TOP_K):
    """Return the `top_k` stored chunks most similar to the question,
    best first, as a list of (entry, similarity_score) pairs.
    """
    question_vec = embed([question])[0]

    scored = [
        (entry, cosine(question_vec, entry["embedding"])) for entry in store
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return [(entry, round(score, 3)) for entry, score in scored[:top_k]]


def main():
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    store = load_store()

    print(f"Question: {question}\n")

    matched = sorted(set(re.findall(r"[a-z]{4,}", question.lower())))
    kw_hits = keyword_search(question, store)
    print(f"Keyword search — chunks containing the exact words {matched}:")
    print(f"  {len(kw_hits)} of {len(store)} chunks matched. First few:\n")
    for entry in kw_hits[:3]:
        m = entry["metadata"]
        print(f"   - [{m['section']}  p.{m['page']}]  {entry['text'][:110]}...")

    print(f"\nVector search — top {TOP_K} chunks by cosine similarity:\n")
    for entry, score in vector_search(question, store):
        m = entry["metadata"]
        print(f"  {score:.2f}   [{m['section']}  p.{m['page']}]")
        print(f"          {entry['text'][:160]}...\n")


if __name__ == "__main__":
    main()
