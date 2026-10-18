"""STEP 3 — Find the chunks most similar to a question.

The question is turned into an embedding with the same model that
embedded the chunks, then compared against every stored embedding
using cosine similarity.

Because every vector has length 1 (normalized), cosine similarity is
just a dot product — a sum of the products of matching numbers. We
compute it in plain Python, so you can see exactly what "similar" means.

    vector_store.json + question  -->  top 5 chunks        (this step)

Run:
  python step3_search.py
  python step3_search.py "How much vacation do I get after one year?"
"""

import json
import sys

from helpers import embed

STORE_PATH = "vector_store.json"
DEFAULT_QUESTION = "Can I take a day off to attend my sister's wedding?"
TOP_K = 5  # how many chunks to retrieve


def load_store(path=STORE_PATH):
    with open(path) as f:
        return json.load(f)


def cosine(a, b):
    """Similarity of two length-1 vectors: a dot product.

    1.0 = same direction (very similar meaning), 0 = unrelated, -1 = opposite.
    """
    return sum(x * y for x, y in zip(a, b))


def vector_search(question, store, top_k=TOP_K):
    """Return the top_k stored chunks closest to the question, best first,
    as (entry, similarity_score) pairs."""
    question_vec = embed([question])[0]
    scored = [(entry, cosine(question_vec, entry["embedding"])) for entry in store]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [(entry, round(score, 3)) for entry, score in scored[:top_k]]


def main():
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    store = load_store()

    print(f"Question: {question}\n")
    print(f"Top {TOP_K} chunks by similarity:\n")
    for entry, score in vector_search(question, store):
        print(f"  {score:.2f}  [p.{entry['metadata']['page']}]")
        print(f"         {entry['text'][:150]}...\n")


if __name__ == "__main__":
    main()
