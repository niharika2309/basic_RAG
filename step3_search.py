"""STEP 3 — Find the chunks most relevant to a question.

This step has two passes, because "find it fast" and "judge it well" are
two different jobs:

  1. SEARCH — the question gets an embedding with the same model that
     embedded the chunks, then is compared against every stored embedding
     using cosine similarity. Cheap, so it can scan the whole handbook.

     Because every vector has length 1 (normalized), cosine similarity is
     just a dot product — a sum of the products of matching numbers. We
     compute it in plain Python, so you can see exactly what "similar"
     means.

  2. RERANK — a second, more careful model (a *cross-encoder*) re-reads
     the question together with each candidate and scores how well that
     chunk actually answers it. Better judgment, but far too slow to run
     over every chunk — so it only sees the 20 survivors from pass 1.

     vector_store.json + question  -->  20 candidates  -->  top 5   (this step)

The two models answer different questions, and they disagree in useful
ways. Similarity asks "do these mean something alike?", which is why a
page of table-of-contents dots can score high. Relevance asks "does this
chunk answer THIS question?", and only the reranker can answer that — it
alone reads the question and the chunk at the same time.

Run:
  python step3_search.py
  python step3_search.py "How much vacation do I get after one year?"
  python step3_search.py --no-rerank "..."   # search only, no second pass
"""

import json
import sys

from helpers import RERANK_CANDIDATES, embed, rerank

STORE_PATH = "vector_store.json"
DEFAULT_QUESTION = "Can I take a day off to attend my sister's wedding?"
TOP_K = 5                 # chunks that finally go to the LLM
CANDIDATE_K = RERANK_CANDIDATES  # chunks the search hands to the reranker


def load_store(path=STORE_PATH):
    with open(path) as f:
        return json.load(f)


def cosine(a, b):
    """Similarity of two length-1 vectors: a dot product.

    1.0 = same direction (very similar meaning), 0 = unrelated, -1 = opposite.
    """
    return float(sum(x * y for x, y in zip(a, b)))


def vector_search(question, store, top_k=TOP_K):
    """Return the top_k stored chunks closest to the question, best first,
    as (entry, similarity_score) pairs."""
    question_vec = embed([question])[0]
    scored = [(entry, cosine(question_vec, entry["embedding"])) for entry in store]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [(entry, round(score, 3)) for entry, score in scored[:top_k]]


def rerank_hits(question, hits, top_k=TOP_K):
    """Re-judge search candidates with the cross-encoder, best first.

    `hits` are the (entry, similarity) pairs from vector_search. Returns
    (entry, relevance, similarity) triples, so a caller can show both
    scores and see what the second pass changed.
    """
    scores = rerank(question, [entry["text"] for entry, _ in hits])
    judged = [
        (entry, round(score, 3), sim)
        for (entry, sim), score in zip(hits, scores)
    ]
    judged.sort(key=lambda triple: triple[1], reverse=True)
    return judged[:top_k]


def search_and_compare(question, store, top_k=TOP_K, candidate_k=CANDIDATE_K,
                       use_reranker=True):
    """Step 3 in full — plus the evidence of what the reranker changed.

    Returns four things:

        candidates — the candidate_k closest chunks, in similarity order
        before     — the first top_k of those: what the search would have
                     sent to the LLM on its own
        after      — the top_k the reranker chose
        reranked   — False when the reranker didn't get the final say, either
                     because it's switched off or because it couldn't load
                     (no internet for the one-time download, say); then
                     `after` is just `before`, and the reason gets printed
                     instead of an answer ruined by a crash.
    """
    candidates = vector_search(question, store, top_k=max(candidate_k, top_k))
    before = candidates[:top_k]
    if not use_reranker:
        return candidates, before, before, False
    try:
        after = [(entry, relevance) for entry, relevance, _ in
                 rerank_hits(question, candidates, top_k)]
    except Exception as exc:
        print(f"(reranker unavailable — falling back to similarity order: {exc})\n")
        return candidates, before, before, False
    return candidates, before, after, True


def search(question, store, top_k=TOP_K, candidate_k=CANDIDATE_K, use_reranker=True):
    """Step 3 as one call: search broadly, then rerank down to top_k.

    A drop-in replacement for vector_search — same (entry, score) pairs,
    just in better order (the score is the reranker's relevance, or the
    similarity when the reranker is off). Callers that want to *show* the
    reranker's effect want search_and_compare() instead.
    """
    return search_and_compare(question, store, top_k, candidate_k, use_reranker)[2]


def main():
    args = list(sys.argv[1:])
    use_reranker = "--no-rerank" not in args
    args = [a for a in args if a != "--no-rerank"]
    question = " ".join(args) or DEFAULT_QUESTION
    store = load_store()

    print(f"Question: {question}\n")

    if not use_reranker:
        print(f"Top {TOP_K} chunks by similarity (reranker off):\n")
        for entry, score in vector_search(question, store, top_k=TOP_K):
            print(f"  {score:.2f}  [p.{entry['metadata']['page']}]")
            print(f"         {entry['text'][:150]}...\n")
        return

    candidates = vector_search(question, store, top_k=max(CANDIDATE_K, TOP_K))
    print(f"{len(candidates)} candidates found by similarity, then the reranker")
    print(f"re-reads them all and keeps the best {TOP_K} — both scores are shown,\n"
          f"so you can see what the second pass changed:\n")
    print(f"  {'page':>4}  {'similar':>8}  {'answers':>8}")
    for entry, relevance, similarity in rerank_hits(question, candidates):
        page = entry["metadata"]["page"]
        print(f"  {page:>4}  {similarity:>8.2f}  {relevance:>8.3f}")
        print(f"         {entry['text'][:150]}...\n")


if __name__ == "__main__":
    main()
