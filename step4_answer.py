"""STEP 4 — Put it all together: a working RAG app.

RAG = Retriever + Augmented prompt + Generator:

    R — retrieve the most relevant chunks     (step 3)
    A — augment the prompt with them          (this step)
    G — ask the LLM to answer using ONLY that context

The LLM never sees the whole handbook — just the top 5 chunks.
That's the whole point of RAG: smaller context, cheaper, grounded answers.

The LLM is any OpenAI-compatible API, configured in .env
(LLM_BASE_URL / LLM_MODEL / LLM_API_KEY).

Run:
  python step4_answer.py "Can I take a day off to attend my sister's wedding?"
  python step4_answer.py --dry-run "How much vacation do I get after one year?"
  (add --dry-run to see the exact prompt without calling the LLM — no key needed)
"""

import sys

from helpers import LLM_MAX_TOKENS, LLM_MODEL, get_llm, require_api_key
from step3_search import TOP_K, load_store, vector_search

DEFAULT_QUESTION = "Can I take a day off to attend my sister's wedding?"

SYSTEM_PROMPT = (
    "You are a helpful HR assistant at a small nonprofit. "
    "Answer the employee's question using ONLY the handbook excerpts provided. "
    "If the excerpts contain the answer, answer directly and mention which "
    "page you used. If they do not contain the answer, say so clearly "
    "and do not guess."
)


def build_prompt(question, hits):
    """Augmentation: wrap the retrieved chunks around the question."""
    excerpts = "\n\n".join(
        f"[Excerpt {i + 1}] — handbook, page {h[0]['metadata']['page']} "
        f"(similarity {h[1]:.2f})\n{h[0]['text']}"
        for i, h in enumerate(hits)
    )
    return f"Handbook excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def main():
    args = list(sys.argv[1:])
    dry_run = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]
    question = " ".join(args) or DEFAULT_QUESTION

    hits = vector_search(question, load_store(), top_k=TOP_K)

    print(f"Question: {question}\n")
    print("Retrieved chunks (R):")
    for entry, score in hits:
        print(f"  {score:.2f}  [p.{entry['metadata']['page']}]")
    print()

    prompt = build_prompt(question, hits)

    if dry_run:
        print(f"DRY RUN — the full prompt that would be sent to {LLM_MODEL}:\n")
        print("=" * 70)
        print("[system]\n" + SYSTEM_PROMPT + "\n" + "-" * 70)
        print("[user]\n" + prompt)
        print("=" * 70)
        return

    require_api_key()
    response = get_llm().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,  # grounded answers, not creativity
        max_tokens=LLM_MAX_TOKENS,
    )

    content = response.choices[0].message.content
    if not content:
        print(
            "The LLM returned no text — this can happen with 'reasoning' models "
            "that spend their token budget thinking. Try a bigger budget, e.g.:\n"
            f"    export LLM_MAX_TOKENS={LLM_MAX_TOKENS * 2}\n"
        )
        return

    print(f"Answer (from {LLM_MODEL}):\n")
    print(content)
    print()

    seen, sources = set(), []
    for entry, _ in hits:
        label = f"page {entry['metadata']['page']}"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    print("Sources: " + ", ".join(sources))


if __name__ == "__main__":
    main()
