"""A ChatGPT-like web UI for the handbook RAG app.

The same four steps as the CLI scripts, but in a browser:

    your question ──► [3] search + rerank ──► [4] LLM answer ──► chat bubble
                        (steps 1 + 2 already ran ahead of time)

Run it from the project folder (with the venv activated):

    streamlit run app.py

...then open http://localhost:8501
"""

import streamlit as st

from helpers import (
    EMBEDDING_MODEL,
    LLM_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    RERANKER_MODEL,
    get_llm,
)
from step3_search import CANDIDATE_K, TOP_K, load_store, search_and_compare
from step4_answer import SYSTEM_PROMPT, build_prompt

st.set_page_config(page_title="Handbook Assistant", page_icon="📖", layout="centered")

GREETING = (
    "Hi! 👋 I'm your **handbook assistant**.\n\n"
    "Ask me anything about the employee handbook — vacation, leave, pay, "
    "benefits, visitors…\n\n"
    "I answer **only from the handbook**: I look up the most relevant "
    "passages, re-check them against your question, and build my answer "
    "from them. If the handbook doesn't cover your question, I'll say so "
    "— instead of guessing.\n\n"
    "The **How this works** panel on the left shows exactly what happens "
    "behind each answer."
)

SUGGESTIONS = [
    "How much vacation do I get after one year?",
    "Do I get paid if I am called for jury duty?",
    "Can I smoke in the office?",
    "What is the policy on remote work?",
]


@st.cache_data(show_spinner=False)
def get_store():
    """Our "vector database" — a plain JSON file."""
    return load_store()


def run_turn(question, store, use_reranker=True):
    """Run steps 3 + 4 for one question: retrieve, then generate.

    Returns (answer_text, hits, comparison). `comparison` is None unless the
    reranker had the final say, in which case it holds the passages in both
    orders so the page can show what the second pass changed. Every failure
    becomes a friendly message instead of a traceback.
    """
    # Step 3 — retrieval (the first call also loads the local models).
    try:
        with st.spinner(
            f"Searching, then re-checking the {CANDIDATE_K} closest passages…"
            if use_reranker
            else "Searching the handbook for relevant passages…"
        ):
            candidates, before, hits, reranked = search_and_compare(
                question, store, top_k=TOP_K, use_reranker=use_reranker
            )
    except Exception as exc:
        st.error(
            f"Search failed: {exc}\n\nThe first run downloads the local models "
            "(embedding ~130 MB, reranker ~100 MB). Check your internet "
            "connection and reload."
        )
        return None, [], None

    comparison = {"before": before, "after": hits, "candidates": candidates} \
        if reranked else None

    # Step 4 — the LLM.
    if not LLM_API_KEY:
        return (
            "I found the passages below, but I can't draft an answer: **no LLM "
            "API key is configured.** Add your key to `.env` (see the README) "
            "and I'll switch to full answers.",
            hits,
            comparison,
        )

    try:
        with st.spinner(f"Drafting an answer with {LLM_MODEL}…"):
            response = get_llm().chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(question, hits)},
                ],
                temperature=0,
                max_tokens=LLM_MAX_TOKENS,
            )
        content = response.choices[0].message.content
    except Exception as exc:
        st.error(
            f"I couldn't reach the LLM: {exc}\n\nCheck the `LLM_*` values in "
            "your `.env`. In the meantime, here's what I found in the handbook 👇"
        )
        return None, hits, comparison

    if not content:
        content = (
            f"The model came back without any text — this can happen with "
            f"'reasoning' models that spend their token budget thinking. Try "
            f"raising `LLM_MAX_TOKENS` in `.env` (it's currently "
            f"{LLM_MAX_TOKENS})."
        )
    return content, hits, comparison


def _keyed(hits):
    """(identity, score) pairs, for spotting the same passage in two orderings.

    Two chunks can hold exactly the same text (this handbook repeats itself,
    and chunking overlaps), so repeats of a page's text are numbered — the
    second copy counts as a different passage here.
    """
    seen, keyed = {}, []
    for entry, score in hits:
        base = (entry["metadata"]["page"], entry["text"])
        seen[base] = seen.get(base, 0) + 1
        keyed.append(((base[0], base[1], seen[base]), score))
    return keyed


def render_sources(hits, reranked=True):
    """The little "grounding" footer under an answer: its sources, plus an
    expander with the exact passages behind them."""
    if not hits:
        return
    pages = []
    for entry, _ in hits:
        p = entry["metadata"]["page"]
        if p not in pages:
            pages.append(p)
    st.markdown("**Sources:** handbook pages " + ", ".join(map(str, pages)))
    label = "relevance" if reranked else "similarity"
    with st.expander(f"🔍 The {len(hits)} handbook passages I used"):
        for entry, score in hits:
            excerpt = entry["text"]
            if len(excerpt) > 260:
                excerpt = excerpt[:260].rsplit(" ", 1)[0] + " …"
            st.caption(
                f"{label} {score:.3f} — p.{entry['metadata']['page']}: {excerpt}"
            )


def render_rerank_diff(comparison):
    """The before/after panel under an answer: the order the search gave,
    next to the order the reranker gave, with every move marked.

    This is the reranker's whole argument in one view — similarity alone
    cannot tell a real policy paragraph from a page of table-of-contents
    dots, and here you can watch it get corrected.
    """
    if not comparison:
        return
    candidates, before, after = (
        comparison["candidates"], comparison["before"], comparison["after"]
    )
    # Where each chunk started (its rank among all the candidates) and where
    # the reranker ended up putting it.
    started = {key: rank for rank, (key, _) in enumerate(_keyed(candidates), 1)}
    ended = {key: rank for rank, (key, _) in enumerate(_keyed(after), 1)}

    def happened_to(key, rank):
        """Before-list view: what became of this passage."""
        now = ended.get(key)
        if now is None:
            return "✕ dropped out"
        return "✓ kept" if now == rank else f"→ moved to #{now}"

    def came_from(key, rank):
        """After-list view: where this passage came from."""
        was = started[key]
        if was == rank:
            return "✓ stayed"
        if was > TOP_K:
            return f"▲ new — was #{was}"
        return f"▲ from #{was}" if was > rank else f"▼ from #{was}"

    st.caption(
        "🔁 The re-check moved "
        f"{sum(1 for key, rank in ended.items() if started[key] != rank)} of "
        f"{len(after)} passages"
        + "".join(
            f" · dropped page {key[0]}"
            for key, _ in _keyed(before)
            if key not in ended
        )
    )
    with st.expander("🔁 The same passages, before and after the re-check"):
        columns = st.columns(2)
        for column, title, hits, label, describe in (
            (columns[0], "Before — the search's own top 5", before, "similar",
             happened_to),
            (columns[1], "After — the reranker's top 5", after, "relevance",
             came_from),
        ):
            with column:
                st.markdown(f"**{title}**")
                for rank, (key, score) in enumerate(_keyed(hits), 1):
                    st.caption(
                        f"{rank}. p.{key[0]} · {label} {score:.3f} — "
                        f"{describe(key, rank)}"
                    )


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": GREETING, "hits": []}
    ]

try:
    store = get_store()
except FileNotFoundError:
    st.error(
        "The vector store (`vector_store.json`) doesn't exist yet. Run the "
        "one-time preparation first (see the README), then reload:\n\n"
        "```bash\npython step1_extract_and_chunk.py\npython step2_embed.py\n```"
    )
    st.stop()

with st.sidebar:
    st.header("How this works")
    st.markdown(
        "1️⃣ **Read & cut** — *ahead of time.* The handbook is split into "
        "small pieces (*chunks*), each tagged with the page it came from.\n\n"
        "2️⃣ **Embed** — *ahead of time.* A small model running on your "
        "computer turns each chunk into 384 numbers: a numeric fingerprint "
        "of its meaning.\n\n"
        "3️⃣ **Search** — *when you ask.* Your question gets the same kind "
        f"of fingerprint, and I shortlist the {CANDIDATE_K} chunks whose "
        "fingerprints are closest. Close numbers ≈ close meaning — even "
        "when **no words overlap**.\n\n"
        "🔁 **Re-check** — *when you ask.* Fingerprints are only ever "
        "compared to other fingerprints, so a page of table-of-contents "
        "dots can sneak in. A second, more careful model reads your "
        "question *together with* each shortlisted chunk and scores how "
        "well it actually answers it; only the best "
        f"{TOP_K} survive. Under each answer you can see the two orders "
        "side by side.\n\n"
        "4️⃣ **Answer** — *when you ask.* Those chunks + your question go to "
        "the LLM, with one rule: *answer only from what you're shown.*"
    )
    use_reranker = st.toggle(
        "🔁 Re-check results with the reranker",
        value=True,
        help=(
            f"On: shortlist {CANDIDATE_K} chunks by similarity, then keep the "
            f"top {TOP_K} judged by {RERANKER_MODEL}, and show both orders "
            f"under the answer. Off: straight to the top {TOP_K} by "
            "similarity, with nothing to compare."
        ),
    )
    st.info(
        "**No vector database here — on purpose.** Those 384-number vectors "
        "live in a plain JSON file (`vector_store.json`), and the search is "
        "a few lines of plain Python. This is a *basic* RAG demo, built to "
        "show how the pieces work. A real vector database (Chroma, Milvus, "
        "…) would be a drop-in upgrade to steps 2 and 3."
    )
    st.caption(
        f"{len(store)} chunks · `{EMBEDDING_MODEL}` · "
        f"reranker: `{RERANKER_MODEL if use_reranker else 'off'}` · "
        f"LLM: `{LLM_MODEL}`"
    )
    if st.button("🧹 Reset chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": GREETING, "hits": []}
        ]
        st.rerun()

st.title("📖 Handbook Assistant")
st.caption(
    "A basic RAG demo — answers are grounded in the handbook only, and the "
    "“vector database” is a plain JSON file, not a real one."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("hits") or [], message.get("reranked", True))
        render_rerank_diff(message.get("comparison"))

question = None
if len(st.session_state.messages) == 1:
    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        if cols[i % 2].button(suggestion, key=f"suggestion-{i}"):
            question = suggestion

question = st.chat_input("Ask me anything about the handbook…") or question

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question, "hits": []}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        content, hits, comparison = run_turn(question, store, use_reranker)
        reranked = comparison is not None  # False if it was off or unavailable
        if content:
            st.markdown(content)
        render_sources(hits, reranked)
        render_rerank_diff(comparison)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content or "",
                "hits": hits,
                "reranked": reranked,
                "comparison": comparison,
            }
        )
