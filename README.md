# A Basic RAG App — Employee Handbook Q&A

A tiny, complete **RAG** (Retrieval-Augmented Generation) application.
You ask a real, 40-page employee handbook questions in plain English, and
an LLM answers **only from what it finds in the handbook** — with sources
cited, and no hallucinated policies.

This project is built for the "one concept at a time" series. Every file
does one job, and every step maps to a concept from the RAG posts:

| Step | File | Concept |
|------|------|---------|
| 1 | `step1_extract_and_chunk.py` | chunking + metadata |
| 2 | `step2_embed.py` | embeddings + storing them in a vector "database" |
| 3 | `step3_search.py` | vector search (similarity search) |
| 4 | `step4_answer.py` | the LLM: a grounded answer |

The document we use is a real (sample) employee handbook for nonprofits:
[data/employee-handbook.pdf](data/employee-handbook.pdf).

## How it works

```
            AHEAD OF TIME (run once)                          AT QUESTION TIME
   ┌──────────────────────────────────────┐   ┌─────────────────────────────────────────┐
   │                                      │   │                                         │
   │  PDF ──► [1] chunks ──► [2] embeddings  │   │  question ──► embedding ──► [3] vector  │
   │                                      │   │                     search ──► top 5    │
   └──────────────────────────────────────┘   │                                   chunks │
                                               │                              │           │
                                               │  [4] prompt = context + question ──► LLM  │
                                               │                              │           │
                                               │                          final answer   │
                                               └─────────────────────────────────────────┘
```

Two important ideas:

1. **The preparation happens once.** Steps 1 and 2 run ahead of time,
   before any question is asked. (In a production system you'd re-run
   them whenever the document changes.)
2. **The LLM never sees the whole handbook.** It only sees the 5 most
   relevant chunks. Smaller context → cheaper, faster, and *grounded*
   answers.

## What you need

- **Python 3.9+** (3.10+ recommended)
- **An API key for an LLM** — only for step 4. Any OpenAI-compatible
  API works: OpenAI, Groq, OpenRouter, a local Ollama server, and so on.
- **Nothing else.** The embedding model (MiniLM) runs on your own
  computer: no key, no cost, private. It downloads itself once (~135 MB)
  on the first run.

## Setup

```bash
# 1. Create a virtual environment and activate it
python3 -m venv .venv
source .venv/bin/activate        # (Windows: .venv\Scripts\activate)

# 2. Install the (four) dependencies
pip install -r requirements.txt
```

> Heads up: the first install is large (~1–2 GB) because
> `sentence-transformers` pulls in PyTorch. That's a one-time cost;
> afterwards everything runs locally.

```bash
# 3. Configure your LLM
cp .env.example .env
```

Open `.env` and put your key in `LLM_API_KEY`. The defaults call
`gpt-4o-mini` at `api.openai.com`. To use a different model, change
`LLM_MODEL` / `LLM_BASE_URL` — see [Swapping models](#swapping-models).

No LLM key? No problem — every step still works, and step 4 has a
`--dry-run` mode that shows you the exact prompt the LLM would receive.

## Run it (in order)

### Step 1 — Chunk the document

```bash
python step1_extract_and_chunk.py
```

```
Reading data/employee-handbook.pdf ...

Done! 47 sections  ->  226 chunks  ->  chunks.json

A few sample chunks (the metadata is what lets us cite sources later):

  [V. BENEFITS AND LEAVES OF ABSENCE > B. Vacation Benefits  p.13]
  B. Vacation Benefits [ORGANIZATION] has established the following
  vacation plan to provide eligible employees time off with pay ...
```

The handbook is cut into 226 chunks of ~600 characters each. Every chunk
carries **metadata**: which section it came from and which page. That's
what lets us cite sources later.

### Step 2 — Embed and store

```bash
python step2_embed.py
```

```
Reading chunks.json ...
Loading embedding model 'all-MiniLM-L12-v2' (first run downloads it, ~130 MB)...
Embedding model ready.

Batches: 100%|██████████| 8/8 [00:04<00:00, 2.03it/s]

Embedded 226 chunks in 4.6 seconds.

Each embedding is a vector of 384 numbers.
Example — the first 8 numbers of chunk 0:
  [ 0.0234  0.0687 -0.0108 -0.0016  0.0118  0.034  -0.0459 -0.0201]

Saved vector_store.json — our simple 'vector database'.
```

This is the "ahead of time" half of RAG, and it's the part that matches
Day 4 of the series: for every chunk, `vector_store.json` keeps the
**three things a vector database stores** — the text, its embedding
(384 numbers), and the metadata. (This is a JSON file standing in for a
real vector database; the search step works identically.)

### Step 3 — Vector search

```bash
python step3_search.py
```

```
Question: Can I take a day off to attend my sister's wedding?

Keyword search — chunks containing the exact words ['attend', 'sister', 'take', 'wedding']:
  60 of 226 chunks matched. First few:

   - [Front matter  p.1]  issues of poverty within Los Angeles County. ...
   - [II. GENERAL EMPLOYMENT INFORMATION > C. Harassment  p.7]  creates an
     intimidating, offensive, or hostile working environment ...

Vector search — top 5 chunks by cosine similarity:

  0.39   [V. BENEFITS AND LEAVES OF ABSENCE > C. Holidays  p.15]
          of vacation or sick leave by the [TITLE]. Without prior approval, ...
  0.37   [V. BENEFITS AND LEAVES OF ABSENCE > L. Bereavement Leave  p.28]
          Employees who wish to take time off due to the death of an immediate
          family member should notify their supervisor. ...
  ...
```

Notice what just happened:

- The word **"wedding" appears nowhere in the handbook** — so keyword
  search can only match generic words like "attend" and "take", and it
  dutifully returns 60 mostly-irrelevant chunks.
- Vector search never compares words. It compares **meaning**: the
  question's embedding lands closest to chunks about *taking time off,
  vacation, and family events*. That's the whole promise of
  embeddings, in one command.

### Step 4 — Get an answer

```bash
python step4_answer.py "Can I take a day off to attend my sister's wedding?"
```

```
Question: Can I take a day off to attend my sister's wedding?

Retrieved chunks (R):
  0.39   [V. BENEFITS AND LEAVES OF ABSENCE > C. Holidays  p.15]
  0.37   [VI. STANDARDS OF CONDUCT > L. Visitors in the Workplace  p.38]
  0.37   [V. BENEFITS AND LEAVES OF ABSENCE > L. Ber eavem ent Leave  p.28]
  0.36   [IV. PAYROLL, SCHEDULING AND OVERTIME PRACTICES > C. Meals and Rest Bre aks  p.11]
  0.36   [IV. PAYROLL, SCHEDULING AND OVERTIME PRACTICES > C. Meals and Rest Bre aks  p.11]

Answer (from Qwen3.8-27B):

  The provided handbook excerpts do not contain a policy about taking a
  day off to attend a sister's wedding. I can't confirm whether you may
  take that day off based only on these excerpts.

Sources: V. BENEFITS AND LEAVES OF ABSENCE > C. Holidays (p.15), ...
```

(Real output from a local Qwen3.8-27B run; other models will word it
differently but behave the same.)

This is R = retrieve, A = augment the prompt with the context,
G = generate. And notice what an honest RAG system does when the
handbook has no wedding-specific policy: it says so, instead of
inventing a policy that doesn't exist. No hallucinated "5 personal
days per year" — because the model was only ever shown what it found.

**No API key?** Add `--dry-run` to see the exact prompt without calling
the LLM:

```bash
python step4_answer.py --dry-run "Do I get paid if I am called for jury duty?"
```

## Try your own questions

```bash
python step4_answer.py "How much vacation do I get after one year?"
python step4_answer.py "Do I get paid if I am called for jury duty?"
python step4_answer.py "Can I smoke in the office?"
python step4_answer.py "Can I use my personal phone during work hours?"
python step4_answer.py "What is the policy on remote work?"
```

The last one is a good experiment: the handbook says little about remote
work, so watch the model *admit what the handbook doesn't say* — a
grounded RAG app's superpower (and, in a way, its limitation).

## Swapping models

Everything is configured through environment variables (in `.env` or in
your shell) — no code changes:

| Variable | Default | What it controls |
|----------|---------|------------------|
| `LLM_API_KEY` | — (required for step 4) | key for your LLM API (a local server may accept `none`) |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | any OpenAI-compatible endpoint |
| `LLM_MODEL` | `gpt-4o-mini` | the chat model |
| `LLM_MAX_TOKENS` | `2000` | budget for the answer |
| `EMBEDDING_MODEL` | `all-MiniLM-L12-v2` | the local embedding model |

Examples:

```bash
# Groq (fast, free tier)
export LLM_BASE_URL=https://api.groq.com/openai/v1
export LLM_MODEL=llama-3.1-8b-instant
export LLM_API_KEY=gsk_...

# A local Ollama server
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=qwen2.5:7b
export LLM_API_KEY=ollama   # any value works with Ollama
```

One quirk to know: **"reasoning" models** (Qwen3, the o-series, ...)
spend token budget *thinking* before they answer. This project
defaults to a generous 2000-token budget precisely so those models
can finish their thinking; if an answer ever comes back empty, the
script tells you to raise `LLM_MAX_TOKENS`.

For embeddings, `all-MiniLM-L6-v2` is a smaller/faster alternative to the
default `all-MiniLM-L12-v2` (both produce 384-number vectors). If you
change the embedding model, **re-run steps 2 and 3** — the stored
embeddings were made with the old model, and a question's embedding is
only comparable to chunks embedded with the *same* model.

## What this version leaves out

Deliberately. It's a learning tool, and each of these is its own
follow-up:

- **A real vector database.** Ours is a JSON file + Python. Fine for a
  handbook; for thousands of documents you'd reach for FAISS, Chroma,
  LanceDB, or a hosted one (Pinecone, Qdrant, ...). Steps 2 and 3 are
  the only files you'd touch.
- **Hybrid search.** Combining keyword + vector search (and merging the
  two result lists) — the best of both worlds.
- **Evaluation.** A small test set of questions + expected answers, so
  you can measure whether a change (bigger chunks, different model,
  reranking) actually makes retrieval better.

One concept at a time.

## Project layout

```
├── data/employee-handbook.pdf   the 40-page source document
├── helpers.py                   .env loading, embedder, LLM client
├── step1_extract_and_chunk.py   PDF → chunks + metadata   → chunks.json
├── step2_embed.py               chunks → embeddings      → vector_store.json
├── step3_search.py              question → top-k chunks (cosine similarity)
├── step4_answer.py              top-k chunks + question → LLM answer
├── requirements.txt
└── .env.example                 copy to .env, add your key
```

`chunks.json` and `vector_store.json` are generated by the steps (and
gitignored); delete them at any time and re-run the steps to rebuild.

*The handbook is a sample template (by Public Counsel), so it contains
placeholders like `[ORGANIZATION]` and `[NAME, TITLE]`. The RAG pipeline
doesn't care — but it's a reminder that your data will be messier than
your demo data.*
