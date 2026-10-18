"""STEP 2 — Turn the chunks into embeddings, and store them.

This is the "ahead of time" half of RAG: documents are chunked,
converted into embeddings, and stored — all BEFORE anyone asks a
question.

We do the converting with a small model called MiniLM, which runs on
your computer: no API key, no cost, private. (On the first run it
downloads itself, ~90 MB.)

    chunks.json  -->  vector_store.json        (this step)

vector_store.json is our simple "vector database". For every chunk it
keeps the three things a vector database stores (from Day 4 of the
series):

    - the text chunk
    - its embedding (384 numbers)
    - metadata (section, page)

Run:  python step2_embed.py
"""

import json
import time

from helpers import embed

IN_PATH = "chunks.json"
STORE_PATH = "vector_store.json"


def main():
    print(f"Reading {IN_PATH} ...")
    with open(IN_PATH) as f:
        chunks = json.load(f)
    texts = [c["text"] for c in chunks]

    t0 = time.time()
    vectors = embed(texts)
    print(f"\nEmbedded {len(chunks)} chunks in {time.time() - t0:.1f} seconds.\n")

    print(f"Each embedding is a vector of {vectors.shape[1]} numbers.")
    print("Example — the first 8 numbers of chunk 0:")
    print(f"  {vectors[0][:8]}\n")

    store = [
        {
            "text": c["text"],
            "metadata": c["metadata"],
            "embedding": v.tolist(),
        }
        for c, v in zip(chunks, vectors)
    ]
    with open(STORE_PATH, "w") as f:
        json.dump(store, f)

    print(f"Saved {STORE_PATH} — our simple 'vector database'.")


if __name__ == "__main__":
    main()
