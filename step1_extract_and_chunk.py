"""STEP 1 — Read the handbook and cut it into chunks.

Before anyone can ask a question, the document has to be prepared:
cut into small pieces, each tagged with the page it came from.
The page tag is what lets us cite sources in the final answer (step 4).

    employee-handbook.pdf  -->  chunks.json        (this step)

No AI needed here — just reading a PDF and splitting text.

Run:  python step1_extract_and_chunk.py
"""

import json

import pypdf

PDF_PATH = "data/employee-handbook.pdf"
OUT_PATH = "chunks.json"

CHUNK_SIZE = 600  # target characters per chunk (about 100 words)
OVERLAP = 10      # words of overlap between consecutive chunks


def read_pages(path):
    """Extract the text of every page, as (page_number, text) pairs."""
    reader = pypdf.PdfReader(path)
    return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]


def chunk_text(text):
    """Cut one page of text into chunks of about CHUNK_SIZE characters.

    We split between words (never mid-word), and each new chunk starts
    a few words before the previous one ended (the OVERLAP), so a
    sentence that gets cut in half keeps some context.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j, size = i, 0
        while j < len(words):
            size += len(words[j]) + (1 if j > i else 0)
            if j > i and size > CHUNK_SIZE:
                break
            j += 1
        chunks.append(" ".join(words[i:j]))
        if j >= len(words):
            return chunks
        i = max(i + 1, j - OVERLAP)
    return chunks


def main():
    print(f"Reading {PDF_PATH} ...")
    pages = read_pages(PDF_PATH)

    chunks = [
        {"text": piece, "metadata": {"page": page}}
        for page, text in pages
        for piece in chunk_text(text)
    ]

    with open(OUT_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"\nDone! {len(pages)} pages  ->  {len(chunks)} chunks  ->  {OUT_PATH}\n")
    print("A few sample chunks (the page number lets us cite sources later):\n")
    for c in chunks[:3]:
        print(f"  [p.{c['metadata']['page']}]  {c['text'][:120]}...")
    print()


if __name__ == "__main__":
    main()
