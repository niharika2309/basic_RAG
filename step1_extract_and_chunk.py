"""STEP 1 — Read the handbook and cut it into chunks.

Before anyone can ask a question, the document has to be prepared:
cut into small, searchable chunks, each tagged with metadata about
where it came from (which section, which page). That metadata is what
lets us cite sources in the final answer (step 4).

    employee-handbook.pdf  -->  chunks.json        (this step)

How it works:
  1. Extract the text of every page, throwing away the noise that PDFs
     add (running headers, page numbers, table-of-contents lines).
  2. Scan for section headers ("V. BENEFITS..." / "B. Vacation...") so
     we always know which section we are inside.
  3. Cut the text into chunks of about CHUNK_SIZE characters. Chunks
     never cross a section boundary, and each one repeats a little of
     the previous chunk, so a sentence cut in half keeps some context.

No AI needed here — just reading a PDF and splitting text.

Run:  python step1_extract_and_chunk.py
"""

import json
import re

import pypdf

PDF_PATH = "data/employee-handbook.pdf"
OUT_PATH = "chunks.json"

CHUNK_SIZE = 600  # target characters per chunk (roughly 100-120 words)
OVERLAP = 50      # repeat this many characters of the previous chunk, so a
                  # sentence that gets cut in half keeps some context

# PDFs are messy. These patterns match the noise that appears on every page
# (running header, bare page numbers, symbols) or in the table of contents,
# so we can throw it away before chunking.
JUNK_LINE = re.compile(
    r"^\s*(?:\d{1,3}\s*$|PUBLIC COUNSEL.*|.*COMMUNITY DEVELOPMENT.*PROJECT.*|.*▪.*)$",
    re.IGNORECASE,
)
TOC_LINE = re.compile(r"\.{5,}")  # a long run of dots = a table-of-contents line

# Section headers look like "V.  BENEFITS AND LEAVES OF ABSENCE" or
# "B.  Vacation Benefits". Matching is a simple heuristic — good enough
# for metadata; the embedding model doesn't need it to be perfect.
#
# How do we tell "V.  BENEFITS..." (a major section) from "I. Pregnancy Leave"
# (a subsection that just happens to be numbered I)? In this handbook, major
# section titles are ALL CAPS, while subsection titles are in normal case.
ROMAN_HEADER = re.compile(r"^\s*([IVX]+)\.\s+([A-Z][A-Z&',/ -]{3,45})\s*$")
LETTER_HEADER = re.compile(r"^\s*([A-Z])\.\s+([A-Za-z][^\n]{3,50})")


def clean_header(name):
    """Tidy a header name.

    PDF extraction is imperfect: it sometimes glues page numbers onto the
    end ('Holidaysxiv') or splits words ('Open Door P olicy'). We can't fix
    everything, but we can do a little cosmetic cleanup.
    """
    name = " ".join(name.split())  # collapse doubled spaces
    for _ in range(3):            # strip trailing roman-numeral junk, e.g. 'xiv'
        cleaned = re.sub(r"[ivxIVX]{1,6}$", "", name).strip()
        if cleaned == name:
            break
        name = cleaned
    return name


# A section marker left on a line by itself ("V." with the title on the next
# line) — a common artifact of PDF text extraction.
ORPHAN_MARKER = re.compile(r"^\s*(?:[IVX]+|[A-Z])\.\s*$")


def extract_lines(path):
    """Return (page_number, line) pairs for the PDF — noise removed, and
    split section headers glued back together."""
    reader = pypdf.PdfReader(path)
    lines = []
    for page_number, page in enumerate(reader.pages, start=1):
        for line in (page.extract_text() or "").splitlines():
            if JUNK_LINE.match(line) or TOC_LINE.search(line):
                continue  # running header, page number, or TOC line
            lines.append((page_number, line))

    # Glue back split headers: marker on one line, title on the next.
    merged = []
    i = 0
    while i < len(lines):
        if ORPHAN_MARKER.match(lines[i][1]) and i + 1 < len(lines):
            merged.append((lines[i + 1][0], lines[i][1].strip() + " " + lines[i + 1][1]))
            i += 2
        else:
            merged.append(lines[i])
            i += 1
    return merged


def update_section(line, section):
    """Update our idea of the current section if `line` looks like a header.

    We try the ALL-CAPS major-section pattern first, so a subsection like
    "I. Pregnancy Leave" (normal case) is not mistaken for a new major
    section "I.".
    """
    m = ROMAN_HEADER.match(line)
    if m:
        section["roman"] = f"{m.group(1)}. {clean_header(m.group(2))}"
        section["letter"] = None
        return
    m = LETTER_HEADER.match(line)
    if m and section.get("roman"):
        section["letter"] = f"{m.group(1)}. {clean_header(m.group(2))}"


def section_label(section):
    parts = [p for p in (section.get("roman"), section.get("letter")) if p]
    return " > ".join(parts) if parts else "Front matter"


def group_into_segments(lines):
    """Group the document into runs of lines that all belong to one section.

    A page can contain several sections, so instead of labeling a whole
    page we track the section *line by line*. Each segment remembers the
    section it belongs to.
    """
    segments = []
    section = {}
    current = None
    for page_number, line in lines:
        update_section(line, section)
        label = section_label(section)
        if current is None or current["section"] != label:
            if current:
                segments.append(current)
            current = {"section": label, "lines": []}
        current["lines"].append((page_number, line))
    if current:
        segments.append(current)
    return segments


def chunk_segment(segment, size=CHUNK_SIZE, overlap=OVERLAP):
    """Cut one segment into chunks of about `size` characters.

    We split on word boundaries (never mid-word) and start each new chunk
    a few words before the previous one ended. Each chunk is tagged with
    the page of its first word.
    """
    words = []  # a list of (page_number, word) pairs
    for page_number, line in segment["lines"]:
        words.extend((page_number, w) for w in line.split())

    chunks = []
    overlap_words = max(1, overlap // 5)  # 50 characters ≈ 10 words
    start = 0
    while start < len(words):
        # Grow the current chunk until it would exceed `size` characters.
        end, length = start, 0
        while end < len(words):
            added = len(words[end][1]) + (1 if length > 0 else 0)
            if length > 0 and length + added > size:
                break
            length += added
            end += 1
        chunks.append(
            {
                "text": " ".join(w for _, w in words[start:end]),
                "page": words[start][0],
            }
        )
        if end >= len(words):
            break
        start = max(start + 1, end - overlap_words)  # a little overlap

    return [
        {
            "text": c["text"],
            "metadata": {
                "source": "employee-handbook.pdf",
                "section": segment["section"],
                "page": c["page"],
            },
        }
        for c in chunks
    ]


def main():
    print(f"Reading {PDF_PATH} ...")
    lines = extract_lines(PDF_PATH)

    segments = group_into_segments(lines)
    chunks = [c for seg in segments for c in chunk_segment(seg)]

    with open(OUT_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"\nDone! {len(segments)} sections  ->  {len(chunks)} chunks  ->  {OUT_PATH}\n")
    print("A few sample chunks (the metadata is what lets us cite sources later):\n")
    for c in (chunks[10], chunks[60], chunks[100]):
        m = c["metadata"]
        print(f"  [{m['section']}  p.{m['page']}]")
        print(f"  {c['text'][:150]}...\n")


if __name__ == "__main__":
    main()
