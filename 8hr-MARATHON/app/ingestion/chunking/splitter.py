from typing import List
import logfire

# Coarsest to finest. pypdf and the HTML loader mostly emit single newlines,
# so splitting on "\n\n" alone used to leave whole documents as ONE chunk
# (up to ~62K chars) — far past the embedding and reranker input limits.
SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """Recursively split text until every piece fits in chunk_size."""
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        # Nothing left to split on (long URL, base64 blob): hard cut.
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, finer = separators[0], separators[1:]
    pieces: List[str] = []
    current = ""

    for part in text.split(sep):
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            pieces.append(current)
        # A single part can itself be too big — recurse with a finer separator.
        if len(part) > chunk_size:
            pieces.extend(_split(part, chunk_size, finer))
            current = ""
        else:
            current = part

    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """
    Recursive character splitter: paragraphs → lines → sentences → words.
    Every chunk is at most chunk_size + overlap chars, and carries the last
    `overlap` chars of the previous chunk so facts spanning a boundary survive.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        base = [p.strip() for p in _split(text, chunk_size, SEPARATORS) if p.strip()]

        chunks: List[str] = []
        for i, piece in enumerate(base):
            if i > 0 and overlap > 0:
                tail = base[i - 1][-overlap:]
                # Start the overlap on a word boundary.
                space = tail.find(" ")
                if 0 <= space < len(tail) - 1:
                    tail = tail[space + 1:]
                piece = f"{tail} {piece}"
            chunks.append(piece)

        logfire.info(f"✅ Generated {len(chunks)} chunks")
        return chunks
