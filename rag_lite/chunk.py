"""Document chunking for retrieval preprocessing.

Splits long text into overlapping chunks so each retrieval unit stays
small enough for a context window, while overlap preserves boundary
context between chunks.
"""
from __future__ import annotations


def _split_plain(text: str, chunk_size: int, overlap: int) -> list:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap must be in [0, chunk_size)")
    chunks = []
    i, n = 0, len(text)
    if n == 0:
        return []
    while True:
        chunks.append(text[i:i + chunk_size])
        if i + chunk_size >= n:
            break
        i += chunk_size - overlap
    return chunks


def chunk_document(text: str, chunk_size: int = 200,
                   overlap: int = 20, by_paragraph: bool = False) -> list:
    """Split a document into chunks.

    With by_paragraph=True, split on blank lines first and only further
    split paragraphs longer than chunk_size. Otherwise a plain
    fixed-size sliding window.
    """
    if not by_paragraph:
        return _split_plain(text, chunk_size, overlap)
    chunks = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            chunks.extend(_split_plain(para, chunk_size, overlap))
    return chunks
