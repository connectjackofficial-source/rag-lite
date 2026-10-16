"""Cosine similarity search over sparse vectors."""
from __future__ import annotations

import math


def cosine_similarity(a: dict, b: dict) -> float:
    """Cosine similarity of two sparse {term: weight} dicts."""
    if not a or not b:
        return 0.0
    dot = 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    for t, v in small.items():
        w = large.get(t)
        if w is not None:
            dot += v * w
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def cosine_search(query_vec: dict, vectors, limit: int = 5,
                  threshold: float = 0.0) -> list:
    """Rank document vectors against a query vector.

    Returns [{"index": int, "score": float}] sorted by score desc,
    filtered by threshold, capped at limit.
    """
    scored = [(i, cosine_similarity(query_vec, v))
              for i, v in enumerate(vectors) if v]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"index": i, "score": s}
            for i, s in scored[:limit] if s >= threshold]
