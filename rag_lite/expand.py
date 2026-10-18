"""Query expansion: neighbour terms enrich the query vector.

When a query term is rare, exact matching misses good documents whose
wording differs by one edit (typos, plural forms, word variants).
expand_query() returns a dict {term: weight} where each query term
keeps its original weight plus a discounted weight for every dictionary
term within edit distance 1 — capped so the query stays focused.
"""
from __future__ import annotations

from .tokenize import tokenize


def _edits1(word: str):
    """All strings within one edit of *word* (deletion, transposition,
    replacement, insertion)."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits for c in "abcdefghijklmnopqrstuvwxyz" if b]
    inserts = [a + c + b for a, b in splits for c in "abcdefghijklmnopqrstuvwxyz"]
    return set(deletes + transposes + replaces + inserts)


def expand_query(query: str, vocabulary: set, ngram: int = 2,
                 stopwords=frozenset(), discount: float = 0.5,
                 max_per_term: int = 3) -> dict:
    """Expand a query into a weighted term dict.

    Each query term gets weight 1.0; every vocabulary term at edit
    distance 1 (same length class preferred) contributes at most
    *max_per_term* neighbours at weight *discount*.
    """
    terms = tokenize(query, ngram, stopwords)
    weights = {t: 1.0 for t in terms}
    if ngram != 1:
        return weights  # expansion targets single words only
    for t in set(terms):
        if len(t) < 4:
            continue
        cands = sorted(v for v in vocabulary if v in _edits1(t) and v != t)
        for v in cands[:max_per_term]:
            weights[v] = max(weights.get(v, 0.0), discount)
    return weights
