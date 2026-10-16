"""Keyword extraction: top terms by TF-IDF weight per document."""
from __future__ import annotations

from .tfidf import TFIDF, STOPWORDS


def keyword_extract(doc: str, limit: int = 5, ngram: int = 2,
                    stopwords=STOPWORDS) -> list:
    """Return the top keywords of a single document.

    Ranking is done against a one-document collection, so idf collapses
    to a constant and terms rank by raw TF (normalized). Returns
    [{"term": str, "weight": float}] sorted desc, capped at limit.
    """
    t = TFIDF([doc], ngram, stopwords)
    vec = t.vectors[0]
    ranked = sorted(vec.items(), key=lambda kv: kv[1], reverse=True)
    return [{"term": term, "weight": round(w, 6)}
            for term, w in ranked[:limit]]
