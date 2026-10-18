"""BM25 ranking (Okapi BM25) for sparse retrieval.

Complements the cosine/TF-IDF path in search.py: BM25 factors in
document length, so it favours precision on uneven-length corpora.
Pure stdlib — the score for term t in doc d is

    idf(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
"""
from __future__ import annotations

import math

from .tokenize import tokenize


class BM25:
    """Static BM25 scorer over a fixed tokenized collection."""

    def __init__(self, docs, ngram: int = 2, stopwords=None,
                 k1: float = 1.5, b: float = 0.75):
        from .tfidf import STOPWORDS
        stopwords = frozenset(stopwords or STOPWORDS)
        self.k1 = k1
        self.b = b
        self.ngram = ngram
        self.docs = docs
        self.tokenized = [tokenize(d, ngram, stopwords) for d in docs]
        self.doc_len = [len(toks) for toks in self.tokenized]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)
                      if self.doc_len else 0.0)
        self.df = {}
        for toks in self.tokenized:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        n = len(docs)
        self.idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5))
                    for t, d in self.df.items()}

    def _tf_weight(self, freq: int, dl: int) -> float:
        k1, b = self.k1, self.b
        denom = freq + k1 * (1 - b + b * dl / self.avgdl) if self.avgdl else freq
        return freq * (k1 + 1) / denom

    def score_doc(self, terms: list, doc_id: int) -> float:
        toks = self.tokenized[doc_id]
        freq = {}
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
        total = 0.0
        for t in set(terms):
            f = freq.get(t, 0)
            if f:
                total += self.idf.get(t, 0.0) * self._tf_weight(
                    f, self.doc_len[doc_id])
        return total

    def rank(self, query: str, limit: int = 5) -> list:
        """Rank docs by BM25; returns [{"index", "score", "text"}, ...]."""
        terms = tokenize(query, self.ngram,
                         frozenset(getattr(self, "stopwords", None) or [])) \
            if False else tokenize(query, self.ngram, frozenset())
        scored = [(i, self.score_doc(terms, i)) for i in range(len(self.docs))]
        scored.sort(key=lambda p: p[1], reverse=True)
        out = []
        for i, s in scored[:limit]:
            if s > 0.0:
                out.append({"index": i, "score": round(s, 6),
                            "text": self.docs[i]})
        return out
