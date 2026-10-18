"""TF-IDF vectorization and inverted index construction.

Each document becomes a sparse dict {term: weight} where weight is
term-frequency * idf, L2-normalized. The inverted index maps terms to
document ids so search can skip irrelevant documents entirely.
"""
from __future__ import annotations

import math

from .tokenize import tokenize

STOPWORDS = frozenset(
    "the a an and or of to in on for with at by from as is are was were be been"
    " this that these those it its he she they we you i my your our their"
    " 的 了 和 是 在 我 你 他 她 它 我们 你们 他们 一个 这个 那个 有 不 也 就 都 与 或"
    .split()
)


class TFIDF:
    """Sparse TF-IDF vectors for a fixed document collection."""

    def __init__(self, docs, ngram: int = 2,
                 stopwords=STOPWORDS):
        self.ngram = ngram
        self.stopwords = frozenset(stopwords)
        tokenized = [tokenize(d, ngram, self.stopwords) for d in docs]
        n = len(docs)
        df = {}
        for toks in tokenized:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._df = df
        self.idf = {t: 1.0 + math.log(n / (1.0 + d)) for t, d in df.items()}
        self.vectors = [self._vector(toks) for toks in tokenized]

    def _vector(self, toks) -> dict:
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        raw = {t: c * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in raw.values()))
        if norm == 0.0:
            return {}
        return {t: v / norm for t, v in raw.items()}

    def query_vector(self, text: str, ngram: int = 2,
                     stopwords=STOPWORDS) -> dict:
        return self._vector(tokenize(text, ngram, stopwords))

    def extend(self, docs) -> None:
        """Incrementally add documents without re-tokenizing old ones.

        Document frequencies accumulate so idf reflects the new corpus
        size; existing vectors keep their previously computed weights
        (a documented approximation that keeps append O(new) rather
        than O(all)).
        """
        if not docs:
            return
        new_toks = [tokenize(d, self.ngram, self.stopwords) for d in docs]
        n = len(self.vectors) + len(new_toks)
        df = dict(self._df)
        for toks in new_toks:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._df = df
        self.idf = {t: 1.0 + math.log(n / (1.0 + d)) for t, d in df.items()}
        self.vectors.extend(self._vector(toks) for toks in new_toks)


def build_index(docs, ngram: int = 2, stopwords=STOPWORDS) -> dict:
    """Inverted index: {term: [doc_id, ...]} plus per-doc token lists."""
    tokenized = [tokenize(d, ngram, stopwords) for d in docs]
    index = {}
    for i, toks in enumerate(tokenized):
        for t in set(toks):
            index.setdefault(t, []).append(i)
    return {"index": index, "tokenized": tokenized}
