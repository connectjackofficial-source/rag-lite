"""SearchIndex: document store with TF-IDF retrieval and persistence."""
from __future__ import annotations

import json
from pathlib import Path

from .tfidf import TFIDF, STOPWORDS
from .search import cosine_search


class SearchIndex:
    """An in-memory document collection you can query.

    add_document/add_documents rebuild the TF-IDF model incrementally;
    query() returns [{"index", "score", "text"}, ...] ranked by cosine
    similarity; save/load round-trips through JSON.
    """

    def __init__(self, ngram: int = 2, stopwords=STOPWORDS):
        self.ngram = ngram
        self.stopwords = frozenset(stopwords)
        self.docs = []
        self._tfidf = TFIDF([], ngram, self.stopwords)

    def add_document(self, text: str) -> None:
        self.docs.append(text)
        self._tfidf.extend([text])

    def add_documents(self, texts) -> None:
        if not texts:
            return
        self.docs.extend(texts)
        self._tfidf.extend(texts)

    def _rebuild(self) -> None:
        self._tfidf = TFIDF(self.docs, self.ngram, self.stopwords)

    def query(self, text: str, limit: int = 5, threshold: float = 0.0) -> list:
        q = self._tfidf.query_vector(text, self.ngram, self.stopwords)
        hits = cosine_search(q, self._tfidf.vectors, limit, threshold)
        return [{"index": h["index"], "score": h["score"],
                 "text": self.docs[h["index"]]} for h in hits]

    def save(self, path) -> None:
        data = {"docs": self.docs, "ngram": self.ngram,
                "stopwords": sorted(self.stopwords)}
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "SearchIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        idx = cls(ngram=data["ngram"], stopwords=set(data["stopwords"]))
        idx.docs = list(data["docs"])
        idx._rebuild()
        return idx


def save_index(index: SearchIndex, path) -> None:
    index.save(path)


def load_index(path) -> SearchIndex:
    return SearchIndex.load(path)
