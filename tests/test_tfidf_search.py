"""Tests for rag_lite.tfidf and rag_lite.search."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_lite.tfidf import TFIDF, build_index
from rag_lite.search import cosine_similarity, cosine_search


DOCS = [
    "the quick brown fox jumps over the lazy dog",
    "a fox is quick but the dog is lazy",
    "python implements vector search for rag",
    "rag systems retrieve documents with cosine similarity",
]


def test_idf_penalizes_common():
    t = TFIDF(DOCS, stopwords=frozenset())  # keep "the" for the check
    # "the" appears in 2 docs, "vector" in 1 -> idf("the") < idf("vector")
    assert t.idf["the"] < t.idf["vector"]
    print("test_idf_penalizes_common: ok")


def test_vectors_normalized():
    t = TFIDF(DOCS, stopwords=frozenset())
    for v in t.vectors:
        norm = sum(x * x for x in v.values()) ** 0.5
        assert norm == 1.0 or norm == 0.0
    print("test_vectors_normalized: ok")


def test_cosine_identical():
    v = {"a": 1.0, "b": 2.0}
    assert abs(cosine_similarity(v, dict(v)) - 1.0) < 1e-9
    print("test_cosine_identical: ok")


def test_search_rank():
    t = TFIDF(DOCS, stopwords=frozenset())
    q = t.query_vector("fox dog lazy", stopwords=frozenset())
    hits = cosine_search(q, t.vectors, limit=2)
    assert len(hits) == 2
    assert hits[0]["index"] in (0, 1)  # fox/dog docs on top
    print("test_search_rank: ok")


def test_build_index():
    idx = build_index(DOCS, stopwords=frozenset())
    assert 0 in idx["index"]["fox"]
    assert 3 in idx["index"]["cosine"]
    print("test_build_index: ok")


if __name__ == "__main__":
    test_idf_penalizes_common()
    test_vectors_normalized()
    test_cosine_identical()
    test_search_rank()
    test_build_index()
    print("rag-lite tfidf+search tests passed")
