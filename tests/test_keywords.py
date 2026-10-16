"""Tests for rag_lite.keywords."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_lite.keywords import keyword_extract


def test_top_terms():
    doc = ("vector vector vector search search rag retrieval "
           "the quick fox")
    kw = keyword_extract(doc, limit=3)
    assert kw[0]["term"] == "vector"
    assert kw[1]["term"] == "search"
    assert len(kw) == 3
    assert all(k["weight"] > 0 for k in kw)
    print("test_top_terms: ok")


def test_stopwords_excluded():
    kw = keyword_extract("the the the and and or or document", limit=5)
    assert all(k["term"] not in ("the", "and", "or") for k in kw)
    print("test_stopwords_excluded: ok")


def test_empty():
    assert keyword_extract("", limit=5) == []
    print("test_empty: ok")


if __name__ == "__main__":
    test_top_terms()
    test_stopwords_excluded()
    test_empty()
    print("rag-lite keywords tests passed")
