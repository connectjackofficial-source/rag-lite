"""Tests for rag-lite batch 2: BM25, incremental extend, expansion, HTTP."""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_lite.bm25 import BM25
from rag_lite.store import SearchIndex
from rag_lite.expand import expand_query


def test_bm25_ranks_relevant_first():
    docs = [
        "vector search indexes documents by term frequency",
        "python packaging with pyproject and setuptools",
        "retrieval augmented generation chains", ]
    bm = BM25(docs)
    hits = bm.rank("vector search", limit=3)
    assert hits[0]["index"] == 0, hits
    assert hits[0]["score"] > 0.0
    print("test_bm25_ranks_relevant_first: ok")


def test_bm25_doc_length_effect():
    # one mention in a short doc beats one mention in a long doc
    docs = ["cat", "cat " + "dog fox bird lion " * 30]
    bm = BM25(docs)
    hits = bm.rank("cat", limit=2)
    assert hits[0]["index"] == 0, hits
    print("test_bm25_doc_length_effect: ok")


def test_incremental_extend():
    idx = SearchIndex()
    idx.add_documents(["alpha beta", "gamma delta"])
    before = idx.query("beta", limit=5)
    assert any(h["index"] == 0 for h in before)
    # append without full rebuild
    idx.add_document("alpha omega")
    assert len(idx.docs) == 3
    after = idx.query("omega", limit=5)
    assert any(h["index"] == 2 for h in after), after
    # old documents still retrievable
    old = idx.query("delta", limit=5)
    assert any(h["index"] == 1 for h in old), old
    print("test_incremental_extend: ok")


def test_expand_adds_neighbours():
    vocab = {"vector", "vectors", "vektor", "python"}
    w = expand_query("vector", vocab, ngram=1)
    assert w.get("vector") == 1.0
    assert w.get("vectors") == 0.5
    assert w.get("vektor") == 0.5
    assert "python" not in w
    print("test_expand_adds_neighbours: ok")


def test_http_serve():
    idx = SearchIndex()
    idx.add_documents(["alpha beta gamma", "delta epsilon"])
    from rag_lite.serve import serve
    httpd = serve(idx, port=0, blocking=False)
    port = httpd.server_address[1]
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/?q=beta") as r:
            data = json.loads(r.read().decode("utf-8"))
        assert data["count"] >= 1, data
        assert data["hits"][0]["index"] == 0, data
    finally:
        httpd.shutdown()
    print("test_http_serve: ok")


if __name__ == "__main__":
    test_bm25_ranks_relevant_first()
    test_bm25_doc_length_effect()
    test_incremental_extend()
    test_expand_adds_neighbours()
    test_http_serve()
    print("rag-lite batch2 tests passed")
