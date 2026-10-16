"""Tests for rag_lite.chunk and rag_lite.store."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_lite.chunk import chunk_document
from rag_lite.store import SearchIndex, save_index, load_index


def test_plain_chunks():
    chunks = chunk_document("a" * 100, chunk_size=30, overlap=5)
    assert len(chunks) == 4
    assert chunks[0] == "a" * 30
    assert all(len(c) <= 30 for c in chunks)
    assert "".join(chunks).count("a") == 100 + 5 * 3  # overlap repeats
    print("test_plain_chunks: ok")


def test_overlap_invalid():
    try:
        chunk_document("x", chunk_size=10, overlap=10)
        assert False, "should raise"
    except ValueError:
        pass
    print("test_overlap_invalid: ok")


def test_by_paragraph():
    doc = "short\n\n" + "b" * 50 + "\n\n" + "c" * 10
    chunks = chunk_document(doc, chunk_size=20, overlap=0,
                            by_paragraph=True)
    assert chunks[0] == "short"
    assert all(len(c) <= 20 for c in chunks)
    print("test_by_paragraph: ok")


def test_empty():
    assert chunk_document("") == []
    print("test_empty: ok")


def test_store_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        idx = SearchIndex()
        idx.add_documents([
            "python vector search for rag",
            "the quick brown fox",
            "sqlite persistence example",
        ])
        hits = idx.query("vector search", limit=1)
        assert hits and hits[0]["index"] == 0

        p = Path(td) / "idx.json"
        save_index(idx, p)
        loaded = load_index(p)
        assert loaded.docs == idx.docs
        assert loaded.ngram == idx.ngram
        hits2 = loaded.query("fox", limit=1)
        assert hits2[0]["index"] == 1
    print("test_store_roundtrip: ok")


if __name__ == "__main__":
    test_plain_chunks()
    test_overlap_invalid()
    test_by_paragraph()
    test_empty()
    test_store_roundtrip()
    print("rag-lite chunk+store tests passed")
