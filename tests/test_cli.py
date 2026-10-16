"""Tests for rag_lite.cli (subprocess-level smoke tests)."""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _run(*args):
    return subprocess.run([sys.executable, "-m", "rag_lite.cli", *args],
                          capture_output=True, text=True)


def test_index_and_search():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "docs"
        d.mkdir()
        (d / "a.txt").write_text("vector search for rag systems")
        (d / "b.txt").write_text("the lazy dog sleeps")
        idx = Path(td) / "idx.json"

        r = _run("index", str(d), str(idx))
        assert r.returncode == 0, r.stderr
        assert "indexed 2 chunks" in r.stdout

        r = _run("search", str(idx), "vector search", "--limit", "1")
        assert r.returncode == 0, r.stderr
        assert "[0]" in r.stdout
        assert "1 hits" in r.stdout
    print("test_index_and_search: ok")


def test_add():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "docs"
        d.mkdir()
        (d / "a.txt").write_text("only one doc")
        idx = Path(td) / "idx.json"
        _run("index", str(d), str(idx))

        r = _run("add", str(idx), "second document here")
        assert r.returncode == 0, r.stderr
        assert "2 chunks total" in r.stdout

        r = _run("search", str(idx), "second")
        assert r.returncode == 0
        assert "[1]" in r.stdout
    print("test_add: ok")


if __name__ == "__main__":
    test_index_and_search()
    test_add()
    print("rag-lite cli tests passed")
