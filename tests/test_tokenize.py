"""Tests for rag_lite.tokenize."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_lite.tokenize import tokenize


def test_ascii():
    assert tokenize("Hello World") == ["hello", "world"]
    assert tokenize("token-counter_v2") == ["token", "counter_v2"]
    print("test_ascii: ok")


def test_cjk_bigram():
    assert tokenize("世界你好") == ["世界", "界你", "你好"]
    assert tokenize("你好") == ["你好"]
    print("test_cjk_bigram: ok")


def test_mixed():
    assert tokenize("Hello 世界你好") == ["hello", "世界", "界你", "你好"]
    print("test_mixed: ok")


def test_short_cjk():
    assert tokenize("A 我 B") == ["a", "b", "我"]
    print("test_short_cjk: ok")


def test_stopwords():
    assert tokenize("the quick fox", stopwords={"the"}) == ["quick", "fox"]
    print("test_stopwords: ok")


def test_empty():
    assert tokenize("") == []
    assert tokenize("!!!") == []
    print("test_empty: ok")


if __name__ == "__main__":
    test_ascii()
    test_cjk_bigram()
    test_mixed()
    test_short_cjk()
    test_stopwords()
    test_empty()
    print("rag-lite tokenize tests passed")
