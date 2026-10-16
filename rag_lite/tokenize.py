"""Lightweight tokenizer for mixed CJK/English text.

ASCII words are kept whole (lowercased); CJK runs are cut into
character n-grams (default bigrams). No dictionary, no segmentation
model — deterministic and dependency-free.
"""
from __future__ import annotations

import re

_ASCII = re.compile(r"[a-z0-9_]+")
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")


def _cjk_runs(text: str) -> list:
    runs = []
    cur = []
    for ch in text:
        if _CJK.match(ch):
            cur.append(ch)
        elif cur:
            runs.append("".join(cur))
            cur = []
    if cur:
        runs.append("".join(cur))
    return runs


def tokenize(text: str, ngram: int = 2,
             stopwords=frozenset()) -> list:
    """Split text into tokens: ASCII words + CJK n-grams.

    >>> tokenize("Hello 世界你好")
    ['hello', '世界', '界你', '你好']
    """
    if ngram <= 0:
        raise ValueError("ngram must be >= 1")
    text = text.lower()
    toks = _ASCII.findall(text)
    for run in _cjk_runs(text):
        if len(run) < ngram:
            toks.append(run)
        else:
            toks.extend(run[i:i + ngram] for i in range(len(run) - ngram + 1))
    if stopwords:
        toks = [t for t in toks if t not in stopwords]
    return toks
