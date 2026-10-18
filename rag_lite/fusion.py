"""Reciprocal Rank Fusion for hybrid retrieval.

Given several rankings of documents (e.g. keyword TF-IDF, BM25, and a
future neural retriever), RRF merges them into one order without
normalizing scores: each document scores sum(1 / (k + rank_i)).
It is the standard trick for ensemble retrieval in RAG pipelines.
"""
from __future__ import annotations


def reciprocal_rank_fusion(rankings, k: int = 60, top_n: int | None = None):
    """Fuse any number of ranked document lists.

    *rankings* is an iterable of rankables, where each rankable is a
    list of document ids (ranked best-first) or a list of
    (doc_id, score) pairs (the score is ignored, rank decides).

    Returns a list of (doc_id, rrf_score) sorted descending.
    """
    fused: dict = {}
    for ranking in rankings:
        ranked = [item[0] if isinstance(item, tuple) else item
                  for item in ranking]
        for rank, doc in enumerate(ranked, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered if top_n is None else ordered[:top_n]


def fuse_search_results(*rankings, k: int = 60, top_n: int | None = None):
    """Convenience wrapper around :func:`reciprocal_rank_fusion`."""
    return reciprocal_rank_fusion(rankings, k=k, top_n=top_n)
