"""Tests for rag-lite batch 3: reciprocal rank fusion."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_lite.fusion import reciprocal_rank_fusion, fuse_search_results


def test_fusion_prefers_consensus():
    a = ["doc2", "doc1", "doc3"]
    b = ["doc2", "doc1", "doc4"]
    fused = reciprocal_rank_fusion([a, b])
    # doc2 ranks #1 in both lists -> strongest consensus
    assert fused[0][0] == "doc2", fused
    assert fused[1][0] == "doc1", fused
    assert fused[2][0] == "doc3", fused
    print("test_fusion_prefers_consensus: ok")


def test_fusion_accepts_scored_pairs():
    a = [("x", 0.9), ("y", 0.4)]
    b = [("y", 0.8), ("z", 0.7)]
    fused = reciprocal_rank_fusion([a, b])
    names = [d for d, _ in fused]
    assert names == ["y", "x", "z"], names
    print("test_fusion_accepts_scored_pairs: ok")


def test_top_n_truncates():
    a = ["d1", "d2", "d3"]
    b = ["d3", "d2", "d4"]
    fused = fuse_search_results(a, b, top_n=2)
    assert len(fused) == 2
    print("test_top_n_truncates: ok")


if __name__ == "__main__":
    test_fusion_prefers_consensus()
    test_fusion_accepts_scored_pairs()
    test_top_n_truncates()
    print("rag-lite batch3 tests passed")
