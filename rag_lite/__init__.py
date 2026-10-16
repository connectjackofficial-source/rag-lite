"""rag-lite: zero-dependency local retrieval in pure Python.

A tiny, readable RAG stack: CJK/English tokenizer, TF-IDF vectors with an
inverted index, cosine similarity search, document chunking, keyword
extraction and JSON persistence. No numpy, no network, no third-party deps.
"""
from .tokenize import tokenize
from .tfidf import TFIDF, build_index
from .search import cosine_similarity, cosine_search
from .chunk import chunk_document
from .store import SearchIndex, save_index, load_index
from .keywords import keyword_extract

__version__ = "0.2.0"
__all__ = ["tokenize", "TFIDF", "build_index", "cosine_similarity",
           "cosine_search", "chunk_document", "SearchIndex", "save_index",
           "load_index", "keyword_extract"]
