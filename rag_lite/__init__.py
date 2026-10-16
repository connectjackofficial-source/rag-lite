"""rag-lite: zero-dependency local retrieval in pure Python.

A tiny, readable RAG stack: CJK/English tokenizer, TF-IDF vectors with an
inverted index, cosine similarity search, document chunking and JSON
persistence. No numpy, no network, no third-party deps.
"""
from .tokenize import tokenize
from .tfidf import TFIDF, build_index
from .search import cosine_search
from .chunk import chunk_document
from .store import SearchIndex, save_index, load_index

__version__ = "0.1.0"
__all__ = ["tokenize", "TFIDF", "build_index", "cosine_search",
           "chunk_document", "SearchIndex", "save_index", "load_index"]
