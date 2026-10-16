"""rag-lite: zero-dependency local retrieval in pure Python.

A tiny, readable RAG stack: CJK/English tokenizer, TF-IDF vectors with an
inverted index, cosine similarity search, document chunking and JSON
persistence. No numpy, no network, no third-party deps.
"""
from .tokenize import tokenize

__version__ = "0.1.0"
__all__ = ["tokenize"]
