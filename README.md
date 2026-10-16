# rag-lite

Zero-dependency local retrieval in pure Python — a tiny, readable RAG
stack with no numpy, no network and no third-party deps.

## What's inside

- **tokenizer** — ASCII words + CJK character n-grams (no dictionary)
- **TF-IDF** — sparse vectors, idf weighting, L2 normalization
- **inverted index** — term → document ids for cheap candidate filtering
- **cosine search** — ranked retrieval with threshold and limit
- **chunker** — overlapping chunks, optional paragraph mode
- **SearchIndex** — in-memory store with JSON persistence
- **keywords** — per-document keyword extraction
- **CLI** — `index`, `add`, `search`

## Install

```bash
pip install .
```

## Library use

```python
from rag_lite import SearchIndex

idx = SearchIndex()
idx.add_documents(["vector search for rag systems",
                   "the lazy dog sleeps"])
hits = idx.query("vector search", limit=1)
# [{'index': 0, 'score': 0.66..., 'text': 'vector search for rag systems'}]
```

## CLI

```bash
rag-lite index ./docs idx.json
rag-lite search idx.json "vector search"
rag-lite add idx.json "another document"
```

## Test

```bash
python tests/test_tokenize.py
python tests/test_tfidf_search.py
python tests/test_chunk_store.py
python tests/test_cli.py
python tests/test_keywords.py
```

## License

MIT
