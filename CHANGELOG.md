# Changelog

All notable changes to rag-lite.

## 0.2.0 - 2026-10-16

- `keyword_extract` — per-document keyword ranking
- CLI subcommands: `index`, `add`, `search`
- `SearchIndex` with JSON `save`/`load`

## 0.1.0 - 2026-10-16

- `tokenize` — ASCII words + CJK n-grams with stopword filtering
- `TFIDF` — sparse vectors, idf weighting, L2 normalization
- `build_index` — term → document-id inverted index
- `cosine_similarity` / `cosine_search` — ranked retrieval
- `chunk_document` — overlapping chunks, paragraph mode
- Package scaffold: pyproject, MIT license, console script `rag-lite`
