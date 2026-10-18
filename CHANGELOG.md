# Changelog

All notable changes to rag-lite.

## 0.3.0 - 2026-10-18

- `BM25` — Okapi BM25 ranking with document-length normalization
- `TFIDF.extend` — incremental idf/vector growth; `SearchIndex.add_*`
  no longer rebuilds the whole corpus
- `expand_query` — edit-distance-1 neighbour terms for query expansion
- `serve` — zero-dependency read-only HTTP retrieval endpoint

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
