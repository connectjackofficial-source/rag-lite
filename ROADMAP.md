# Roadmap

- Incremental `add_document` without full TF-IDF rebuild (batch idf updates)
- BM25 scoring as an alternative ranking function
- Query expansion: add top n-gram neighbors of query terms
- Binary index format (jsonl) for large collections
- Stopword auto-detection from collection frequency
- `serve` subcommand: tiny HTTP search endpoint (stdlib only)
