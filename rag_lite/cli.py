"""Command-line interface for rag-lite.

Subcommands:
  index DIR OUT.json    build an index from every file under DIR
  add INDEX.json TEXT   add one document and save back
  search INDEX.json Q   query the index, print top hits
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .store import SearchIndex, save_index, load_index
from .chunk import chunk_document


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-lite", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("index", help="build an index from a directory")
    i.add_argument("dir")
    i.add_argument("out")

    a = sub.add_parser("add", help="add a document to an index")
    a.add_argument("index")
    a.add_argument("text")

    s = sub.add_parser("search", help="query an index")
    s.add_argument("index")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=5)
    s.add_argument("--threshold", type=float, default=0.0)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "index":
        idx = SearchIndex()
        root = Path(args.dir)
        for f in sorted(root.rglob("*")):
            if f.is_file():
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for chunk in chunk_document(text):
                    idx.add_document(chunk)
        save_index(idx, args.out)
        print(f"indexed {len(idx.docs)} chunks from {root}")
    elif args.cmd == "add":
        idx = load_index(args.index)
        idx.add_document(args.text)
        save_index(idx, args.index)
        print(f"added document, {len(idx.docs)} chunks total")
    elif args.cmd == "search":
        idx = load_index(args.index)
        hits = idx.query(args.query, limit=args.limit,
                         threshold=args.threshold)
        for h in hits:
            text = h["text"].replace("\n", " ")[:80]
            print(f"{h['score']:.4f}  [{h['index']}]  {text}")
        print(f"{len(hits)} hits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
