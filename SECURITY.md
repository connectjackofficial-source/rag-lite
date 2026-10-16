# Security

- `SearchIndex.save/load` only reads and writes JSON files you point it at.
- `index` CLI walks a directory recursively; point it only at content you
  trust. Files that fail to read are skipped silently by design.
- No network access, no eval, no subprocess in library code.
- If you find a vulnerability, open an issue privately with a minimal
  reproducer before any disclosure.
