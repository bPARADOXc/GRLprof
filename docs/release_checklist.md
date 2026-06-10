# Release Checklist

Before pushing GRLprof to a public repository:

1. Run `find . -type f -size +50M` and confirm no large artifacts are tracked.
2. Scan for private absolute paths and API key patterns before publishing.
3. Verify `data/`, `checkpoints/`, `llama/`, and `.cache/` are ignored.
4. Publish datasets and checkpoints externally, then add download links to `README.md`.
5. Add the final paper citation to `CITATION.cff` when available.
