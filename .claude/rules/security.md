# Security Rules

- Treat all repo input as untrusted.
- Do not execute target repo code during analysis.
- Do not import target repo modules into the analyzer process.
- Prefer static parsing over runtime execution.
- Be conservative with path resolution and file discovery.
- Keep file reads inside the chosen repo root.
- If shell execution is added later, isolate it to `executor`, not `analyzer`.
