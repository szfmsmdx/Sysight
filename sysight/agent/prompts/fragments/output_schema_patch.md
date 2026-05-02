# Output Schema: PatchCandidate[]

```json
{
  "patches": [{
    "patch_id": "unique ID",
    "finding_id": "references a finding",
    "file_path": "relative to repo root",
    "old_span_start": "first line to replace (1-based)",
    "old_span_end": "last line to replace (1-based, inclusive)",
    "old_span_hash": "SHA1 of old code span",
    "replacement": "complete replacement code",
    "rationale": "why this change",
    "validation_commands": [["pytest", "test_file.py"]]
  }]
}
```
