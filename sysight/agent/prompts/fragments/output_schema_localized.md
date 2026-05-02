# Output Schema: LocalizedFindingSet

```json
{
  "summary": "brief summary of findings",
  "findings": [{
    "finding_id": "{category}:{hash}",
    "category": "C1-C7",
    "title": "one-line description",
    "priority": "high | medium | low",
    "confidence": "confirmed | probable | unresolved",
    "evidence_refs": ["profile evidence references"],
    "file_path": "relative to repo root | null",
    "function": "function name | null",
    "line": "integer | null",
    "description": "what's wrong",
    "suggestion": "how to fix",
    "status": "accepted | rejected | unresolved"
  }],
  "rejected": [],
  "memory_updates": [{
    "path": "wiki page path",
    "content": "markdown content",
    "action": "append | replace | upsert",
    "scope": "workspace | global | benchmark"
  }]
}
```
