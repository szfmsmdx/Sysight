# Testing Rules

- Every new analyzer feature needs at least one focused unit test.
- Tests should create tiny synthetic repos in temp directories.
- Do not depend on external repos for unit tests.
- Keep tests deterministic and fast.
- Document important static-analysis limitations in tests when relevant.
- Before declaring an analyzer change done, run:

```bash
python3 -m unittest -v test_analyzer.py
```
