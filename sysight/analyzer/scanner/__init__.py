"""scanner — repo file management, search, and analysis CLI tools.

Provides static, read-only analysis of a Python repo:
  fs        — file discovery
  search    — keyword / pattern search
  read      — read file with line numbers
  callsites — find all call-sites of a symbol
  symbols   — list symbols in a file, lookup callers/callees
  variants  — resolve variant/factory key to implementation
"""
from __future__ import annotations
