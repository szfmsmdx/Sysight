"""sysight.analyzer.repo — re-export for backward compatibility.

Public names previously living directly in analyzer.py are re-exported here
so that test and external code can import them from this stable path.
"""

from .analyzer import (  # noqa: F401
    build_dag,
    scan_repo,
    FileDAG,
)
