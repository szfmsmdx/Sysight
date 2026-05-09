"""cuda_timer — lightweight GPU timing utility based on torch.cuda.Event.

The Instrument stage writes a single ``_sysight_timer.py`` file at the repo
root (or src root) and then adds one ``import`` line to each instrumented
file.  This avoids duplicating the class definition in every file.

Constants exported:
  CUDA_TIMER_MODULE_CONTENT  — the full content of _sysight_timer.py
  CUDA_TIMER_MODULE_NAME     — filename (without .py) written into the repo
  CUDA_TIMER_IMPORT_LINE     — the import line prepended to each target file
"""

from __future__ import annotations

# Name of the helper module that will be written into the repo root.
CUDA_TIMER_MODULE_NAME = "_sysight_timer"

# Full content of the helper module.
CUDA_TIMER_MODULE_CONTENT = '''\
# ── Sysight cuda_timer (auto-injected, safe to delete after optimization) ──
from __future__ import annotations

import contextlib
import torch


class _CudaTimer:
    """Per-region GPU timer backed by torch.cuda.Event.

    Usage::

        timer = _CudaTimer("forward")
        with timer():
            output = model(input)

    All recorded times are printed with the ``[SYSIGHT_TIMER]`` prefix.
    """
    _registry: list["_CudaTimer"] = []

    def __init__(self, label: str) -> None:
        self.label = label
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self.elapsed_ms: list[float] = []
        _CudaTimer._registry.append(self)

    @contextlib.contextmanager
    def __call__(self):  # type: ignore[override]
        self._start.record()
        try:
            yield
        finally:
            self._end.record()
            torch.cuda.synchronize()
            ms = self._start.elapsed_time(self._end)
            self.elapsed_ms.append(ms)
            print(f"[SYSIGHT_TIMER] {self.label}: {ms:.3f} ms")

    @classmethod
    def summary(cls) -> None:
        """Print aggregate stats for all registered timers."""
        for t in cls._registry:
            if not t.elapsed_ms:
                continue
            n = len(t.elapsed_ms)
            avg = sum(t.elapsed_ms) / n
            lo = min(t.elapsed_ms)
            hi = max(t.elapsed_ms)
            print(
                f"[SYSIGHT_TIMER] {t.label} summary: "
                f"n={n} avg={avg:.3f}ms min={lo:.3f}ms max={hi:.3f}ms"
            )


cuda_timer = _CudaTimer
# ── End Sysight cuda_timer ──
'''

# Import line inserted at the top of each instrumented source file.
# The module is placed at the repo root, so a bare import works for any file
# that is run with the repo root on sys.path (standard practice).
CUDA_TIMER_IMPORT_LINE = (
    "# ── sysight-injected ──\n"
    f"from {CUDA_TIMER_MODULE_NAME} import cuda_timer  # noqa: E402\n"
    "# ── end sysight-injected ──"
)

# ---------------------------------------------------------------------------
# Legacy alias kept for backwards-compat during the transition period.
# TODO: remove once all call-sites are updated.
# ---------------------------------------------------------------------------
CUDA_TIMER_TEMPLATE = CUDA_TIMER_MODULE_CONTENT
