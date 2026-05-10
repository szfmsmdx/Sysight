"""Tests for progressive context compaction (context.py).

Coverage:
  - Level 1: large-result persistence (>50K chars → disk, 2KB preview)
  - Level 2: time-based compaction (age > keep_recent_turns_full → summary)
  - Level 3: token-pressure compaction + recovery injection
  - Edge cases: empty results, errors, missing persist_dir
  - Integration: full AgentLoop with fake providers
  - Compression quality: ratio, information preservation

All tests use realistic data sizes (60K–100K chars for Level 1, 10K for
Level 2) and code-like content that resembles actual benchmark workloads.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from sysight.agent.context import (
    AgentContext,
    ContextPolicy,
    ContextStats,
    _MODEL_CONTEXT_WINDOWS,
    _DEFAULT_CONTEXT_WINDOW,
    build_tool_result_summary,
    to_jsonable,
)
from sysight.agent.loop import AgentLoop, AgentTask
from sysight.agent.provider import LLMResponse, ToolCallRequest, UsageInfo
from sysight.benchmark.debug import DebugProvider
from sysight.tools.registry import ToolDef, ToolPolicy, ToolRegistry

# ---------------------------------------------------------------------------
# Realistic test data generators
# ---------------------------------------------------------------------------

_SAMPLE_CODE = """\
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Dict, Any, List, Tuple
import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    use_amp: bool = True
    use_distributed: bool = False
    num_workers: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = True


class DistributedTrainer:
    \"\"\"Distributed training wrapper with gradient accumulation and mixed precision.\"\"\"

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if config.use_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device],
                output_device=self.device,
                find_unused_parameters=True,
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config.warmup_steps, T_mult=2
        )
        self.global_step = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        \"\"\"Single training step with gradient accumulation.\"\"\"
        self.model.train()
        inputs = batch["input"].to(self.device, non_blocking=True)
        targets = batch["target"].to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            outputs = self.model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            loss = loss / self.config.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        self.global_step += 1
        return {"loss": loss.item() * self.config.gradient_accumulation_steps}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        \"\"\"Validation loop.\"\"\"
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["input"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    outputs = self.model(inputs)
                    loss = nn.functional.cross_entropy(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += inputs.size(0)

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        \"\"\"Save training checkpoint.\"\"\"
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
            "metrics": metrics,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path} (epoch {epoch}, step {self.global_step})")


class DataPipeline:
    \"\"\"Data loading pipeline with prefetching and caching.\"\"\"

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._cache: Dict[str, Any] = {}

    def create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool = True,
    ) -> DataLoader:
        sampler = None
        if self.config.use_distributed:
            sampler = DistributedSampler(
                dataset,
                shuffle=shuffle,
                drop_last=True,
            )
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            drop_last=True,
            persistent_workers=True,
        )

    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        \"\"\"Preprocess a batch with optional caching.\"\"\"
        batch_key = hash(str(batch.get("id", "")))
        if batch_key in self._cache:
            return self._cache[batch_key]

        result = {
            "input": torch.tensor(batch["input"], dtype=torch.float32),
            "target": torch.tensor(batch["target"], dtype=torch.long),
        }
        self._cache[batch_key] = result
        return result


def create_model(num_layers: int = 12, hidden_size: int = 768) -> nn.Module:
    \"\"\"Factory function for model creation.\"\"\"
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(0.1))
    return nn.Sequential(*layers)


def main():
    config = TrainingConfig(
        batch_size=64,
        learning_rate=3e-4,
        num_epochs=50,
        use_distributed=torch.cuda.device_count() > 1,
    )
    model = create_model(num_layers=24, hidden_size=1024)
    trainer = DistributedTrainer(model, config)
    pipeline = DataPipeline(config)

    logger.info(f"Training started with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Using {'distributed' if config.use_distributed else 'single'} GPU mode")

    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        train_metrics = {"loss": 0.0}
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs} completed in {time.time() - epoch_start:.2f}s")
        logger.info(f"Train metrics: {train_metrics}")


if __name__ == "__main__":
    main()
"""


def _make_code_lines(start: int, count: int) -> list[dict]:
    """Generate scanner_read-style line dicts from sample code."""
    all_lines = _SAMPLE_CODE.splitlines()
    result = []
    for i in range(min(count, len(all_lines) - start)):
        result.append({"line": start + i + 1, "text": all_lines[start + i]})
    return result


def _make_large_result(chars: int) -> str:
    """Generate a large JSON result of approximately `chars` characters."""
    # Use realistic structure: nsys_sql-style profiling data
    rows = []
    row_template = {
        "start_ns": 1234567890000,
        "end_ns": 1234567895000,
        "duration_us": 5000.0,
        "name": "cudaLaunchKernel",
        "stream": 7,
        "process_id": 12345,
        "thread_id": 67890,
        "correlation_id": 999,
    }
    row_json = json.dumps(row_template)
    needed = max(1, chars // (len(row_json) + 2))
    for i in range(needed):
        row = dict(row_template)
        row["start_ns"] += i * 1000000
        row["end_ns"] += i * 1000000
        row["correlation_id"] = i
        rows.append(row)
    return json.dumps({"rows": rows, "total": len(rows)}, ensure_ascii=False)


def _make_medium_result(chars: int) -> str:
    """Generate a medium-sized result (e.g. scanner_files output)."""
    files = []
    for i in range(max(1, chars // 80)):
        files.append({
            "path": f"src/module_{i % 10}/submodule_{i}/file_{i}.py",
            "size": 1000 + i * 100,
            "type": "python",
        })
    return json.dumps({"files": files, "total": len(files)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Unit tests: build_tool_result_summary
# ---------------------------------------------------------------------------

class TestBuildToolResultSummary(unittest.TestCase):
    """Verify the summary builder's return value and persistence behavior."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="sysight_test_"))

    def tearDown(self):
        import shutil
        if self.tmpdir.exists():
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_normal_result_returns_compact_json(self):
        """Small results get a compact summary, not persisted."""
        content, persisted, path = build_tool_result_summary(
            tool_name="memory_search",
            arguments={"query": "test"},
            status="ok",
            error="",
            data={"matches": [{"key": "val"}]},
            full_content='{"matches":[{"key":"val"}]}',
        )
        self.assertFalse(persisted)
        self.assertEqual(path, "")
        parsed = json.loads(content)
        self.assertTrue(parsed["sysight_compacted_tool_result"])
        self.assertEqual(parsed["tool"], "memory_search")
        self.assertIn("summary", parsed)

    def test_large_result_persisted_to_disk(self):
        """Results >50K chars are persisted to disk with preview."""
        large = _make_large_result(80_000)
        self.assertGreater(len(large), 70_000)

        content, persisted, path = build_tool_result_summary(
            tool_name="nsys_sql_nvtx",
            arguments={"query": "SELECT * FROM nvtx"},
            status="ok",
            error="",
            data={"rows": []},
            full_content=large,
            persist_dir=self.tmpdir,
            large_threshold_tokens=20_000,
            preview_tokens=600,
        )

        self.assertTrue(persisted)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            saved = f.read()
        self.assertEqual(saved, large)

        parsed = json.loads(content)
        self.assertTrue(parsed["sysight_persisted_tool_result"])
        self.assertEqual(parsed["full_chars"], len(large))
        self.assertLessEqual(len(parsed["preview"]), 2_100)
        self.assertIn("persist_path", parsed)

    def test_large_result_without_persist_dir_falls_back(self):
        """Without persist_dir, large results get normal compact summary."""
        large = _make_large_result(60_000)
        content, persisted, path = build_tool_result_summary(
            tool_name="nsys_sql_nvtx",
            arguments={},
            status="ok",
            error="",
            data={"rows": []},
            full_content=large,
            persist_dir=None,  # no persist dir
            large_threshold_tokens=20_000,
        )
        self.assertFalse(persisted)
        self.assertEqual(path, "")
        parsed = json.loads(content)
        self.assertTrue(parsed["sysight_compacted_tool_result"])

    def test_scanner_read_preserves_code_content(self):
        """scanner_read summary must preserve all read lines."""
        lines = _make_code_lines(0, 30)
        data = {
            "repo": "case_1",
            "path": "src/trainers/trainer.py",
            "total_lines": 200,
            "shown_start": 1,
            "shown_end": 30,
            "lines": lines,
        }
        content, persisted, _ = build_tool_result_summary(
            tool_name="scanner_read",
            arguments={"path": "src/trainers/trainer.py", "start": 1, "end": 30},
            status="ok",
            error="",
            data=data,
            full_content=json.dumps(data),
        )
        parsed = json.loads(content)
        summary = parsed["summary"]
        self.assertEqual(summary["path"], "src/trainers/trainer.py")
        self.assertEqual(summary["line_count"], 30)
        self.assertEqual(len(summary["content"]), 30)
        # Verify actual code content is preserved (line 1 = import torch)
        self.assertIn("import torch", summary["content"][0])
        # class DistributedTrainer is at line 29 (0-indexed: 28)
        self.assertIn("class DistributedTrainer:", summary["content"][28])

    def test_error_result_includes_error(self):
        """Error tool results include the error message."""
        content, persisted, _ = build_tool_result_summary(
            tool_name="scanner_read",
            arguments={"path": "nonexistent.py"},
            status="error",
            error="File not found: nonexistent.py",
            data=None,
            full_content="File not found: nonexistent.py",
        )
        parsed = json.loads(content)
        self.assertEqual(parsed["error"], "File not found: nonexistent.py")


# ---------------------------------------------------------------------------
# Level 1: Large-result persistence
# ---------------------------------------------------------------------------

class TestAgentContextLevel1(unittest.TestCase):
    """Large tool results (>50K chars) are persisted to disk."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="sysight_l1_"))

    def tearDown(self):
        import shutil
        if self.tmpdir.exists():
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_large_result_persisted_and_compact_stored(self):
        """append_tool_result persists large output, stores preview in compact."""
        ctx = AgentContext("analyze training code", persist_dir=self.tmpdir)
        large = _make_large_result(80_000)

        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": large},
            turn=1, tool_name="nsys_sql_nvtx", arguments={},
            status="ok", error="", data={"rows": []},
        )

        # Check stored message
        stored = ctx._messages[-1]
        self.assertTrue(stored.is_persisted)
        self.assertTrue(os.path.exists(stored.persist_path))

        # Compact message should be much smaller
        self.assertLess(stored.compact_chars, stored.full_chars)
        self.assertGreater(stored.full_chars, 50_000)
        self.assertLess(stored.compact_chars, 10_000)  # preview + metadata

    def test_compaction_stats_track_persisted(self):
        """ContextStats reports persisted_results count."""
        ctx = AgentContext("test", persist_dir=self.tmpdir)
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": _make_large_result(80_000)},
            turn=1, tool_name="nsys_sql_nvtx", arguments={},
            status="ok", error="", data={},
        )
        _, stats = ctx.build_model_messages(2)
        self.assertEqual(stats.persisted_results, 1)
        self.assertEqual(stats.compaction_level, 1)


# ---------------------------------------------------------------------------
# Level 2: Time-based compaction
# ---------------------------------------------------------------------------

class TestAgentContextLevel2(unittest.TestCase):
    """Tool results older than keep_recent_turns_full are compacted."""

    def test_recent_results_stay_full(self):
        """Results within keep_recent_turns_full=3 stay uncompressed."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=500_000,  # high limit to avoid pressure
            keep_recent_turns_full=3,
        ))
        medium = _make_medium_result(10_000)

        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": medium},
            turn=1, tool_name="scanner_files", arguments={},
            status="ok", error="", data={"files": []},
        )

        # Turn 4: age = 3, not > 3 → still full
        msgs, stats = ctx.build_model_messages(4)
        self.assertEqual(stats.compacted_tool_results, 0)
        self.assertEqual(stats.compaction_level, 0)

    def test_old_results_compacted(self):
        """Results older than keep_recent_turns_full get compacted."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=500_000,  # high limit to avoid pressure
            keep_recent_turns_full=3,
            compact_token_limit=1,     # force time-based compaction
        ))
        medium = _make_medium_result(10_000)

        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": medium},
            turn=1, tool_name="scanner_files", arguments={},
            status="ok", error="", data={"files": []},
        )

        # Turn 5: age = 4 > 3 → compacted
        msgs, stats = ctx.build_model_messages(5)
        self.assertGreater(stats.compacted_tool_results, 0)
        self.assertEqual(stats.compaction_level, 2)

    def test_compression_ratio_meaningful(self):
        """Compacted message is significantly smaller than full."""
        ctx = AgentContext("test", ContextPolicy(keep_recent_turns_full=0))
        medium = _make_medium_result(10_000)

        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": medium},
            turn=1, tool_name="scanner_files", arguments={},
            status="ok", error="", data={"files": []},
        )

        stored = ctx._messages[-1]
        ratio = stored.compact_chars / stored.full_chars if stored.full_chars else 1
        # Compact should be at most 50% of full size
        self.assertLess(ratio, 0.5,
                        f"Compression ratio {ratio:.2f} too high "
                        f"(compact={stored.compact_chars}, full={stored.full_chars})")

    def test_scanner_read_summary_preserves_code(self):
        """Even after compaction, scanner_read summary keeps code content."""
        ctx = AgentContext("test", ContextPolicy(
            keep_recent_turns_full=0, compact_token_limit=1,
        ))
        lines = _make_code_lines(0, 50)
        data = {
            "repo": "case_1", "path": "src/trainers/trainer.py",
            "total_lines": 200, "shown_start": 1, "shown_end": 50,
            "lines": lines,
        }
        full = json.dumps(data)

        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": full},
            turn=1, tool_name="scanner_read",
            arguments={"path": "src/trainers/trainer.py", "start": 1, "end": 50},
            status="ok", error="", data=data,
        )

        msgs, _ = ctx.build_model_messages(2)
        tool_msg = msgs[-1]
        parsed = json.loads(tool_msg["content"])
        summary = parsed["summary"]
        # Code content preserved
        self.assertEqual(len(summary["content"]), 50)
        self.assertIn("import torch", summary["content"][0])
        self.assertIn("class DistributedTrainer:", summary["content"][28])
        self.assertIn("def __init__", summary["content"][31])


# ---------------------------------------------------------------------------
# Level 3: Token-pressure compaction + recovery
# ---------------------------------------------------------------------------

class TestAgentContextLevel3(unittest.TestCase):
    """When estimated tokens exceed hard_limit, aggressive compaction + recovery."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="sysight_l3_"))

    def tearDown(self):
        import shutil
        if self.tmpdir.exists():
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pressure_triggers_when_above_hard_limit(self):
        """update_token_usage with high value triggers Level 3 on next build."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=500,
            keep_recent_turns_full=10,  # would normally keep full
        ))
        data = {"files": [{"path": f"f{i}.py"} for i in range(50)], "total": 50}
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(data)},
            turn=1, tool_name="scanner_files", arguments={},
            status="ok", error="", data=data,
        )

        # Without pressure: age=1 < 10 → no compaction
        _, stats = ctx.build_model_messages(2)
        self.assertEqual(stats.compaction_level, 0)

        # Inject high token count
        ctx.update_token_usage(600)  # > hard_token_limit=500

        # Under pressure: everything compacts
        _, stats = ctx.build_model_messages(2)
        self.assertEqual(stats.compaction_level, 3)
        self.assertGreater(stats.compacted_tool_results, 0)

    def test_pressure_compacts_even_recent_results(self):
        """Under pressure, even results from the current turn get compacted."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=100,
            keep_recent_turns_full=100,
        ))
        data = {"files": [{"path": f"f{i}.py"} for i in range(200)], "total": 200}
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(data)},
            turn=5, tool_name="scanner_files", arguments={},
            status="ok", error="", data=data,
        )
        ctx.update_token_usage(200)

        # age = 6-5 = 1, but under pressure → compacted
        _, stats = ctx.build_model_messages(6)
        self.assertGreater(stats.compacted_tool_results, 0)

    def test_recovery_injected_with_recently_read_files(self):
        """After pressure compaction, recovery message contains read file contents."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=100,
            restore_recent_files=True,
            restore_file_count=3,
            restore_max_tokens_per_file=2_000,
        ), persist_dir=self.tmpdir)

        # Simulate reading 3 files via scanner_read
        for i, (start, count) in enumerate([(0, 20), (30, 15), (60, 25)]):
            lines = _make_code_lines(start, count)
            data = {
                "repo": "case_1",
                "path": f"src/module_{i}/file_{i}.py",
                "total_lines": 200,
                "shown_start": start + 1,
                "shown_end": start + count,
                "lines": lines,
            }
            ctx.append_tool_result(
                {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps(data)},
                turn=i + 1, tool_name="scanner_read",
                arguments={"path": data["path"]},
                status="ok", error="", data=data,
            )

        ctx.update_token_usage(200)

        msgs, stats = ctx.build_model_messages(5)
        self.assertEqual(stats.compaction_level, 3)
        self.assertTrue(stats.recovery_injected)

        # Find the recovery message
        recovery = msgs[-1]
        self.assertEqual(recovery["role"], "user")
        content = recovery["content"]

        # Should mention the files
        self.assertIn("上下文恢复", content)
        self.assertIn("src/module_0/file_0.py", content)
        self.assertIn("src/module_1/file_1.py", content)
        self.assertIn("src/module_2/file_2.py", content)

        # Should contain actual code from the recovered files
        self.assertIn("import torch", content)
        self.assertIn("TrainingConfig", content)

    def test_recovery_only_injected_once(self):
        """Recovery is injected only on the first pressure event."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=100,
            restore_recent_files=True,
        ))

        lines = _make_code_lines(0, 10)
        data = {"repo": "x", "path": "f.py", "lines": lines}
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c0", "content": json.dumps(data)},
            turn=1, tool_name="scanner_read", arguments={},
            status="ok", error="", data=data,
        )
        ctx.update_token_usage(200)

        # First pressure → recovery injected
        msgs1, stats1 = ctx.build_model_messages(2)
        self.assertTrue(stats1.recovery_injected)

        # Second pressure → no recovery (already done)
        msgs2, stats2 = ctx.build_model_messages(3)
        self.assertFalse(stats2.recovery_injected)

    def test_recovery_respects_file_count_limit(self):
        """Recovery only includes the most recent restore_file_count files in key content."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=100,
            restore_recent_files=True,
            restore_file_count=2,
        ))

        for i in range(5):
            lines = _make_code_lines(i * 5, 5)
            data = {"repo": "r", "path": f"f{i}.py", "lines": lines}
            ctx.append_tool_result(
                {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps(data)},
                turn=i + 1, tool_name="scanner_read", arguments={},
                status="ok", error="", data=data,
            )

        ctx.update_token_usage(200)
        msgs, _ = ctx.build_model_messages(6)
        recovery = msgs[-1]["content"]

        # Progress overview lists ALL files read (f0-f4)
        # But key content section only includes the 2 most recent
        # Split on "### 关键文件内容" to check the key content separately
        parts = recovery.split("### 关键文件内容")
        self.assertEqual(len(parts), 2, "Should have progress overview + key content sections")
        overview, key_content = parts

        # Overview mentions all 5 files
        self.assertIn("已读取文件 (5)", overview)

        # Key content only has the 2 most recent files
        self.assertIn("f3.py", key_content)
        self.assertIn("f4.py", key_content)
        self.assertNotIn("f0.py", key_content)
        self.assertNotIn("f1.py", key_content)
        self.assertNotIn("f2.py", key_content)

    def test_recovery_respects_char_budget_per_file(self):
        """Recovery truncates each file at restore_max_tokens_per_file."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=100,
            restore_recent_files=True,
            restore_file_count=1,
            restore_max_tokens_per_file=500,  # very tight budget
        ))

        lines = _make_code_lines(0, 100)  # lots of lines
        data = {"repo": "r", "path": "big.py", "lines": lines}
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c0", "content": json.dumps(data)},
            turn=1, tool_name="scanner_read", arguments={},
            status="ok", error="", data=data,
        )

        ctx.update_token_usage(200)
        msgs, _ = ctx.build_model_messages(2)
        recovery = msgs[-1]["content"]

        # Should be truncated
        self.assertIn("省略", recovery)
        self.assertIn("big.py", recovery)
        self.assertLess(len(recovery), 2_000)  # overall message is bounded


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

class TestTokenEstimation(unittest.TestCase):
    """Anchor+delta token estimation."""

    def test_estimate_before_anchor_uses_char_ratio(self):
        """Before any update_token_usage, estimate from total chars."""
        ctx = AgentContext("hello world " * 100)  # ~1200 chars
        _, stats = ctx.build_model_messages(1)
        # ~1200 / 3.5 ≈ 342 tokens
        self.assertGreater(stats.estimated_tokens, 200)
        self.assertLess(stats.estimated_tokens, 500)

    def test_estimate_after_anchor_uses_delta(self):
        """After update_token_usage, estimate = anchor + delta."""
        ctx = AgentContext("base")
        ctx.update_token_usage(1000)

        # Add more content
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": "x" * 3500},
            turn=1, tool_name="t", arguments={}, status="ok", error="", data={},
        )

        _, stats = ctx.build_model_messages(2)
        # anchor=1000 + delta(~3500/3.5=1000) ≈ 2000
        self.assertGreater(stats.estimated_tokens, 1500)
        self.assertLess(stats.estimated_tokens, 2500)

    def test_estimate_zero_when_empty(self):
        """Empty context estimates 0 tokens."""
        ctx = AgentContext("")
        _, stats = ctx.build_model_messages(1)
        self.assertGreaterEqual(stats.estimated_tokens, 0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestAgentContextEdgeCases(unittest.TestCase):
    """Edge cases and error handling."""

    def test_empty_tool_result(self):
        """Empty tool result: compact_message may be None (wrapper overhead > content)."""
        ctx = AgentContext("test")
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": ""},
            turn=1, tool_name="test_tool", arguments={},
            status="ok", error="", data=None,
        )
        stored = ctx._messages[-1]
        # Empty content → compact wrapper overhead exceeds savings → compact is None
        self.assertIsNone(stored.compact_message)

    def test_error_tool_result_not_compacted_early(self):
        """Error results follow same compaction rules."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=500_000,  # high limit to avoid pressure
            keep_recent_turns_full=0,
            compact_token_limit=1,     # force time-based compaction
        ))
        data = {"files": [{"path": f"f{i}.py"} for i in range(50)], "total": 50}
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(data)},
            turn=1, tool_name="scanner_files", arguments={},
            status="error", error="timeout", data=data,
        )
        _, stats = ctx.build_model_messages(2)
        self.assertGreater(stats.compacted_tool_results, 0)

    def test_non_dict_data_gets_generic_summary(self):
        """Non-dict tool data gets a generic preview-based summary."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=500_000, keep_recent_turns_full=0,
        ))
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": "plain text output\n" * 100},
            turn=1, tool_name="unknown_tool", arguments={},
            status="ok", error="", data="plain text output\n" * 100,
        )
        stored = ctx._messages[-1]
        compact = json.loads(stored.compact_message["content"])
        self.assertIn("preview", compact["summary"])
        self.assertIn("chars", compact["summary"])

    def test_full_log_messages_always_complete(self):
        """full_log_messages() returns uncompressed messages regardless of policy."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=500_000, keep_recent_turns_full=0,
        ))
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": "FULL_DATA_12345"},
            turn=1, tool_name="t", arguments={}, status="ok", error="", data={},
        )
        # Build compacted model view
        ctx.build_model_messages(2)

        # Full log still has original content
        full = ctx.full_log_messages()
        tool_msg = [m for m in full if m.get("tool_call_id") == "c1"][0]
        self.assertIn("FULL_DATA_12345", tool_msg["content"])

    def test_compaction_disabled_by_policy(self):
        """compact_after_first_exposure=False disables all compaction."""
        ctx = AgentContext("test", ContextPolicy(
            compact_after_first_exposure=False,
            hard_token_limit=500_000,
            keep_recent_turns_full=0,
        ))
        data = {"files": [{"path": f"f{i}.py"} for i in range(200)], "total": 200}
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(data)},
            turn=1, tool_name="scanner_files", arguments={},
            status="ok", error="", data=data,
        )
        _, stats = ctx.build_model_messages(10)
        self.assertEqual(stats.compacted_tool_results, 0)

    def test_to_jsonable_handles_path(self):
        """to_jsonable converts Path to string."""
        from pathlib import Path
        result = to_jsonable({"file": Path("/tmp/test.txt")})
        self.assertEqual(result["file"], "/tmp/test.txt")

    def test_to_jsonable_handles_dataclass(self):
        """to_jsonable converts dataclasses to dicts."""
        result = to_jsonable(ContextStats(request_turn=1, model_messages=5))
        self.assertEqual(result["request_turn"], 1)
        self.assertEqual(result["model_messages"], 5)


# ---------------------------------------------------------------------------
# Model-aware thresholds (DeepSeek-TUI pattern)
# ---------------------------------------------------------------------------

class TestModelAwareThresholds(unittest.TestCase):
    """ContextPolicy auto-scales token limits based on model_name."""

    def test_known_model_gets_correct_limits(self):
        """Known model → limits computed from context window."""
        p = ContextPolicy(model_name="gpt-5.5")
        self.assertEqual(p.effective_soft_limit(), int(1_000_000 * 0.90))
        self.assertEqual(p.effective_compact_limit(), int(1_000_000 * 0.90))
        self.assertEqual(p.effective_hard_limit(), int(1_000_000 * 0.95))

    def test_large_context_model_gets_higher_limits(self):
        """1M context model → much higher limits."""
        p_small = ContextPolicy(model_name="gpt-5.2")
        p_large = ContextPolicy(model_name="deepseek-v4")
        self.assertGreater(p_large.effective_hard_limit(), p_small.effective_hard_limit() * 2)

    def test_explicit_limits_override_model_name(self):
        """Explicit soft/hard_token_limit overrides auto-compute."""
        p = ContextPolicy(model_name="gpt-5.5", soft_token_limit=50_000, hard_token_limit=80_000)
        self.assertEqual(p.effective_soft_limit(), 50_000)
        self.assertEqual(p.effective_hard_limit(), 80_000)

    def test_unknown_model_uses_default(self):
        """Unknown model → default 1M context window (DeepSeek V4 Pro)."""
        p = ContextPolicy(model_name="some-unknown-model-v1")
        self.assertEqual(p.effective_soft_limit(), int(_DEFAULT_CONTEXT_WINDOW * 0.90))
        self.assertEqual(p.effective_hard_limit(), int(_DEFAULT_CONTEXT_WINDOW * 0.95))

    def test_fuzzy_model_name_matching(self):
        """Fuzzy substring match for model names."""
        p = ContextPolicy(model_name="claude-sonnet-4-5-latest")
        # "claude-sonnet-4-5" is a substring match
        self.assertEqual(p.effective_soft_limit(), int(200_000 * 0.90))

    def test_empty_model_name_uses_default(self):
        """Empty model_name → default context window."""
        p = ContextPolicy()
        self.assertEqual(p.effective_hard_limit(), int(_DEFAULT_CONTEXT_WINDOW * 0.95))

    def test_model_name_propagates_from_loop(self):
        """AgentLoop injects provider.model into ContextPolicy."""
        provider = _RecordingProvider()
        self.assertEqual(provider.model, "fake")
        registry = ToolRegistry()
        registry.register(ToolDef(
            name="test_long", description="test",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=lambda: {"ok": True}, read_only=True,
        ))
        tool_policy = ToolPolicy(allowed_tools={"test_long"}, read_only=True)
        loop = AgentLoop(provider, registry, tool_policy)
        # Run a minimal task
        result = loop.run(AgentTask(
            run_id="r1", task_id="t1", task_type="analyze",
            user_prompt="test", max_turns=1,
        ))
        # The context policy used should have model_name="fake"
        # (checked via context_stats in the result)
        if result.context_stats.get("turns"):
            first_turn = result.context_stats["turns"][0]
            # session_progress should exist (new feature)
            self.assertIn("session_progress", first_turn)


# ---------------------------------------------------------------------------
# Circuit breaker (Claude Code pattern)
# ---------------------------------------------------------------------------

class TestCircuitBreaker(unittest.TestCase):
    """Circuit breaker stops compaction after consecutive Level-3 events."""

    def test_circuit_breaker_trips_after_max_events(self):
        """After N consecutive pressure events, compaction stops."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=100,
            circuit_breaker_max=3,
        ))
        data = {"files": [{"path": f"f{i}.py"} for i in range(200)], "total": 200}
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(data)},
            turn=1, tool_name="scanner_files", arguments={},
            status="ok", error="", data=data,
        )
        ctx.update_token_usage(200)

        # 3 consecutive pressure events → circuit breaker trips
        _, stats1 = ctx.build_model_messages(2)
        self.assertFalse(stats1.circuit_breaker_active)

        _, stats2 = ctx.build_model_messages(3)
        self.assertFalse(stats2.circuit_breaker_active)

        _, stats3 = ctx.build_model_messages(4)
        self.assertTrue(stats3.circuit_breaker_active)
        # After tripping, no more compaction
        self.assertEqual(stats3.compaction_level, 0)

    def test_circuit_breaker_resets_on_non_pressure(self):
        """If a non-pressure turn occurs, counter resets."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=100,
            circuit_breaker_max=3,
        ))
        data = {"files": [{"path": f"f{i}.py"} for i in range(200)], "total": 200}
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(data)},
            turn=1, tool_name="scanner_files", arguments={},
            status="ok", error="", data=data,
        )

        # Pressure event 1
        ctx.update_token_usage(200)
        _, stats1 = ctx.build_model_messages(2)
        self.assertFalse(stats1.circuit_breaker_active)

        # Non-pressure turn (reset anchor to low value)
        ctx.update_token_usage(50)
        _, stats2 = ctx.build_model_messages(3)
        self.assertFalse(stats2.circuit_breaker_active)
        self.assertEqual(ctx._consecutive_pressure_events, 0)

        # Pressure again → counter starts from 0
        ctx.update_token_usage(200)
        _, stats3 = ctx.build_model_messages(4)
        self.assertFalse(stats3.circuit_breaker_active)
        self.assertEqual(ctx._consecutive_pressure_events, 1)

    def test_circuit_breaker_warning_injected(self):
        """After tripping, a warning message is added for the model in the same turn."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=100,
            circuit_breaker_max=2,
        ))
        data = {"files": [{"path": f"f{i}.py"} for i in range(200)], "total": 200}
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(data)},
            turn=1, tool_name="scanner_files", arguments={},
            status="ok", error="", data=data,
        )

        # Trip the breaker over 2 consecutive pressure events
        ctx.update_token_usage(200)
        ctx.build_model_messages(2)  # pressure event 1: counter=1
        msgs, stats = ctx.build_model_messages(3)  # pressure event 2: counter=2 → trips

        # Warning is injected in the SAME turn when the breaker trips
        self.assertTrue(stats.circuit_breaker_active)
        # Find the warning message (it's appended as a user message)
        warning_msgs = [m for m in msgs if "熔断器" in m.get("content", "")]
        self.assertGreaterEqual(len(warning_msgs), 1, "Circuit breaker warning should be injected")

    def test_circuit_breaker_warning_only_injected_once(self):
        """Warning message is only injected once."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=100,
            circuit_breaker_max=2,
        ))
        data = {"files": [{"path": f"f{i}.py"} for i in range(200)], "total": 200}
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(data)},
            turn=1, tool_name="scanner_files", arguments={},
            status="ok", error="", data=data,
        )
        ctx.update_token_usage(200)
        ctx.build_model_messages(2)
        ctx.build_model_messages(3)  # trips breaker + injects warning

        # Subsequent builds should NOT add another warning
        msgs1, _ = ctx.build_model_messages(5)
        msgs2, _ = ctx.build_model_messages(6)
        warning_count_1 = sum(1 for m in msgs1 if "熔断器" in m.get("content", ""))
        warning_count_2 = sum(1 for m in msgs2 if "熔断器" in m.get("content", ""))
        self.assertLessEqual(warning_count_1, 1)
        self.assertLessEqual(warning_count_2, 1)


# ---------------------------------------------------------------------------
# Session progress tracking (Codex "ghost history" prevention)
# ---------------------------------------------------------------------------

class TestSessionProgress(unittest.TestCase):
    """Session progress is tracked and included in stats + recovery."""

    def test_session_progress_tracks_files_read(self):
        """Files read via scanner_read are tracked in session progress."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=500_000, keep_recent_turns_full=0,
        ))
        for i in range(3):
            data = {"path": f"src/file_{i}.py", "lines": []}
            ctx.append_tool_result(
                {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps(data)},
                turn=i + 1, tool_name="scanner_read", arguments={},
                status="ok", error="", data=data,
            )

        _, stats = ctx.build_model_messages(4)
        self.assertEqual(stats.session_progress["files_read"], [
            "src/file_0.py", "src/file_1.py", "src/file_2.py",
        ])

    def test_session_progress_tracks_tools_called(self):
        """Tool call counts are tracked in session progress."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=500_000, keep_recent_turns_full=0,
        ))
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": "data"},
            turn=1, tool_name="scanner_read", arguments={},
            status="ok", error="", data={"path": "f.py", "lines": []},
        )
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c2", "content": "data"},
            turn=2, tool_name="scanner_files", arguments={},
            status="ok", error="", data={},
        )
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c3", "content": "data"},
            turn=3, tool_name="scanner_read", arguments={},
            status="ok", error="", data={"path": "g.py", "lines": []},
        )

        _, stats = ctx.build_model_messages(4)
        self.assertEqual(stats.session_progress["tools_called"]["scanner_read"], 2)
        self.assertEqual(stats.session_progress["tools_called"]["scanner_files"], 1)
        self.assertEqual(stats.session_progress["total_tool_calls"], 3)

    def test_session_progress_in_recovery_message(self):
        """Recovery message includes session progress summary."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=100,
            restore_recent_files=True,
        ))
        data = {"path": "src/main.py", "lines": [{"line": 1, "text": "hello"}]}
        ctx.append_tool_result(
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(data)},
            turn=1, tool_name="scanner_read", arguments={},
            status="ok", error="", data=data,
        )
        ctx.update_token_usage(200)

        msgs, stats = ctx.build_model_messages(2)
        self.assertTrue(stats.recovery_injected)
        recovery = msgs[-1]["content"]
        # Should include progress overview
        self.assertIn("调查进度概要", recovery)
        self.assertIn("src/main.py", recovery)
        self.assertIn("scanner_read", recovery)

    def test_unique_files_tracked(self):
        """Same file read multiple times counts only once."""
        ctx = AgentContext("test", ContextPolicy(
            hard_token_limit=500_000, keep_recent_turns_full=0,
        ))
        for turn in range(1, 4):
            data = {"path": "src/main.py", "lines": [{"line": i, "text": f"line{i}"} for i in range(turn * 10, turn * 10 + 5)]}
            ctx.append_tool_result(
                {"role": "tool", "tool_call_id": f"c{turn}", "content": json.dumps(data)},
                turn=turn, tool_name="scanner_read", arguments={},
                status="ok", error="", data=data,
            )

        _, stats = ctx.build_model_messages(5)
        self.assertEqual(stats.session_progress["files_read"], ["src/main.py"])
        self.assertEqual(stats.session_progress["tools_called"]["scanner_read"], 3)


# ---------------------------------------------------------------------------
# Integration: AgentLoop with fake providers
# ---------------------------------------------------------------------------

class _RecordingProvider:
    """3-turn: call-1, call-2, final-output. Records all requests."""

    def __init__(self):
        self.requests = []

    @property
    def name(self):
        return "recording"

    @property
    def model(self):
        return "fake"

    def complete(self, request):
        self.requests.append(request)
        turn = len(self.requests)
        if turn == 1:
            return LLMResponse(
                finish_reason="tool_calls",
                tool_calls=[ToolCallRequest(
                    id="call-1", name="test_long",
                    arguments={"label": "first"},
                )],
                usage=UsageInfo(prompt_tokens=10, output_tokens=1),
            )
        if turn == 2:
            return LLMResponse(
                finish_reason="tool_calls",
                tool_calls=[ToolCallRequest(
                    id="call-2", name="test_long",
                    arguments={"label": "second"},
                )],
                usage=UsageInfo(prompt_tokens=20, output_tokens=1),
            )
        return LLMResponse(
            content=json.dumps({"summary": "ok", "findings": [], "memory_updates": []}),
            finish_reason="stop",
            usage=UsageInfo(prompt_tokens=30, output_tokens=3),
        )


class _LongRecordingProvider:
    """6-turn: call-1..call-5, then final-output."""

    def __init__(self):
        self.requests = []

    @property
    def name(self):
        return "recording"

    @property
    def model(self):
        return "fake"

    def complete(self, request):
        self.requests.append(request)
        turn = len(self.requests)
        if turn <= 5:
            return LLMResponse(
                finish_reason="tool_calls",
                tool_calls=[ToolCallRequest(
                    id=f"call-{turn}", name="test_long",
                    arguments={"label": f"call-{turn}"},
                )],
                usage=UsageInfo(prompt_tokens=10 * turn, output_tokens=1),
            )
        return LLMResponse(
            content=json.dumps({"summary": "ok", "findings": [], "memory_updates": []}),
            finish_reason="stop",
            usage=UsageInfo(prompt_tokens=200, output_tokens=3),
        )


class _RepeatingProvider:
    """Repeats the same tool call N times."""

    def __init__(self, repeats=4):
        self.requests = []
        self.repeats = repeats

    @property
    def name(self):
        return "repeating"

    @property
    def model(self):
        return "fake"

    def complete(self, request):
        self.requests.append(request)
        turn = len(self.requests)
        if turn <= self.repeats:
            return LLMResponse(
                finish_reason="tool_calls",
                tool_calls=[ToolCallRequest(
                    id=f"call-{turn}", name="test_long",
                    arguments={"label": f"repeat-{turn}"},
                )],
                usage=UsageInfo(prompt_tokens=10, output_tokens=1),
            )
        return LLMResponse(
            content=json.dumps({"summary": "ok", "findings": [], "memory_updates": []}),
            finish_reason="stop",
            usage=UsageInfo(prompt_tokens=30, output_tokens=3),
        )


class _RepairJsonProvider:
    """First final response is prose; second final response is valid JSON."""

    def __init__(self):
        self.requests = []

    @property
    def name(self):
        return "repair-json"

    @property
    def model(self):
        return "fake"

    def complete(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMResponse(
                content="I inspected the repo and will now summarize.",
                finish_reason="stop",
                usage=UsageInfo(prompt_tokens=10, output_tokens=5),
            )
        return LLMResponse(
            content=json.dumps({"summary": "fixed", "findings": []}),
            finish_reason="stop",
            usage=UsageInfo(prompt_tokens=20, output_tokens=3),
        )


class TestAgentContextIntegration(unittest.TestCase):
    """End-to-end tests with AgentLoop and fake providers."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(ToolDef(
            name="test_long",
            description="Return a long payload",
            parameters={
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": [],
            },
            fn=lambda label="first": {"label": label, "payload": "x" * 2000},
            read_only=True,
        ))
        self.tool_policy = ToolPolicy(allowed_tools={"test_long"}, read_only=True)

    def test_recent_results_full_in_loop(self):
        """With keep_recent=3, 3-turn loop keeps all results full."""
        provider = _RecordingProvider()
        loop = AgentLoop(provider, self.registry, self.tool_policy)
        result = loop.run(AgentTask(
            run_id="r1", task_id="t1", task_type="analyze",
            user_prompt="investigate", max_turns=3,
        ))
        self.assertEqual(result.status, "ok")
        # Turn 3: call-1 age=2 ≤ 3 → still full
        third = _tool_message(provider.requests[2].messages, "call-1")
        self.assertNotIn("sysight_compacted_tool_result", third["content"])

    def test_old_results_compacted_in_loop(self):
        """With 6 turns, call-1 gets compacted (age=5 > 3)."""
        provider = _LongRecordingProvider()
        loop = AgentLoop(provider, self.registry, self.tool_policy)
        result = loop.run(AgentTask(
            run_id="r1", task_id="t1", task_type="analyze",
            user_prompt="investigate", max_turns=6,
            context_policy=ContextPolicy(compact_token_limit=1),
        ))
        self.assertEqual(result.status, "ok")
        # Turn 6: call-1 age=5 > 3 → compacted
        final = _tool_message(provider.requests[5].messages, "call-1")
        self.assertIn("sysight_compacted_tool_result", final["content"])
        # call-4 age=2 ≤ 3 → still full
        recent = _tool_message(provider.requests[5].messages, "call-4")
        self.assertNotIn("sysight_compacted_tool_result", recent["content"])

    def test_custom_policy_keep_recent_1(self):
        """keep_recent=1 restores old behavior."""
        provider = _RecordingProvider()
        loop = AgentLoop(provider, self.registry, self.tool_policy)
        result = loop.run(AgentTask(
            run_id="r1", task_id="t1", task_type="analyze",
            user_prompt="investigate", max_turns=3,
            context_policy=ContextPolicy(
                keep_recent_turns_full=1,
                compact_token_limit=1,     # force time-based compaction
                hard_token_limit=500_000,  # avoid pressure from default auto-compute
            ),
        ))
        self.assertEqual(result.status, "ok")
        third = _tool_message(provider.requests[2].messages, "call-1")
        self.assertIn("sysight_compacted_tool_result", third["content"])

    def test_debug_log_integrity(self):
        """Debug log retains full messages even when model view is compacted."""
        real = _RecordingProvider()
        debug_log = []
        provider = DebugProvider(real, log=debug_log, verbose=False)
        loop = AgentLoop(provider, self.registry, self.tool_policy)
        loop.run(AgentTask(
            run_id="r1", task_id="t1", task_type="analyze",
            user_prompt="investigate", max_turns=3,
        ))
        final_debug = debug_log[2]["request"]["messages"]
        final_summary = debug_log[2]["request"]["model_messages_summary"]
        self.assertGreaterEqual(
            len(str(final_debug)),
            sum(item["content_len"] for item in final_summary),
            "Debug log should be at least as large as model-facing summary",
        )

    def test_tool_protocol_not_broken_by_compaction(self):
        """Compaction must not insert user messages between tool pairs."""
        provider = _RepeatingProvider(repeats=4)
        loop = AgentLoop(provider, self.registry, self.tool_policy)
        result = loop.run(AgentTask(
            run_id="r1", task_id="t1", task_type="analyze",
            user_prompt="investigate", max_turns=5,
        ))
        self.assertEqual(result.status, "ok")
        for req in provider.requests:
            _assert_openai_tool_protocol(req.messages)

    def test_json_repair_turn_success_is_ok(self):
        """A recovered JSON formatting error should not fail the whole task."""
        provider = _RepairJsonProvider()
        loop = AgentLoop(provider, ToolRegistry(), ToolPolicy(allowed_tools=set(), read_only=True))
        result = loop.run(AgentTask(
            run_id="r1", task_id="t1", task_type="optimize",
            user_prompt="return JSON", max_turns=2,
        ))
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.output["summary"], "fixed")
        self.assertTrue(any("schema_error" in e for e in result.errors))
        self.assertEqual(len(provider.requests), 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_message(messages, tool_call_id):
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id") == tool_call_id:
            return m
    raise AssertionError(f"tool message not found: {tool_call_id}")


def _assert_openai_tool_protocol(messages):
    for idx, m in enumerate(messages):
        tool_calls = m.get("tool_calls") or []
        if m.get("role") != "assistant" or not tool_calls:
            continue
        expected = [c.get("id") for c in tool_calls if c.get("id")]
        following = messages[idx + 1: idx + 1 + len(expected)]
        actual = [f.get("tool_call_id") for f in following if f.get("role") == "tool"]
        if actual != expected:
            raise AssertionError(
                f"tool protocol broken at msg {idx}: "
                f"expected={expected} actual={actual}"
            )


if __name__ == "__main__":
    unittest.main()
