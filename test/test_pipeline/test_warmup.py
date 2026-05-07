"""Tests for deterministic warmup repository orientation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sysight.pipeline.warmup import run_warmup


class MemoryStub:
    def __init__(self, root: Path):
        self.root = root
        self.last_content = ""
        self.last_path = ""

    def workspace_namespace(self, repo_root: str | None = None, namespace: str | None = None) -> str:
        if namespace:
            return namespace
        return Path(repo_root or "default").name

    def write_page(self, path: str, content: str, **kwargs) -> Path:
        self.last_path = path
        self.last_content = content
        target = self.root / "wiki" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target


class TestWarmupDeterministic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.memory = MemoryStub(self.root / ".sysight" / "memory")

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, rel: str, text: str = "") -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_nsys_case_uses_case_yaml_and_ignores_dependency_pytest(self):
        self._write("case.yaml", """\
id: case_1
entrypoint: run.py
requires:
  gpu: true
  min_gpus: 1
  nsys: true
profile:
  sqlite: profiles/case_1.sqlite
  nsys_rep: profiles/case_1.nsys-rep
run:
  command: python run.py --config configs/config_v2.yaml
  active_config: configs/config_v2.yaml
  active_model_variant: v5
""")
        self._write("run.py", "from src.runtime import main\n\nif __name__ == '__main__':\n    main()\n")
        self._write("src/runtime/__init__.py", "from .launcher import main\n")
        self._write("src/runtime/launcher.py", "from src.config import load\n\ndef main():\n    load()\n")
        self._write("src/config.py", "def load():\n    return {}\n")
        self._write("configs/config_v1.yaml", "train:\n  batch_size: 4\n")
        self._write("configs/config_v2.yaml", "train:\n  batch_size: 8\n  compile:\n    use_compile: true\n")
        self._write("scripts/start.sh", "python run.py --config configs/config_v2.yaml\n")
        self._write(
            "scripts/profile_local.sh",
            "nsys profile --trace=cuda -o profiles/case_1 python run.py --config configs/config_v2.yaml\n",
        )
        self._write("profiles/case_1.sqlite", "")
        self._write("external/pkg/site-packages/dep/tests/test_dep.py", "import pytest\n")

        result = run_warmup(str(self.root), None, self.memory)

        self.assertEqual(result.repo_setup.source, "warmup_verified")
        self.assertEqual(result.repo_setup.entry_point, "python run.py --config configs/config_v2.yaml")
        self.assertEqual(result.summary["active_config"], "configs/config_v2.yaml")
        self.assertEqual(result.summary["active_variants"]["model"], "v5")
        self.assertEqual(result.summary["profile_sqlite"], "profiles/case_1.sqlite")
        self.assertGreaterEqual(result.summary["hot_path_count"], 3)
        self.assertIn("external", result.summary["ignored_counts"])
        self.assertEqual(result.repo_setup.test_commands, [])
        self.assertIn("configs/config_v1.yaml", self.memory.last_content)
        self.assertNotIn("test_dep.py", self.memory.last_content)

    def test_bazel_runfiles_repo_uses_job_script_and_redacts_secrets(self):
        self._write("mlp.hope", """\
[roles]
workers = 1
worker.vcore = 160
worker.memory = 819200
worker.gcores80g = 8
worker.script = bash start.sh

[others]
afo.container.password = abc123

[application]
appName = TrainJob
""")
        self._write("start.sh", """\
export SECRET_KEY='abc123'
cd trainer_package.tar.gz
cd trainer.runfiles/autocar
(python3.11 -m torch.distributed.launch --nproc_per_node=2 common/python/train/trainer --cfg_file common/python/train/config/config.yaml) && STATE=0 || STATE=1
""")
        self._write("trainer", "main_rel_path = 'autocar/common/python/train/trainer.py'\n")
        self._write(
            "trainer.runfiles/autocar/common/python/train/trainer.py",
            "from common.python.train.model import Model\n\nif __name__ == '__main__':\n    Model()\n",
        )
        self._write("trainer.runfiles/autocar/common/python/train/model.py", "class Model:\n    pass\n")
        self._write("trainer.runfiles/autocar/common/python/train/config/config.yaml", "train:\n  batch_size_per_gpu: 184\n  use_compile: true\n")
        self._write("trainer.runfiles/python311_x86_64_deps_tensorboard/site-packages/tensorboard/__init__.py", "")

        result = run_warmup(str(self.root), None, self.memory)

        self.assertEqual(result.repo_setup.source, "warmup_verified")
        self.assertEqual(result.repo_setup.entry_point, "bash start.sh")
        self.assertIn("torch.distributed.launch", " ".join(result.repo_setup.minimal_run))
        self.assertEqual(
            result.summary["active_config"],
            "trainer.runfiles/autocar/common/python/train/config/config.yaml",
        )
        self.assertIn("runfiles-deps", result.summary["ignored_counts"])
        self.assertNotIn("tensorboard", "\n".join(result.summary.get("warnings", [])))
        self.assertNotIn("tensorboard/__init__.py", self.memory.last_content)
        self.assertNotIn("abc123", self.memory.last_content)
        self.assertIn("检测到分布式启动", result.repo_setup.constraints)
        self.assertIn("作业 GPU: 8", result.repo_setup.constraints)

    def test_generic_run_py_fallback(self):
        self._write("run.py", "if __name__ == '__main__':\n    print('hi')\n")

        result = run_warmup(str(self.root), None, self.memory)

        self.assertEqual(result.repo_setup.entry_point, "python run.py")
        self.assertEqual(result.repo_setup.source, "warmup_verified")
        self.assertEqual(result.summary["hot_path_count"], 1)

    def test_missing_config_keeps_source_partial(self):
        self._write("run.py", "if __name__ == '__main__':\n    pass\n")
        self._write("scripts/start.sh", "python run.py --config configs/missing.yaml\n")

        result = run_warmup(str(self.root), None, self.memory)

        self.assertEqual(result.repo_setup.source, "warmup_partial")
        self.assertIn("active config 无法验证", result.warnings)


if __name__ == "__main__":
    unittest.main()
