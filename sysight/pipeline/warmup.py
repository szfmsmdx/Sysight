"""WARMUP stage — deterministic repository orientation.

Runs once per repo. Discovers the developer run command, active config,
important source paths, and dependency noise to skip. The result is persisted
as a compact workspace overview for later pipeline stages.
"""

from __future__ import annotations

import ast
import configparser
import re
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sysight.types.repo_setup import RepoSetup


_CONFIG_EXTS = {
    ".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".conf", ".txt", ".pbtxt"
}
_PROFILE_EXTS = {".sqlite", ".nsys-rep"}
_SCRIPT_EXTS = {".sh", ".bash", ".zsh"}
_TEST_FILE_RE = re.compile(r"(^|/)(tests?/|test_[^/]+\.py$|[^/]+_test\.py$)")
_ASSIGNMENT_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_.-]*(?:secret|token|password|passwd|api[_-]?key|auth|credential)[A-Za-z0-9_.-]*)"
    r"(?P<sep>\s*[=:]\s*)"
    r"(?P<value>'[^']*'|\"[^\"]*\"|[^\s#]+)",
    re.I,
)
_ENTRY_FILES = {"run.py", "main.py", "train.py", "__main__.py"}
_PYTHON_COMMAND_RE = re.compile(r"(^|[/\"'])(python[0-9.]*|torchrun)$")
_SKIP_DIR_NAMES = {
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox",
    ".eggs", "dist", "build", ".ipynb_checkpoints",
}


@dataclass
class WarmupResult:
    repo: str = ""
    repo_setup: RepoSetup = field(default_factory=RepoSetup)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    overview_path: str = ""
    summary: dict = field(default_factory=dict)


@dataclass
class ScriptCommand:
    command: str
    script: str
    line: int
    cwd: str = ""


@dataclass
class Inventory:
    app_py_files: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    script_files: list[str] = field(default_factory=list)
    profile_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    ignored_counts: dict[str, int] = field(default_factory=dict)
    ignored_examples: dict[str, list[str]] = field(default_factory=dict)
    total_files_seen: int = 0


@dataclass
class WarmupFacts:
    case_info: dict = field(default_factory=dict)
    job_info: dict = field(default_factory=dict)
    script_commands: list[ScriptCommand] = field(default_factory=list)
    primary_command: str = ""
    primary_source: str = ""
    primary_script: str = ""
    profile_command: str = ""
    profile_sqlite: str = ""
    active_config: str = ""
    active_config_exists: bool = False
    active_config_source: str = ""
    active_variants: dict = field(default_factory=dict)
    app_roots: list[str] = field(default_factory=list)
    entry_files: list[str] = field(default_factory=list)
    hot_path_files: list[str] = field(default_factory=list)
    import_tree: dict[str, list[str]] = field(default_factory=dict)
    inactive_configs: list[str] = field(default_factory=list)
    perf_knobs: list[str] = field(default_factory=list)
    workload: str = ""
    constraints: list[str] = field(default_factory=list)


def run_warmup(
    repo: str,
    registry,
    provider,
    knowledge=None,
) -> WarmupResult:
    """Explore a repo and produce developer-facing warmup facts."""
    del registry, provider  # warmup is deterministic in this iteration.

    root = Path(repo).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    repo_setup = RepoSetup(source="warmup_partial")

    if not root.is_dir():
        errors.append(f"repo 不是目录: {root}")
        return WarmupResult(repo=str(root), repo_setup=repo_setup, errors=errors)

    inventory = _build_inventory(root)
    facts = _collect_facts(root, inventory, warnings)
    _populate_repo_setup(root, repo_setup, facts, warnings, errors)

    overview_path = ""
    if knowledge:
        ns = knowledge.workspace_namespace(repo_root=str(root))
        overview = _build_overview(root, inventory, facts, repo_setup, warnings, errors)
        try:
            written = knowledge.write_page(
                f"workspaces/{ns}/overview.md",
                overview,
                title="Repo Overview",
                scope="workspace",
            )
            overview_path = str(written)
        except Exception as e:
            errors.append(f"wiki write failed: {e}")
            repo_setup.source = "warmup_partial"

    summary = _build_summary(root, inventory, facts, repo_setup, warnings, errors, overview_path)
    return WarmupResult(
        repo=str(root),
        repo_setup=repo_setup,
        errors=errors,
        warnings=warnings,
        overview_path=overview_path,
        summary=summary,
    )


# Inventory

def _build_inventory(root: Path) -> Inventory:
    inventory = Inventory()
    stack = [root]

    while stack:
        cur = stack.pop()
        try:
            items = sorted(cur.iterdir(), key=lambda p: p.name)
        except (OSError, PermissionError):
            continue

        for item in items:
            rel = item.relative_to(root)
            rel_str = rel.as_posix()

            if item.is_dir():
                reason = _skip_dir_reason(rel_str, item.name)
                if reason:
                    _record_ignored(inventory, reason, rel_str)
                    continue
                stack.append(item)
                continue

            if not item.is_file():
                continue

            inventory.total_files_seen += 1
            suffix = item.suffix.lower()
            if suffix == ".py":
                inventory.app_py_files.append(rel_str)
                if _TEST_FILE_RE.search(rel_str):
                    inventory.test_files.append(rel_str)
            elif suffix in _CONFIG_EXTS:
                inventory.config_files.append(rel_str)
            elif suffix in _SCRIPT_EXTS or item.name == "test.sh":
                inventory.script_files.append(rel_str)
            elif item.name.endswith(".hope") or item.name == ".hopemeta":
                inventory.config_files.append(rel_str)
            elif suffix in _PROFILE_EXTS:
                inventory.profile_files.append(rel_str)

    inventory.app_py_files.sort()
    inventory.config_files.sort()
    inventory.script_files.sort()
    inventory.profile_files.sort()
    inventory.test_files.sort()
    return inventory


def _skip_dir_reason(rel_str: str, name: str) -> str:
    lower = name.lower()
    parts = rel_str.split("/")

    if name in _SKIP_DIR_NAMES:
        return name
    if lower in {"external", "third_party"}:
        return lower
    if lower == "site-packages":
        return "site-packages"
    if lower.endswith((".dist-info", ".egg-info")):
        return "package-metadata"
    if len(parts) >= 2 and parts[-2].endswith(".runfiles"):
        if re.match(r"python.*_deps", lower):
            return "runfiles-deps"

    if re.match(r"python.*_deps", lower):
        return "python-deps"

    return ""


def _record_ignored(inventory: Inventory, reason: str, rel_str: str) -> None:
    inventory.ignored_counts[reason] = inventory.ignored_counts.get(reason, 0) + 1
    examples = inventory.ignored_examples.setdefault(reason, [])
    if len(examples) < 5:
        examples.append(rel_str)


# Fact collection

def _collect_facts(root: Path, inventory: Inventory, warnings: list[str]) -> WarmupFacts:
    facts = WarmupFacts()
    facts.case_info = _parse_case_yaml(root, warnings)
    facts.job_info = _parse_job_configs(root, inventory, warnings)
    facts.script_commands = _extract_script_commands(root, inventory.script_files)

    _select_commands(inventory, facts, warnings)
    _resolve_active_config(root, inventory, facts, warnings)
    facts.active_variants = facts.case_info.get("active_variants", {})
    facts.profile_sqlite = _select_profile_sqlite(root, inventory, facts)
    facts.app_roots = _infer_app_roots(inventory.app_py_files)
    facts.entry_files = _infer_entry_files(root, inventory, facts)
    facts.hot_path_files, facts.import_tree = _trace_import_tree(root, inventory.app_py_files, facts.entry_files)
    facts.inactive_configs = _find_inactive_configs(root, inventory, facts.active_config)
    facts.perf_knobs = _extract_perf_knobs(root, facts.active_config)
    facts.workload = _infer_workload(root, inventory, facts)
    facts.constraints = _infer_constraints(facts)
    return facts


def _parse_case_yaml(root: Path, warnings: list[str]) -> dict:
    case_yaml = root / "case.yaml"
    if not case_yaml.exists():
        return {}

    try:
        import yaml
        data = yaml.safe_load(case_yaml.read_text(encoding="utf-8")) or {}
    except ModuleNotFoundError:
        from sysight.agent.config_loader import _parse_yaml_simple
        data = _parse_yaml_simple(case_yaml)
    except Exception as e:
        warnings.append(f"无法解析 case.yaml: {e}")
        return {"path": "case.yaml"}

    run_cfg = data.get("run", {}) if isinstance(data.get("run"), dict) else {}
    profile_cfg = data.get("profile", {}) if isinstance(data.get("profile"), dict) else {}
    requires_cfg = data.get("requires", {}) if isinstance(data.get("requires"), dict) else {}
    variants = {
        key[len("active_"):-len("_variant")]: val
        for key, val in run_cfg.items()
        if key.startswith("active_") and key.endswith("_variant")
    }

    return {
        "path": "case.yaml",
        "id": data.get("id", ""),
        "entrypoint": data.get("entrypoint", ""),
        "command": run_cfg.get("command", ""),
        "active_config": run_cfg.get("active_config", ""),
        "active_variants": variants,
        "profile_sqlite": profile_cfg.get("sqlite", ""),
        "profile_nsys_rep": profile_cfg.get("nsys_rep", ""),
        "requires": requires_cfg,
    }


def _parse_job_configs(root: Path, inventory: Inventory, warnings: list[str]) -> dict:
    hope_files = [p for p in inventory.config_files if p.endswith(".hope")]
    if not hope_files:
        return {}

    path = hope_files[0]
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(root / path, encoding="utf-8")
    except Exception as e:
        warnings.append(f"无法解析 {path}: {e}")
        return {"path": path}

    roles = parser["roles"] if parser.has_section("roles") else {}
    application = parser["application"] if parser.has_section("application") else {}
    docker = parser["docker"] if parser.has_section("docker") else {}

    env = {}
    if parser.has_section("others"):
        for key, value in parser["others"].items():
            if ".env." in key:
                env[key.rsplit(".env.", 1)[-1]] = _redact(value)

    return {
        "path": path,
        "worker_script": roles.get("worker.script", ""),
        "workers": roles.get("workers", ""),
        "gpus": _first_existing(roles, ("worker.gcores80g", "worker.gcores", "worker.gpu")),
        "vcore": roles.get("worker.vcore", ""),
        "memory": roles.get("worker.memory", ""),
        "app_name": application.get("appname", ""),
        "docker_image": docker.get("afo.docker.image.name", ""),
        "env": env,
    }


def _first_existing(section, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = section.get(key, "")
        if value:
            return value
    return ""


def _extract_script_commands(root: Path, script_files: list[str]) -> list[ScriptCommand]:
    commands: list[ScriptCommand] = []
    for rel in script_files:
        path = root / rel
        cwd = ""
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for idx, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cd_target = _parse_cd(line)
            if cd_target is not None:
                cwd = _join_rel(cwd, cd_target)
                continue
            cmd = _extract_command_from_line(line)
            if cmd:
                commands.append(ScriptCommand(_redact(cmd), rel, idx, cwd))
    return commands


def _parse_cd(line: str) -> str | None:
    if not line.startswith("cd "):
        return None
    try:
        parts = shlex.split(line)
    except ValueError:
        return None
    if len(parts) < 2:
        return None
    return parts[1]


def _join_rel(base: str, target: str) -> str:
    if target.startswith("$") or target.startswith("/"):
        return base
    joined = Path(base) / target if base else Path(target)
    parts: list[str] = []
    for part in joined.as_posix().split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _extract_command_from_line(line: str) -> str:
    if line.startswith(("if ", "elif ", "else", "fi", "for ", "done", "then", "while ")):
        return ""
    if line.startswith(("set ", "export ", "source ", "echo ", "mkdir ", "ln ", "ps ", "netstat ", "ls ")):
        return ""
    if "=" in line and not line.startswith(("python", "python3", "nsys", "torchrun", "bash", "$")):
        stripped_left = line.split("=", 1)[0].strip()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", stripped_left):
            return ""

    clean = line.strip()
    if clean.startswith("("):
        clean = clean[1:].strip()
    clean = clean.split("&&", 1)[0].strip()
    clean = clean.split("||", 1)[0].strip()
    if clean.endswith(")"):
        clean = clean[:-1].strip()

    if not _looks_like_runtime_command(clean):
        return ""
    return clean


def _looks_like_runtime_command(command: str) -> bool:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    if not parts:
        return False
    head = parts[0].strip("\"'")
    if _PYTHON_COMMAND_RE.search(head):
        return True
    if head in {"nsys", "torchrun", "bash", "sh", "$PYTHON_BIN", "${PYTHON_BIN}"}:
        return True
    return " nsys profile " in f" {command} " or "torch.distributed.launch" in command


def _select_commands(inventory: Inventory, facts: WarmupFacts, warnings: list[str]) -> None:
    case_command = facts.case_info.get("command", "")
    if case_command:
        facts.primary_command = _redact(case_command)
        facts.primary_source = "case.yaml run.command"
    elif facts.job_info.get("worker_script"):
        facts.primary_command = _redact(facts.job_info["worker_script"])
        facts.primary_source = f"{facts.job_info.get('path', '.hope')} worker.script"
    elif "start.sh" in inventory.script_files:
        facts.primary_command = "bash start.sh"
        facts.primary_source = "start.sh"
    elif "scripts/start.sh" in inventory.script_files:
        facts.primary_command = "bash scripts/start.sh"
        facts.primary_source = "scripts/start.sh"
    else:
        script_command = _first_script_command(facts.script_commands)
        if script_command:
            facts.primary_command = script_command.command
            facts.primary_source = f"{script_command.script}:{script_command.line}"
            facts.primary_script = script_command.script

    if not facts.primary_script:
        facts.primary_script = _script_from_command(facts.primary_command)

    if not facts.primary_command:
        entry = _fallback_python_entry(inventory.app_py_files)
        if entry:
            facts.primary_command = f"python {entry}"
            facts.primary_source = "python entry fallback"

    facts.profile_command = _select_profile_command(facts.script_commands)
    if not facts.primary_command:
        warnings.append("未发现入口命令")


def _first_script_command(commands: list[ScriptCommand]) -> ScriptCommand | None:
    preferred = ["scripts/start.sh", "start.sh", "scripts/run.sh", "run.sh", "scripts/profile_local.sh"]
    for script in preferred:
        for command in commands:
            if command.script == script and not command.command.startswith("nsys "):
                return command
    for command in commands:
        if not command.command.startswith("nsys "):
            return command
    return None


def _select_profile_command(commands: list[ScriptCommand]) -> str:
    for command in commands:
        if command.script.endswith("profile_local.sh") and "nsys profile" in command.command:
            return command.command
    for command in commands:
        if "nsys profile" in command.command:
            return command.command
    return ""


def _script_from_command(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    if len(parts) >= 2 and parts[0] in {"bash", "sh", "zsh"}:
        return parts[1]
    if parts and parts[0].endswith((".sh", ".bash", ".zsh")):
        return parts[0]
    return ""


def _fallback_python_entry(py_files: list[str]) -> str:
    for name in ("run.py", "main.py", "train.py"):
        if name in py_files:
            return name
    for path in py_files:
        if path.endswith("/__main__.py"):
            return path
    for path in py_files:
        if Path(path).name in _ENTRY_FILES:
            return path
    return py_files[0] if py_files else ""


def _resolve_active_config(root: Path, inventory: Inventory, facts: WarmupFacts,
                           warnings: list[str]) -> None:
    config_ref = facts.case_info.get("active_config", "")
    source = "case.yaml run.active_config" if config_ref else ""

    if not config_ref:
        command_candidates = [facts.primary_command]
        command_candidates.extend(c.command for c in facts.script_commands)
        for command in command_candidates:
            config_ref = _find_config_arg(command)
            if config_ref:
                source = "command argument"
                break

    if not config_ref:
        return

    cwd = ""
    for command in facts.script_commands:
        if config_ref in command.command:
            cwd = command.cwd
            break

    resolved = _resolve_repo_path(root, config_ref, cwd, inventory.config_files)
    if resolved:
        facts.active_config = resolved
        facts.active_config_exists = True
        facts.active_config_source = source
    else:
        facts.active_config = config_ref
        facts.active_config_source = source
        warnings.append(f"引用的 active config 未找到: {config_ref}")


def _find_config_arg(command: str) -> str:
    if not command:
        return ""
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    keys = {"--config", "--cfg_file", "--base_cfg_file", "-c"}
    for idx, part in enumerate(parts):
        if part in keys and idx + 1 < len(parts):
            return parts[idx + 1]
        for key in keys - {"-c"}:
            prefix = key + "="
            if part.startswith(prefix):
                return part[len(prefix):]
    return ""


def _resolve_repo_path(root: Path, ref: str, cwd: str, known_files: list[str]) -> str:
    ref = ref.strip().strip("\"'")
    if not ref or ref.startswith("$"):
        return ""
    ref_path = Path(ref)
    candidates: list[Path] = []
    if ref_path.is_absolute():
        candidates.append(ref_path)
    else:
        if cwd:
            candidates.append(root / cwd / ref)
            stripped = _strip_archive_component(cwd)
            if stripped != cwd:
                candidates.append(root / stripped / ref)
        candidates.append(root / ref)

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve().relative_to(root).as_posix()
        except (OSError, ValueError):
            continue

    normalized = ref.replace("\\", "/").lstrip("./")
    matches = [p for p in known_files if p.endswith(normalized)]
    if len(matches) == 1:
        return matches[0]
    if matches:
        shortest = sorted(matches, key=len)[0]
        return shortest
    return ""


def _strip_archive_component(cwd: str) -> str:
    parts = [
        p for p in cwd.split("/")
        if not p.endswith((".tar.gz", ".tgz", ".zip", ".tar"))
    ]
    return "/".join(parts)


def _select_profile_sqlite(root: Path, inventory: Inventory, facts: WarmupFacts) -> str:
    case_sqlite = facts.case_info.get("profile_sqlite", "")
    if case_sqlite and (root / case_sqlite).exists():
        return case_sqlite
    sqlite_files = [p for p in inventory.profile_files if p.endswith(".sqlite")]
    if len(sqlite_files) == 1:
        return sqlite_files[0]
    if sqlite_files:
        return sorted(sqlite_files, key=len)[0]
    return case_sqlite


def _infer_app_roots(py_files: list[str]) -> list[str]:
    roots: set[str] = set()
    for path in py_files:
        parts = path.split("/")
        if len(parts) == 1:
            roots.add(".")
            continue
        if ".runfiles" in path:
            idx = next((i for i, part in enumerate(parts) if part.endswith(".runfiles")), -1)
            if idx >= 0 and idx + 1 < len(parts):
                roots.add("/".join(parts[:idx + 2]))
                continue
        roots.add(parts[0])
    return sorted(roots)


def _infer_entry_files(root: Path, inventory: Inventory, facts: WarmupFacts) -> list[str]:
    entries: list[str] = []

    for command in [facts.primary_command] + [c.command for c in facts.script_commands]:
        entry = _python_entry_from_command(root, inventory, command)
        if entry and entry not in entries:
            entries.append(entry)

    wrapper_entry = _entry_from_bazel_wrapper(root, inventory.app_py_files)
    if wrapper_entry and wrapper_entry not in entries:
        entries.append(wrapper_entry)

    fallback = _fallback_python_entry(inventory.app_py_files)
    if not entries and fallback:
        entries.append(fallback)

    return entries


def _python_entry_from_command(root: Path, inventory: Inventory, command: str) -> str:
    if not command:
        return ""
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()

    for idx, part in enumerate(parts):
        clean = part.strip("\"'")
        if _looks_like_python_executable(clean):
            for candidate in parts[idx + 1:]:
                candidate = candidate.strip("\"'")
                if candidate.startswith("-"):
                    if candidate == "-m":
                        break
                    continue
                resolved = _resolve_python_path(root, candidate, "", inventory.app_py_files)
                if resolved:
                    return resolved
                break
        resolved = _resolve_python_path(root, clean, "", inventory.app_py_files)
        if resolved:
            return resolved
    return ""


def _looks_like_python_executable(value: str) -> bool:
    return bool(_PYTHON_COMMAND_RE.search(value) or value in {"$PYTHON_BIN", "${PYTHON_BIN}"})


def _resolve_python_path(root: Path, ref: str, cwd: str, py_files: list[str]) -> str:
    if not ref or ref.startswith("-") or ref.startswith("$"):
        return ""
    candidates = [ref]
    if not ref.endswith(".py"):
        candidates.append(ref + ".py")
    for candidate in candidates:
        resolved = _resolve_repo_path(root, candidate, cwd, py_files)
        if resolved and resolved in py_files:
            return resolved
    return ""


def _entry_from_bazel_wrapper(root: Path, py_files: list[str]) -> str:
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        if not path.is_file() or path.suffix:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"main_rel_path\s*=\s*['\"]([^'\"]+)['\"]", text)
        if not match:
            continue
        main_rel_path = match.group(1)
        matches = [p for p in py_files if p.endswith(main_rel_path)]
        if matches:
            return sorted(matches, key=len)[0]
    return ""


# Import tracing

def _trace_import_tree(root: Path, py_files: list[str],
                       entry_files: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    py_file_set = set(py_files)
    import_roots = _infer_import_roots(py_files, entry_files)
    visited: set[str] = set()
    tree: dict[str, list[str]] = {}

    def resolve_import(module_path: str, current_file: str, level: int = 0) -> str | None:
        current_dir = "/".join(current_file.split("/")[:-1])
        if level > 0:
            parts = current_dir.split("/") if current_dir else []
            if level > len(parts) + 1:
                return None
            base = parts[:max(0, len(parts) - level + 1)]
            module_rel = module_path.replace(".", "/") if module_path else ""
            full = "/".join([*base, module_rel]).strip("/")
            return _match_module_path(full, py_file_set)

        module_rel = module_path.replace(".", "/")
        for prefix in import_roots:
            full = f"{prefix}/{module_rel}".strip("/")
            match = _match_module_path(full, py_file_set)
            if match:
                return match
        return None

    def trace_file(file_path: str, depth: int = 0) -> list[str]:
        if file_path in visited or depth > 20:
            return []
        visited.add(file_path)

        try:
            source = (root / file_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        try:
            node = ast.parse(source)
        except SyntaxError:
            tree[file_path] = []
            return [file_path]

        imports: list[str] = []
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    resolved = resolve_import(alias.name, file_path)
                    if resolved and resolved not in imports:
                        imports.append(resolved)
            elif isinstance(stmt, ast.ImportFrom):
                module = stmt.module or ""
                resolved = resolve_import(module, file_path, stmt.level or 0)
                if resolved and resolved not in imports:
                    imports.append(resolved)
                for alias in stmt.names:
                    if alias.name == "*":
                        continue
                    submodule = f"{module}.{alias.name}" if module else alias.name
                    sub_resolved = resolve_import(submodule, file_path, stmt.level or 0)
                    if sub_resolved and sub_resolved not in imports:
                        imports.append(sub_resolved)

        tree[file_path] = imports
        result = [file_path]
        for imp in imports:
            result.extend(trace_file(imp, depth + 1))
        return result

    hot: list[str] = []
    for entry in entry_files:
        if entry in py_file_set:
            hot.extend(trace_file(entry))

    seen: set[str] = set()
    deduped: list[str] = []
    for path in hot:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped, tree


def _infer_import_roots(py_files: list[str], entry_files: list[str]) -> list[str]:
    roots = {""}
    for path in py_files:
        parts = path.split("/")
        for idx, part in enumerate(parts):
            if part.endswith(".runfiles") and idx + 1 < len(parts):
                roots.add("/".join(parts[:idx + 2]))
    for entry in entry_files:
        parts = entry.split("/")
        if len(parts) > 1:
            roots.add("/".join(parts[:-1]))
    return sorted(roots, key=len, reverse=True)


def _match_module_path(module_rel: str, py_file_set: set[str]) -> str | None:
    candidates = [f"{module_rel}.py", f"{module_rel}/__init__.py"]
    for candidate in candidates:
        candidate = candidate.strip("/")
        if candidate in py_file_set:
            return candidate
    return None


# Derived facts

def _find_inactive_configs(root: Path, inventory: Inventory, active_config: str) -> list[str]:
    if not active_config:
        return []
    active_path = Path(active_config)
    config_dir = active_path.parent.as_posix()
    config_suffixes = {".yaml", ".yml", ".json", ".toml"}
    inactive = []
    for config in inventory.config_files:
        if config == active_config:
            continue
        if Path(config).parent.as_posix() == config_dir and Path(config).suffix.lower() in config_suffixes:
            inactive.append(config)
    return sorted(inactive)


def _extract_perf_knobs(root: Path, active_config: str) -> list[str]:
    if not active_config:
        return []
    path = root / active_config
    if not path.exists():
        return []

    knob_re = re.compile(
        r"^\s*([A-Za-z0-9_.-]*(batch|worker|compile|amp|mixed|flash|prefetch|buffer|epoch|iter|learning_rate|optimizer)[A-Za-z0-9_.-]*)\s*:\s*(.+?)\s*(#.*)?$",
        re.I,
    )
    knobs: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return knobs
    for line in lines:
        match = knob_re.match(line)
        if not match:
            continue
        key = match.group(1).strip()
        value = _redact(match.group(3).strip())
        knobs.append(f"{key}: {value}")
        if len(knobs) >= 16:
            break
    return knobs


def _infer_workload(root: Path, inventory: Inventory, facts: WarmupFacts) -> str:
    signals = " ".join([
        facts.primary_command,
        facts.active_config,
        " ".join(str(v) for v in facts.active_variants.values()) if facts.active_variants else "",
    ]).lower()
    if "torch.distributed" in signals or "torchrun" in signals:
        return "distributed training"
    if "train" in signals or _any_file_contains(root, inventory, ("loss", "optimizer", "batch_size")):
        return "training"
    if "eval" in signals or "benchmark" in signals or "bench" in signals:
        return "evaluation/benchmark"
    if facts.profile_command or facts.profile_sqlite:
        return "profiled workload"
    return "python project"


def _any_file_contains(root: Path, inventory: Inventory, needles: tuple[str, ...]) -> bool:
    for path in inventory.app_py_files[:80] + ([inventory.config_files[0]] if inventory.config_files else []):
        try:
            text = (root / path).read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        if any(needle in text for needle in needles):
            return True
    return False


def _infer_constraints(facts: WarmupFacts) -> list[str]:
    constraints: list[str] = []
    requires = facts.case_info.get("requires", {})
    if requires.get("gpu"):
        min_gpus = requires.get("min_gpus")
        constraints.append(f"需要 GPU{f' x{min_gpus}' if min_gpus else ''}")
    if requires.get("nsys"):
        constraints.append("需要 Nsight Systems profiling")
    if requires.get("distributed_fallback"):
        constraints.append("存在分布式 fallback 路径")
    if facts.job_info.get("gpus"):
        constraints.append(f"作业 GPU: {facts.job_info['gpus']}")
    if facts.job_info.get("vcore"):
        constraints.append(f"作业 vCPU: {facts.job_info['vcore']}")
    if facts.job_info.get("memory"):
        constraints.append(f"作业内存: {facts.job_info['memory']}")
    joined_commands = "\n".join([facts.primary_command] + [c.command for c in facts.script_commands])
    if "torch.distributed.launch" in joined_commands or "torchrun" in joined_commands:
        constraints.append("检测到分布式启动")
    return constraints


def _populate_repo_setup(root: Path, repo_setup: RepoSetup, facts: WarmupFacts,
                         warnings: list[str], errors: list[str]) -> None:
    repo_setup.entry_point = facts.primary_command
    repo_setup.minimal_run = _minimal_run_tokens(facts)
    repo_setup.build_commands = _build_commands(root)
    repo_setup.test_commands = _test_commands(root)
    repo_setup.constraints = facts.constraints[:]

    if facts.workload in {"training", "distributed training"}:
        repo_setup.metric_grep = r"iter/s|step/s|throughput|loss"
        repo_setup.metric_lower_is_better = False
    elif facts.workload == "evaluation/benchmark":
        repo_setup.metric_grep = r"latency|time|ms"
        repo_setup.metric_lower_is_better = True

    if not facts.hot_path_files:
        warnings.append("未发现应用热路径 Python 文件")
    if facts.active_config and not facts.active_config_exists:
        warnings.append("active config 无法验证")

    entry_ok = _entry_command_verified(root, facts.primary_command)
    config_ok = not facts.active_config or facts.active_config_exists
    hot_ok = bool(facts.hot_path_files)
    if entry_ok and config_ok and hot_ok and not errors:
        repo_setup.source = "warmup_verified"
        repo_setup.verified_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        repo_setup.source = "warmup_partial"


def _minimal_run_tokens(facts: WarmupFacts) -> list[str]:
    candidates = [facts.primary_command]
    candidates.extend(c.command for c in facts.script_commands)
    for command in candidates:
        if not command:
            continue
        try:
            parts = shlex.split(command)
        except ValueError:
            continue
        if not parts:
            continue
        if _looks_like_python_executable(parts[0]) or parts[0] in {"torchrun"}:
            return parts
        if "torch.distributed.launch" in command:
            return parts
    return []


def _build_commands(root: Path) -> list[list[str]]:
    if (root / "pyproject.toml").exists():
        return [["pip", "install", "-e", "."]]
    if (root / "requirements.txt").exists():
        return [["pip", "install", "-r", "requirements.txt"]]
    return []


def _test_commands(root: Path) -> list[list[str]]:
    test_dir = root / "tests"
    if test_dir.is_dir() and any(test_dir.glob("test_*.py")):
        return [["python", "-m", "pytest", "tests/"]]
    if any(root.glob("test_*.py")):
        return [["python", "-m", "pytest"]]
    if (root / "pytest.ini").exists() or (root / "tox.ini").exists():
        return [["python", "-m", "pytest"]]
    return []


def _entry_command_verified(root: Path, command: str) -> bool:
    if not command:
        return False
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    if not parts:
        return False
    if parts[0] in {"bash", "sh", "zsh"} and len(parts) >= 2:
        return (root / parts[1]).exists()
    saw_python_file = False
    for part in parts:
        clean = part.strip("\"'")
        if clean.endswith(".py") and (root / clean).exists():
            return True
        if clean.endswith(".py"):
            saw_python_file = True
    if saw_python_file:
        return False
    return True


# Output

def _build_summary(root: Path, inventory: Inventory, facts: WarmupFacts, repo_setup: RepoSetup,
                   warnings: list[str], errors: list[str], overview_path: str) -> dict:
    return {
        "repo": str(root),
        "source": repo_setup.source,
        "overview_path": overview_path,
        "entry_point": repo_setup.entry_point,
        "minimal_run": repo_setup.minimal_run,
        "profile_command": facts.profile_command,
        "active_config": facts.active_config,
        "active_variants": facts.active_variants,
        "profile_sqlite": facts.profile_sqlite,
        "hot_path_count": len(facts.hot_path_files),
        "ignored_counts": dict(sorted(inventory.ignored_counts.items())),
        "warnings": warnings,
        "errors": errors,
    }


def _build_overview(root: Path, inventory: Inventory, facts: WarmupFacts,
                    repo_setup: RepoSetup, warnings: list[str], errors: list[str]) -> str:
    lines: list[str] = ["# Workspace Overview", ""]

    lines.append("## 项目概览")
    lines.append(f"- 工作负载: {_display_workload(facts.workload)}")
    if facts.primary_source:
        lines.append(f"- 入口证据: {facts.primary_source}")
    if facts.case_info.get("id"):
        lines.append(f"- Case ID: `{facts.case_info['id']}`")
    lines.append(f"- 已扫描应用 Python 文件: {len(inventory.app_py_files)}")
    lines.append("")

    lines.append("## 如何运行 / 采集 Profile")
    if facts.primary_command:
        lines.append(f"- 主入口: `{_redact(facts.primary_command)}`")
    if repo_setup.minimal_run:
        lines.append(f"- 最小 Python 命令: `{_redact(' '.join(repo_setup.minimal_run))}`")
    if facts.profile_command:
        lines.append(f"- Profile 采集: `{_redact(facts.profile_command)}`")
    if facts.primary_script:
        lines.append(f"- 启动脚本: `{facts.primary_script}`")
    if facts.constraints:
        for constraint in facts.constraints:
            lines.append(f"- 约束: {constraint}")
    lines.append("")

    lines.append("## 当前配置")
    if facts.active_config:
        suffix = "已验证" if facts.active_config_exists else "未找到"
        lines.append(f"- 配置文件: `{facts.active_config}` ({suffix}, {facts.active_config_source})")
    if facts.active_variants:
        for key, value in sorted(facts.active_variants.items()):
            lines.append(f"- 当前变体 {key}: `{value}`")
    if facts.profile_sqlite:
        lines.append(f"- Profile sqlite: `{facts.profile_sqlite}`")
    if facts.case_info.get("profile_nsys_rep"):
        lines.append(f"- Nsight report: `{facts.case_info['profile_nsys_rep']}`")
    if facts.job_info:
        lines.append(f"- 作业配置: `{facts.job_info.get('path', '')}`")
        if facts.job_info.get("app_name"):
            lines.append(f"- 作业名: `{facts.job_info['app_name']}`")
    if facts.perf_knobs:
        lines.append("- 性能相关参数:")
        for knob in facts.perf_knobs[:12]:
            lines.append(f"  - `{knob}`")
    lines.append("")

    lines.append("## 代码地图")
    if facts.app_roots:
        lines.append("- 应用根目录: " + ", ".join(f"`{r}`" for r in facts.app_roots[:12]))
    if facts.entry_files:
        lines.append("- 入口文件: " + ", ".join(f"`{p}`" for p in facts.entry_files[:8]))
    lines.append(f"- 热路径文件数: {len(facts.hot_path_files)}")
    for directory, files in _group_files(facts.hot_path_files).items():
        lines.append(f"- `{directory}/`: " + ", ".join(files[:10]))
    lines.append("")

    if facts.import_tree:
        lines.append("## 导入树")
        lines.append("```")
        lines.extend(_render_import_tree(facts.hot_path_files, facts.import_tree, limit_roots=3))
        lines.append("```")
        lines.append("")

    lines.append("## 跳过 / 噪声")
    if inventory.ignored_counts:
        for reason, count in sorted(inventory.ignored_counts.items()):
            examples = ", ".join(inventory.ignored_examples.get(reason, [])[:3])
            lines.append(f"- 跳过 {count} 个 `{reason}` 目录" + (f": {examples}" if examples else ""))
    if facts.inactive_configs:
        lines.append(f"- 未启用配置 ({len(facts.inactive_configs)}):")
        for config in facts.inactive_configs[:20]:
            lines.append(f"  - `{config}`")
        if len(facts.inactive_configs) > 20:
            lines.append(f"  - ... 还有 {len(facts.inactive_configs) - 20} 个")
    cold_files = [p for p in inventory.app_py_files if p not in set(facts.hot_path_files)]
    if cold_files:
        lines.append(f"- 未在热路径中的其他应用 Python 文件: {len(cold_files)}")
    lines.append("")

    lines.append("## Warmup 可信度")
    lines.append(f"- 来源状态: `{repo_setup.source}`")
    if repo_setup.verified_at:
        lines.append(f"- 验证时间: `{repo_setup.verified_at}`")
    for warning in warnings:
        lines.append(f"- 警告: {warning}")
    for error in errors:
        lines.append(f"- 错误: {error}")

    return "\n".join(lines)


def _display_workload(workload: str) -> str:
    labels = {
        "distributed training": "分布式训练",
        "training": "训练",
        "evaluation/benchmark": "评估 / Benchmark",
        "profiled workload": "已采集 profile 的工作负载",
        "python project": "Python 项目",
    }
    return labels.get(workload, workload)


def _group_files(files: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for path in files[:80]:
        p = Path(path)
        directory = p.parent.as_posix() if p.parent.as_posix() != "." else "."
        grouped.setdefault(directory, []).append(p.name)
    return dict(sorted(grouped.items()))


def _render_import_tree(hot_path_files: list[str], import_tree: dict[str, list[str]],
                        limit_roots: int) -> list[str]:
    all_imported = {child for children in import_tree.values() for child in children}
    roots = [path for path in hot_path_files if path not in all_imported]
    if not roots:
        roots = hot_path_files[:1]

    lines: list[str] = []
    seen_edges: set[tuple[str, str]] = set()

    def walk(path: str, indent: int = 0) -> None:
        if len(lines) >= 80:
            return
        lines.append("  " * indent + path)
        for child in import_tree.get(path, [])[:12]:
            edge = (path, child)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            walk(child, indent + 1)

    for root in roots[:limit_roots]:
        walk(root)
    if len(roots) > limit_roots:
        lines.append(f"... {len(roots) - limit_roots} more roots")
    return lines


def _redact(text: str) -> str:
    if not text:
        return text
    return _ASSIGNMENT_RE.sub(lambda m: f"{m.group('key')}{m.group('sep')}***REDACTED***", text)
