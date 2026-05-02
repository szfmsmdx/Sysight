"""variants — resolve variant / factory keys to concrete implementation classes.

Many Python ML repos use a registry / factory pattern:
    MODEL_REGISTRY = {
        "resnet50": ResNet50,
        "vit_base": ViTBase,
    }
or:
    def build_model(name):
        if name == "resnet50":
            return ResNet50(...)
        elif name == "vit_base":
            return ViTBase(...)

or using decorators:
    @register("resnet50")
    class ResNet50(nn.Module): ...

Commands:
  scanner variants <repo>
      find all registry dicts / build functions in repo

  scanner variants <repo> --key resnet50
      find which class/function maps to "resnet50"

  scanner variants <repo> --file src/models/registry.py
      list all variant mappings in that file
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from .files import _walk


@dataclass
class VariantEntry:
    key: str            # string key, e.g. "resnet50"
    target: str         # class/function name, e.g. "ResNet50"
    file: str           # relative to repo root
    line: int           # 1-based line of the mapping definition
    kind: str           # "dict_literal" | "if_elif" | "decorator" | "assign"
    context: str        # surrounding source line (stripped)


@dataclass
class VariantsResult:
    repo: str
    total: int
    entries: list[VariantEntry] = field(default_factory=list)
    error: str | None = None


# ── AST-based extraction ───────────────────────────────────────────────────────

def _extract_dict_literals(tree: ast.Module, rel: str, lines: list[str]) -> list[VariantEntry]:
    """Find dict literals of the form {"key": ClassName} at module or class body level."""
    entries: list[VariantEntry] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            # key must be a string literal
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                continue
            # value must be a Name (class/func ref) or Attribute
            if isinstance(v, ast.Name):
                target = v.id
            elif isinstance(v, ast.Attribute):
                target = v.attr
            else:
                continue
            src_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
            entries.append(VariantEntry(
                key=k.value,
                target=target,
                file=rel,
                line=getattr(k, "lineno", node.lineno),
                kind="dict_literal",
                context=src_line,
            ))
    return entries


def _extract_if_elif(tree: ast.Module, rel: str, lines: list[str]) -> list[VariantEntry]:
    """Find if/elif chains of the form: if name == "key": return SomeClass(...)."""
    entries: list[VariantEntry] = []

    def _visit_if(node: ast.If) -> None:
        # Look for: if <var> == "key"  or  if "key" == <var>
        cmp = node.test
        if not isinstance(cmp, ast.Compare) or len(cmp.ops) != 1:
            return
        if not isinstance(cmp.ops[0], ast.Eq):
            return

        key_val: str | None = None
        if isinstance(cmp.left, ast.Constant) and isinstance(cmp.left.value, str):
            key_val = cmp.left.value
        elif (cmp.comparators and isinstance(cmp.comparators[0], ast.Constant)
              and isinstance(cmp.comparators[0].value, str)):
            key_val = cmp.comparators[0].value

        if key_val is None:
            return

        # Look for return / assignment with a Name in body
        for stmt in node.body:
            target: str | None = None
            if isinstance(stmt, ast.Return) and stmt.value:
                val = stmt.value
                if isinstance(val, ast.Call):
                    if isinstance(val.func, ast.Name):
                        target = val.func.id
                    elif isinstance(val.func, ast.Attribute):
                        target = val.func.attr
                elif isinstance(val, ast.Name):
                    target = val.id
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                val = stmt.value if isinstance(stmt, ast.Assign) else getattr(stmt, "value", None)
                if val and isinstance(val, ast.Name):
                    target = val.id
                elif val and isinstance(val, ast.Call):
                    if isinstance(val.func, ast.Name):
                        target = val.func.id
                    elif isinstance(val.func, ast.Attribute):
                        target = val.func.attr

            if target:
                src_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                entries.append(VariantEntry(
                    key=key_val,
                    target=target,
                    file=rel,
                    line=node.lineno,
                    kind="if_elif",
                    context=src_line,
                ))
                break

        # Recurse into orelse (elif)
        if node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            _visit_if(node.orelse[0])

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            _visit_if(node)

    return entries


def _extract_decorators(tree: ast.Module, rel: str, lines: list[str]) -> list[VariantEntry]:
    """Find @register("key") or @registry.register("key") patterns."""
    entries: list[VariantEntry] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        target_name = node.name
        for dec in node.decorator_list:
            # @register("key") or @xxx.register("key")
            call_node: ast.Call | None = None
            if isinstance(dec, ast.Call):
                call_node = dec

            if call_node is None:
                continue
            if not call_node.args:
                continue
            first_arg = call_node.args[0]
            if not (isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str)):
                continue
            key_val = first_arg.value
            src_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
            # Line of decorator, not the class/func line
            dec_lineno = getattr(call_node, "lineno", node.lineno)
            entries.append(VariantEntry(
                key=key_val,
                target=target_name,
                file=rel,
                line=dec_lineno,
                kind="decorator",
                context=src_line,
            ))
    return entries


# ── Text-based fallback for common patterns ───────────────────────────────────

# Matches:  "key": SomeClass  or  'key': SomeClass
_DICT_ASSIGN_RE = re.compile(
    r"""['"]([\w\-\.\/]+)['"]\s*:\s*([A-Z][A-Za-z0-9_]*)"""
)


def _extract_text_fallback(src: str, rel: str) -> list[VariantEntry]:
    """Text-based fallback for files that fail AST parsing."""
    entries: list[VariantEntry] = []
    for lineno, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for m in _DICT_ASSIGN_RE.finditer(line):
            entries.append(VariantEntry(
                key=m.group(1),
                target=m.group(2),
                file=rel,
                line=lineno,
                kind="dict_literal",
                context=stripped,
            ))
    return entries


# ── Public API ────────────────────────────────────────────────────────────────

def _extract_from_file(root: Path, rel: str) -> list[VariantEntry]:
    """Extract all variant mappings from a single file."""
    p = root / rel
    try:
        src = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    try:
        tree = ast.parse(src, filename=str(p))
    except SyntaxError:
        return _extract_text_fallback(src, rel)

    lines = src.splitlines()
    entries: list[VariantEntry] = []
    entries.extend(_extract_dict_literals(tree, rel, lines))
    entries.extend(_extract_if_elif(tree, rel, lines))
    entries.extend(_extract_decorators(tree, rel, lines))

    # Deduplicate by (key, target, line)
    seen: set[tuple[str, str, int]] = set()
    unique: list[VariantEntry] = []
    for e in entries:
        sig = (e.key, e.target, e.line)
        if sig not in seen:
            seen.add(sig)
            unique.append(e)
    return unique


def find_variants(
    repo: str,
    key: str | None = None,
    file_filter: str | None = None,
    max_results: int = 500,
) -> VariantsResult:
    """Find variant/factory mappings across the repo.

    Args:
        repo: repo root.
        key: if given, filter to entries with this string key.
        file_filter: if given, restrict to this relative file path.
        max_results: safety cap.
    """
    root = Path(repo).resolve()
    all_entries: list[VariantEntry] = []

    for p in _walk(root):
        if p.suffix.lower() != ".py":
            continue
        rel = str(p.relative_to(root))
        if file_filter and rel != file_filter:
            continue

        entries = _extract_from_file(root, rel)
        if key is not None:
            entries = [e for e in entries if e.key == key]
        all_entries.extend(entries)

        if len(all_entries) >= max_results:
            all_entries = all_entries[:max_results]
            break

    all_entries.sort(key=lambda e: (e.file, e.line))
    return VariantsResult(repo=str(root), total=len(all_entries), entries=all_entries)

# ── ToolDef ────────────────────────────────────────────────────────────────
from sysight.tools.registry import ToolDef  # noqa: E402

VARIANTS_TOOL = ToolDef(
    name="scanner_variants",
    description="Find registry/factory mappings (dict literals, if/elif chains, decorators) and resolve variant keys to concrete classes",
    parameters={
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Path to repo root"},
            "key": {"type": "string", "description": "Specific variant key to look up, e.g. 'resnet50'"},
            "file": {"type": "string", "description": "Restrict search to a specific file"},
            "max_results": {"type": "integer", "default": 200},
        },
        "required": ["repo"],
    },
    fn=find_variants,
    read_only=True,
)
