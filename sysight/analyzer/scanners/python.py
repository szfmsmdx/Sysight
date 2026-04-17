"""Python scanner — AST-based import resolution and function extraction."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

from .base import (
    BaseScanner, CallSiteFacts, CLI_HINTS, FileFacts, FunctionFacts, ImportBinding,
    INFERENCE_KW, TRAINING_KW, ENTRY_FILE_HINTS, norm, should_ignore,
)

# GPU-related file-level tag patterns
_GPU_TAGS_FILE = (
    ('triton_jit',   re.compile(r'@triton\.jit')),
    ('nccl_comm',    re.compile(r'(?:dist\.all_reduce|dist\.broadcast|dist\.all_gather|nccl)')),
    ('dataloader',   re.compile(r'DataLoader')),
    ('autograd_rec', re.compile(r'torch\.autograd\.profiler\.record_function')),
)


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = [node.attr]
        cur: ast.AST = node.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return node.attr
    return None


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare):
        return False
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
        return False
    if len(node.comparators) != 1:
        return False
    lhs, rhs = node.left, node.comparators[0]
    return (
        isinstance(lhs, ast.Name) and lhs.id == "__name__"
        and isinstance(rhs, ast.Constant) and rhs.value == "__main__"
    )


def _collect_calls(nodes: Iterable[ast.stmt]) -> list[str]:
    out: list[str] = []
    for node in nodes:
        try:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    name = _dotted_name(child.func)
                    if name:
                        out.append(name)
        except RecursionError:
            pass
    return out


def _fn_gpu_tags(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Detect GPU-related tags from function decorators/body."""
    tags: list[str] = []
    for deco in fn_node.decorator_list:
        if isinstance(deco, ast.Name) and deco.id == "jit":
            tags.append("triton_jit")
        elif isinstance(deco, ast.Attribute):
            deco_str = ast.unparse(deco) if hasattr(ast, 'unparse') else str(deco.attr)
            if "triton.jit" in deco_str:
                tags.append("triton_jit")
    return tags


def _node_repr(node: ast.expr) -> str:
    """Best-effort single-line repr of an AST expression node."""
    try:
        if hasattr(ast, "unparse"):
            return ast.unparse(node)
    except Exception:
        pass
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return "<expr>"


def _collect_callsites(
    body: list[ast.stmt],
    rel_path: str,
    src_lines: list[str],
    enclosing_fn: str | None,
) -> list[CallSiteFacts]:
    """Walk *body* and emit a CallSiteFacts for every Call node found.

    loop_depth counts nested for/while loops around each call.
    We do a manual DFS instead of ast.walk so we can track loop nesting.
    """
    results: list[CallSiteFacts] = []

    def _visit(stmts: list[ast.stmt], depth: int) -> None:
        for stmt in stmts:
            _visit_stmt(stmt, depth)

    def _visit_stmt(node: ast.stmt, depth: int) -> None:
        # Recurse into loop bodies with depth+1
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            _visit(node.body, depth + 1)
            _visit(node.orelse, depth)
            return
        # Recurse into conditionals / try / with without increasing depth
        if isinstance(node, ast.If):
            _visit(node.body, depth)
            _visit(node.orelse, depth)
            return
        if isinstance(node, (ast.With, ast.AsyncWith)):
            _visit(node.body, depth)
            return
        if isinstance(node, ast.Try):
            _visit(node.body, depth)
            _visit(node.orelse, depth)
            _visit(node.finalbody if hasattr(node, "finalbody") else [], depth)
            for handler in node.handlers:
                _visit(handler.body, depth)
            return
        if hasattr(ast, "TryStar") and isinstance(node, ast.TryStar):  # Python 3.11+
            _visit(node.body, depth)
            for handler in node.handlers:
                _visit(handler.body, depth)
            _visit(node.orelse, depth)
            _visit(node.finalbody, depth)
            return
        # Collect calls from expression statements and assignments
        _collect_from_stmt(node, depth)

    def _collect_from_stmt(stmt: ast.stmt, depth: int) -> None:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Determine call_name and receiver
            if isinstance(func, ast.Attribute):
                call_name = func.attr
                receiver = _node_repr(func.value)
                full_call_name = f"{receiver}.{call_name}"
            elif isinstance(func, ast.Name):
                call_name = func.id
                receiver = None
                full_call_name = call_name
            else:
                continue  # complex expression, skip

            line = node.lineno
            col = node.col_offset
            end_line = getattr(node, "end_lineno", None)
            end_col = getattr(node, "end_col_offset", None)
            source_line = src_lines[line - 1].rstrip() if 0 < line <= len(src_lines) else ""

            args_repr = [_node_repr(a) for a in node.args]
            keywords = {
                kw.arg: _node_repr(kw.value)
                for kw in node.keywords
                if kw.arg is not None
            }

            cs_id = f"{rel_path}:{line}:{col}:{call_name}"
            results.append(CallSiteFacts(
                id=cs_id,
                path=rel_path,
                line=line,
                col=col,
                end_line=end_line,
                end_col=end_col,
                call_name=call_name,
                full_call_name=full_call_name,
                receiver=receiver,
                args_repr=args_repr,
                keywords=keywords,
                enclosing_function=enclosing_fn,
                loop_depth=depth,
                source_line=source_line,
            ))

    try:
        _visit(body, 0)
    except RecursionError:
        pass
    return results


class PythonScanner(BaseScanner):
    EXTENSIONS = frozenset({".py"})

    def __init__(self, repo_root: Path, include_runfiles: bool = False) -> None:
        super().__init__(repo_root, include_runfiles=include_runfiles)
        self._mod2file: dict[str, str] = {}

    def scan(self) -> dict[str, FileFacts]:
        paths = sorted(
            p for p in self.root.rglob("*.py")
            if not should_ignore(self.root, p, include_runfiles=self.include_runfiles)
        )
        self._mod2file = {self._modname(p): str(p.relative_to(self.root)) for p in paths}
        out: dict[str, FileFacts] = {}
        for p in paths:
            rel = str(p.relative_to(self.root))
            try:
                out[rel] = self._parse(p)
            except (OSError, UnicodeDecodeError, SyntaxError, RecursionError) as e:
                self.warnings.append(f"Skipped Python file {rel}: {type(e).__name__}")
        return out

    def scan_paths(self, paths: list[Path]) -> dict[str, FileFacts]:
        """Targeted scan: parse only the given paths.

        Builds a minimal module map from just the selected paths so that
        relative import resolution still works within the selected set.
        """
        py_paths = sorted(
            p for p in paths
            if p.suffix.lower() == ".py"
            and not should_ignore(self.root, p, include_runfiles=self.include_runfiles)
        )
        # Build mod2file only from selected paths
        self._mod2file = {self._modname(p): str(p.relative_to(self.root)) for p in py_paths}
        out: dict[str, FileFacts] = {}
        for p in py_paths:
            rel = str(p.relative_to(self.root))
            try:
                out[rel] = self._parse(p)
            except (OSError, UnicodeDecodeError, SyntaxError, RecursionError) as e:
                self.warnings.append(f"Skipped Python file {rel}: {type(e).__name__}")
        return out

    def _parse_one(self, path: Path) -> FileFacts | None:
        return self._parse(path)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _modname(self, p: Path) -> str:
        parts = list(p.relative_to(self.root).parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = p.stem
        return ".".join(norm(parts))

    def _resolve_rel(self, current: str, level: int, module: str | None) -> str:
        if level <= 0:
            return module or ""
        cur_parts = current.split(".") if current else []
        if cur_parts:
            cur_parts = cur_parts[:-1]
        base = cur_parts[: max(len(cur_parts) - level + 1, 0)]
        extra = module.split(".") if module else []
        return ".".join(norm(base + extra))

    def _resolve_file(self, mod: str) -> str | None:
        if not mod:
            return None
        if mod in self._mod2file:
            return self._mod2file[mod]
        return self._mod2file.get(f"{mod}.__init__")

    # ── parse ─────────────────────────────────────────────────────────────────

    def _parse(self, path: Path) -> FileFacts:
        rel = str(path.relative_to(self.root))
        mod = self._modname(path)
        src = path.read_text(encoding="utf-8")
        src_lines = src.splitlines()
        tree = ast.parse(src, filename=rel)

        fns: dict[str, FunctionFacts] = {}
        imps: dict[str, ImportBinding] = {}
        clis: set[str] = set()
        top: list[ast.stmt] = []
        guard: list[ast.stmt] = []
        has_guard = False
        all_callsites: list[CallSiteFacts] = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                gpu_tags = _fn_gpu_tags(node)
                qname = f"{rel}::{node.name}"
                fns[node.name] = FunctionFacts(
                    name=node.name,
                    qualified_name=qname,
                    line=node.lineno,
                    end_line=getattr(node, 'end_lineno', None),
                    calls=_collect_calls(node.body),
                    is_gpu_kernel=bool(gpu_tags),
                    extra={'gpu_tags': gpu_tags} if gpu_tags else {},
                )
                all_callsites.extend(
                    _collect_callsites(node.body, rel, src_lines, qname)
                )
                continue
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        key = f"{node.name}.{item.name}"
                        qname = f"{rel}::{key}"
                        gpu_tags = _fn_gpu_tags(item)
                        fns[key] = FunctionFacts(
                            name=item.name,
                            qualified_name=qname,
                            line=item.lineno,
                            end_line=getattr(item, 'end_lineno', None),
                            calls=_collect_calls(item.body),
                            is_gpu_kernel=bool(gpu_tags),
                            extra={'gpu_tags': gpu_tags} if gpu_tags else {},
                        )
                        all_callsites.extend(
                            _collect_callsites(item.body, rel, src_lines, qname)
                        )
                top.append(node)
                continue
            if isinstance(node, ast.Import):
                for a in node.names:
                    alias = a.asname or a.name.split(".")[0]
                    imps[alias] = ImportBinding(
                        alias=alias, binding_type="module",
                        target_module=a.name,
                        target_file=self._resolve_file(a.name),
                    )
                    if any(h in a.name for h in CLI_HINTS):
                        clis.add(a.name.split(".")[0])
                top.append(node)
                continue
            if isinstance(node, ast.ImportFrom):
                resolved = self._resolve_rel(mod, node.level, node.module)
                for a in node.names:
                    alias = a.asname or a.name
                    imps[alias] = ImportBinding(
                        alias=alias, binding_type="symbol",
                        target_module=resolved, target_symbol=a.name,
                        target_file=self._resolve_file(resolved),
                    )
                    dotted = f"{resolved}.{a.name}" if resolved else a.name
                    if any(h in dotted for h in CLI_HINTS):
                        clis.add(a.name)
                top.append(node)
                continue
            if isinstance(node, ast.If) and _is_main_guard(node.test):
                has_guard = True
                guard.extend(node.body)
                continue
            top.append(node)

        tr, inf, gen, notes = self._score(path, has_guard, clis, src)
        # Collect file-level GPU tags from both functions and source text
        file_gpu_tags: list[str] = []
        for tag, pat in _GPU_TAGS_FILE:
            if pat.search(src):
                file_gpu_tags.append(tag)
        file_extra: dict[str, str | int | float | list[str]] = {}
        if file_gpu_tags:
            file_extra['gpu_tags'] = file_gpu_tags
            cuda_kernel_count = sum(
                1 for fn in fns.values() if fn.is_gpu_kernel
            )
            if cuda_kernel_count:
                file_extra['triton_kernel_count'] = cuda_kernel_count
        return FileFacts(
            path=rel, module_name=mod, language="python",
            has_main_guard=has_guard, cli_frameworks=sorted(clis),
            top_level_calls=_collect_calls(top),
            main_guard_calls=_collect_calls(guard),
            functions=fns, imports=imps,
            training_score=tr, inference_score=inf, generic_score=gen, notes=notes,
            callsites=all_callsites,
            extra=file_extra,
        )

    def _score(self, path: Path, has_guard: bool, clis: set[str],
               src: str) -> tuple[int, int, int, list[str]]:
        name = path.stem.lower()
        tokens = set(re.split(r"[_\-.]+", name))
        tr = inf = gen = 0
        notes: list[str] = []
        real = (
            has_guard
            or name in {"train", "trainer", "inference", "infer", "predict", "main"}
            or path.parent.name in {"scripts", "bin"}
        )
        for hint, w in ENTRY_FILE_HINTS.items():
            if hint in tokens or name == hint:
                if hint in {"train", "trainer", "pretrain", "finetune", "fit"}:
                    tr += w if real else max(w - 2, 0)
                elif hint in {"infer", "inference", "predict", "generate", "serve", "eval"}:
                    inf += w if real else max(w - 2, 0)
                else:
                    gen += w if real else max(w - 1, 0)
                notes.append(f"filename:{hint}")
        if has_guard:
            tr += 2; inf += 2; gen += 3
            notes.append("main-guard")
        if clis:
            tr += 1; inf += 1
            notes.append(f"cli:{','.join(sorted(clis))}")
        low = src.lower()
        for kw in TRAINING_KW:
            if kw in low:
                tr += 2; notes.append(f"train:{kw}")
        for kw in INFERENCE_KW:
            if kw in low:
                inf += 2; notes.append(f"infer:{kw}")
        return tr, inf, gen, notes
