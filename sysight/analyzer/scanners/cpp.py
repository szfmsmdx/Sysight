"""C/C++ / CUDA scanner."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from .base import (
    BaseScanner, FileFacts, FunctionFacts, ImportBinding,
    should_ignore, match_brace, score_by_content,
)

_INCLUDE = re.compile(r'^\s*#\s*include\s+["<]([^">]+)[">]', re.MULTILINE)
_FN_DEF = re.compile(
    r'^(?![ \t]*(?:class|struct|if|else|for|while|switch|do|namespace)\b)'
    r'[ \t]*(?:[\w:*&<>, \t]+\s+)'
    r'(\w+)\s*\([^;{]*\)\s*(?:const\s*)?(?:override\s*)?(?:noexcept[^;{]*)?\{',
    re.MULTILINE,
)
# Matches CUDA __global__ kernel definitions
_CUDA_GLOBAL = re.compile(
    r'__global__\s+(?:void\s+|[\w:*&<>, \t]+\s+)'
    r'(\w+)\s*\([^;{]*\)\s*\{',
    re.MULTILINE,
)
# CUDA kernel launch  kernel<<<grid, block>>>(...)
_CUDA_LAUNCH = re.compile(r'(\w+)\s*<<<')
_CALL = re.compile(r'\b([A-Za-z_]\w*)\s*\(')
_KW = {
    "if", "for", "while", "switch", "do", "return", "sizeof",
    "static_cast", "dynamic_cast", "reinterpret_cast", "const_cast",
    "new", "delete", "catch", "throw", "decltype", "alignof",
    "offsetof", "typeid", "nullptr",
}


class CppScanner(BaseScanner):
    EXTENSIONS = frozenset({".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh"})

    def __init__(self, repo_root: Path) -> None:
        super().__init__(repo_root)
        self._rel2path: dict[str, str] = {}

    def scan(self) -> dict[str, FileFacts]:
        paths = sorted(
            p for p in self.root.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS and not should_ignore(self.root, p)
        )
        basename_map: dict[str, list[str]] = defaultdict(list)
        for p in paths:
            rel = str(p.relative_to(self.root))
            self._rel2path[rel] = rel
            basename_map[p.name].append(rel)

        out: dict[str, FileFacts] = {}
        for p in paths:
            rel = str(p.relative_to(self.root))
            try:
                out[rel] = self._parse(p, basename_map)
            except (OSError, UnicodeDecodeError, RecursionError) as e:
                self.warnings.append(f"Skipped C/C++ file {rel}: {type(e).__name__}")
        return out

    def scan_paths(self, paths: list[Path]) -> dict[str, FileFacts]:
        """Targeted scan: parse only the given paths.

        Builds a basename_map from just the selected C/C++ paths for accurate
        include resolution within the selected set.
        """
        cpp_paths = sorted(
            p for p in paths
            if p.suffix.lower() in self.EXTENSIONS and not should_ignore(self.root, p)
        )
        basename_map: dict[str, list[str]] = defaultdict(list)
        for p in cpp_paths:
            rel = str(p.relative_to(self.root))
            self._rel2path[rel] = rel
            basename_map[p.name].append(rel)

        out: dict[str, FileFacts] = {}
        for p in cpp_paths:
            rel = str(p.relative_to(self.root))
            try:
                out[rel] = self._parse(p, basename_map)
            except (OSError, UnicodeDecodeError, RecursionError) as e:
                self.warnings.append(f"Skipped C/C++ file {rel}: {type(e).__name__}")
        return out

    def _parse_one(self, path: Path) -> FileFacts | None:
        basename_map: dict[str, list[str]] = defaultdict(list)
        rel = str(path.relative_to(self.root))
        self._rel2path[rel] = rel
        basename_map[path.name].append(rel)
        return self._parse(path, basename_map)

    def _resolve_include(self, inc: str, cur_dir: Path,
                         basename_map: dict[str, list[str]]) -> str | None:
        candidate = (cur_dir / inc).resolve()
        try:
            rel = str(candidate.relative_to(self.root))
            if rel in self._rel2path:
                return rel
        except ValueError:
            pass
        hits = basename_map.get(Path(inc).name, [])
        if len(hits) == 1:
            return hits[0]
        for rel in self._rel2path:
            if rel.replace("\\", "/").endswith(inc.replace("\\", "/")):
                return rel
        return None

    def _parse(self, path: Path, basename_map: dict[str, list[str]]) -> FileFacts:
        rel = str(path.relative_to(self.root))
        src = path.read_text(encoding="utf-8", errors="replace")

        imps = {
            Path(inc).stem: ImportBinding(
                alias=Path(inc).stem, binding_type="include", target_module=inc,
                target_file=self._resolve_include(inc, path.parent, basename_map),
            )
            for m in _INCLUDE.finditer(src)
            for inc in [m.group(1)]
        }

        # Collect CUDA __global__ kernel names for lookup
        cuda_kernel_names: set[str] = {m.group(1) for m in _CUDA_GLOBAL.finditer(src)}
        cuda_launch_names: set[str] = {m.group(1) for m in _CUDA_LAUNCH.finditer(src)}

        fns: dict[str, FunctionFacts] = {}
        for m in _FN_DEF.finditer(src):
            name = m.group(1)
            if name in _KW:
                continue
            bs = src.find("{", m.start())
            if bs == -1:
                continue
            be = match_brace(src, bs)
            body = src[bs + 1: be] if be != -1 else ""
            start_line = src[:m.start()].count("\n") + 1
            end_line = src[:be].count("\n") + 1 if be != -1 else None
            is_kernel = name in cuda_kernel_names
            fn_extra: dict[str, str | int | float | list[str]] = {}
            if is_kernel:
                fn_extra['gpu_tags'] = ['cuda_global']
            # Record any kernel launches from this function body
            launched = [k for k in _CUDA_LAUNCH.findall(body) if k not in _KW]
            if launched:
                fn_extra['cuda_launches'] = launched
            fns[name] = FunctionFacts(
                name=name, qualified_name=f"{rel}::{name}",
                line=start_line,
                end_line=end_line,
                calls=[c for c in _CALL.findall(body) if c not in _KW],
                is_gpu_kernel=is_kernel,
                extra=fn_extra,
            )

        has_main = "main" in fns
        # File-level GPU tags
        file_gpu_tags: list[str] = []
        if cuda_kernel_names:
            file_gpu_tags.append('cuda_global')
        if cuda_launch_names:
            file_gpu_tags.append('cuda_launch')
        if path.suffix.lower() in {'.cu', '.cuh'}:
            if 'cuda_global' not in file_gpu_tags:
                file_gpu_tags.append('cuda_file')
        file_extra: dict[str, str | int | float | list[str]] = {}
        if file_gpu_tags:
            file_extra['gpu_tags'] = file_gpu_tags
        if cuda_kernel_names:
            file_extra['cuda_kernel_count'] = len(cuda_kernel_names)

        tr, inf, gen, notes = score_by_content(src, has_main, path.stem.lower())
        return FileFacts(
            path=rel, module_name=rel, language="cpp",
            has_main_guard=has_main, cli_frameworks=[],
            top_level_calls=[], main_guard_calls=["main"] if has_main else [],
            functions=fns, imports=imps,
            training_score=tr, inference_score=inf, generic_score=gen, notes=notes,
            extra=file_extra,
        )
