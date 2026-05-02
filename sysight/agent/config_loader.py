"""Load LLM config from .sysight/config.yaml."""

from __future__ import annotations

from pathlib import Path

from sysight.agent.provider import LLMConfig


def _parse_yaml_simple(path: Path) -> dict:
    """Minimal YAML parser — avoids PyYAML dependency for simple config.

    Supports: nested mappings, string values, null, integers, floats.
    Does NOT support: lists, multi-line strings, anchors, tags.
    """
    text = path.read_text(encoding="utf-8")
    result: dict = {}
    # Stack of (indent, dict). Each level stores the indent of the KEY
    # that opened this dict. A new line at indent <= stack top means we
    # should pop back.
    stack: list[tuple[int, dict]] = [(-1, result)]
    current: dict = result

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Remove inline comment outside of quotes
        in_quote = False
        for i, ch in enumerate(stripped):
            if ch == '"':
                in_quote = not in_quote
            elif ch == "#" and not in_quote:
                stripped = stripped[:i].rstrip()
                break

        indent = len(line) - len(line.lstrip())

        # Pop back to a level whose indent is strictly less than current
        while indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()

            if val == "":
                # Nested mapping — children must have indent > this key's indent
                new_map: dict = {}
                current[key] = new_map
                stack.append((indent, new_map))
                current = new_map
            else:
                current[key] = _parse_scalar(val)

    return result


def _parse_scalar(val: str):
    if val == "null" or val == "":
        return None
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    if val.startswith('"') and val.endswith('"'):
        return val[1:-1]
    return val


def load_config(path: str | Path | None = None) -> dict[str, LLMConfig]:
    """Load LLM configs from YAML file.

    Returns a dict mapping stage name → LLMConfig:
      {"analyze": LLMConfig(...), "optimize": LLMConfig(...), ...}
    """
    if path is None:
        path = Path.cwd() / ".sysight" / "config.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = _parse_yaml_simple(path)
    configs: dict[str, LLMConfig] = {}

    for stage in ("analyze", "optimize", "learn", "warmup"):
        if stage in raw and isinstance(raw[stage], dict):
            d = raw[stage]
            configs[stage] = LLMConfig(
                provider=d.get("provider", ""),
                model=d.get("model", ""),
                api_key=d.get("api_key", ""),
                base_url=d.get("base_url"),
                temperature=d.get("temperature", 0),
                max_tokens=d.get("max_tokens", 4096),
            )

    return configs
