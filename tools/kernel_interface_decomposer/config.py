from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def _strip_outer_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in shlex.split(inner.replace(",", " "))]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value[0:1] in {"'", '"'}:
        return _strip_outer_quotes(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


class _SimpleYamlParser:
    """Small YAML subset parser for the config shape used by this tool.

    Supports nested mappings, block lists, inline scalar lists, quoted strings,
    numbers, booleans and null. It deliberately avoids arbitrary YAML features
    so the workflow does not depend on PyYAML being installed in the target env.
    """

    def __init__(self, text: str):
        self.lines: list[tuple[int, str]] = []
        for raw in text.splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            self.lines.append((indent, raw.strip()))
        self.index = 0

    def parse(self) -> Any:
        if not self.lines:
            return {}
        result = self._parse_node(self.lines[0][0])
        if self.index != len(self.lines):
            raise ConfigError(f"Could not parse config near: {self.lines[self.index][1]}")
        return result

    def _parse_node(self, indent: int) -> Any:
        if self.index >= len(self.lines):
            return {}
        line_indent, text = self.lines[self.index]
        if line_indent < indent:
            return {}
        if text.startswith("- "):
            return self._parse_list(line_indent)
        return self._parse_dict(line_indent)

    def _parse_dict(self, indent: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        while self.index < len(self.lines):
            line_indent, text = self.lines[self.index]
            if line_indent < indent or text.startswith("- "):
                break
            if line_indent > indent:
                raise ConfigError(f"Unexpected indentation near: {text}")
            if ":" not in text:
                raise ConfigError(f"Expected key/value entry near: {text}")
            key, rest = text.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            self.index += 1
            if rest:
                out[key] = _parse_scalar(rest)
            elif self.index < len(self.lines) and self.lines[self.index][0] > indent:
                out[key] = self._parse_node(self.lines[self.index][0])
            else:
                out[key] = {}
        return out

    def _parse_list(self, indent: int) -> list[Any]:
        out: list[Any] = []
        while self.index < len(self.lines):
            line_indent, text = self.lines[self.index]
            if line_indent != indent or not text.startswith("- "):
                break
            item = text[2:].strip()
            self.index += 1
            if not item:
                out.append(self._parse_node(self.lines[self.index][0]))
            elif ":" in item and not item.startswith(("'", '"')):
                key, rest = item.split(":", 1)
                value: dict[str, Any] = {key.strip(): _parse_scalar(rest.strip())}
                if self.index < len(self.lines) and self.lines[self.index][0] > indent:
                    nested = self._parse_node(self.lines[self.index][0])
                    if isinstance(nested, dict):
                        value.update(nested)
                    else:
                        value[key.strip()] = nested
                out.append(value)
            else:
                out.append(_parse_scalar(item))
        return out


def load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = _SimpleYamlParser(text).parse()
    if not isinstance(data, dict):
        raise ConfigError("Top-level config must be a mapping")
    return data


@dataclass
class DecomposerConfig:
    path: Path
    version: int
    workdir: Path
    output_dir: Path
    target_file: Path
    target_line: int
    service_cmd: str | None
    test_cmd: str | None
    ready: dict[str, Any] = field(default_factory=dict)
    stop: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)
    profiling: dict[str, Any] = field(default_factory=dict)
    resolution: dict[str, Any] = field(default_factory=dict)
    target_qualified_name: str | None = None

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "DecomposerConfig":
        config_path = Path(path).resolve()
        raw = load_mapping(config_path)
        base = config_path.parent
        workdir = Path(raw.get("workdir", base)).expanduser()
        if not workdir.is_absolute():
            workdir = (base / workdir).resolve()
        output_dir = Path(raw.get("output_dir", "kernel_decompose_out")).expanduser()
        if not output_dir.is_absolute():
            output_dir = (workdir / output_dir).resolve()

        target = raw.get("target") or {}
        if not isinstance(target, dict):
            raise ConfigError("target must be a mapping")
        target_file_raw = target.get("file")
        target_line = target.get("line")
        if not target_file_raw or target_line is None:
            raise ConfigError("target.file and target.line are required")
        target_file = Path(str(target_file_raw)).expanduser()
        if not target_file.is_absolute():
            target_file = (workdir / target_file).resolve()

        commands = raw.get("commands") or {}
        if not isinstance(commands, dict):
            raise ConfigError("commands must be a mapping")

        return cls(
            path=config_path,
            version=int(raw.get("version", 1)),
            workdir=workdir,
            output_dir=output_dir,
            target_file=target_file,
            target_line=int(target_line),
            service_cmd=commands.get("service"),
            test_cmd=commands.get("test"),
            ready=commands.get("ready") or {},
            stop=commands.get("stop") or {},
            selection=raw.get("selection") or {},
            profiling=raw.get("profiling") or {},
            resolution=raw.get("resolution") or {},
            target_qualified_name=target.get("qualified_name"),
        )

    def runtime_config_path(self) -> Path:
        return self.output_dir / "runtime_config.json"

    def nsys_output_base(self) -> Path:
        return self.output_dir / "profile"

    def nsys_rep_path(self) -> Path:
        return self.output_dir / "profile.nsys-rep"

    def sqlite_path(self) -> Path:
        return self.output_dir / "profile.sqlite"

    def schema_path(self) -> Path:
        return self.output_dir / "decomposition.schema.json"

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            "workdir": str(self.workdir),
            "output_dir": str(self.output_dir),
            "target": {
                "file": str(self.target_file),
                "line": self.target_line,
                "qualified_name": self.target_qualified_name,
            },
            "selection": self.selection,
            "resolution": self.resolution,
        }

