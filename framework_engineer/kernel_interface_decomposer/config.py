from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RUNTIME_CONFIG_VERSION = "kid-runtime-config/v2"
SAMPLING_STRATEGIES = frozenset(
    {"all", "last_n", "single", "unique_decomposition"}
)


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
    """Parse the small YAML subset accepted by the KID config CLI."""

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
            if line_indent > indent or ":" not in text:
                raise ConfigError(f"Invalid YAML near: {text}")
            key, rest = text.split(":", 1)
            self.index += 1
            if rest.strip():
                out[key.strip()] = _parse_scalar(rest.strip())
            elif self.index < len(self.lines) and self.lines[self.index][0] > indent:
                out[key.strip()] = self._parse_node(self.lines[self.index][0])
            else:
                out[key.strip()] = None
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
                out.append(value)
            else:
                out.append(_parse_scalar(item))
        return out


def load_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    try:
        data = json.loads(text) if path.suffix.lower() == ".json" else _SimpleYamlParser(text).parse()
    except (json.JSONDecodeError, IndexError) as exc:
        raise ConfigError(f"invalid config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("top-level config must be a mapping")
    return data


def _mapping(value: Any, name: str, *, nullable: bool = False) -> dict[str, Any] | None:
    if value is None and nullable:
        return None
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping" + (" or null" if nullable else ""))
    return dict(value)


def _resolve_path(value: Any, *, base: Path, name: str) -> Path:
    if value in {None, ""}:
        raise ConfigError(f"{name} is required")
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True)
class RuntimeCaptureConfig:
    path: Path
    schema_version: str
    backend_name: str
    workdir: Path
    output_dir: Path
    target_file: Path
    target_line: int
    target_qualified_name: str | None
    command: str | None
    test_command: str
    ready: dict[str, Any] | None = None
    stop: dict[str, Any] | None = None
    env: dict[str, str] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)
    profiling: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "RuntimeCaptureConfig":
        config_path = Path(path).expanduser().resolve()
        raw = load_mapping(config_path)
        version = str(raw.get("schema_version", ""))
        if version != RUNTIME_CONFIG_VERSION:
            raise ConfigError(
                f"schema_version must be {RUNTIME_CONFIG_VERSION!r}; got {version!r}"
            )

        backend_name = str(raw.get("backend_name", "")).strip()
        if not backend_name or "/" in backend_name or "\\" in backend_name:
            raise ConfigError("backend_name must be a non-empty directory-safe name")
        base = config_path.parent
        workdir = _resolve_path(raw.get("workdir", base), base=base, name="workdir")
        output_dir = _resolve_path(
            raw.get("output_dir"), base=workdir, name="output_dir"
        )
        if output_dir.name != backend_name:
            raise ConfigError(
                "output_dir must end with backend_name so config/cli_log/output/ref stay aligned"
            )

        target = _mapping(raw.get("target"), "target") or {}
        target_file = _resolve_path(
            target.get("file"), base=workdir, name="target.file"
        )
        try:
            target_line = int(target.get("line"))
        except (TypeError, ValueError) as exc:
            raise ConfigError("target.line must be a positive integer") from exc
        if target_line <= 0:
            raise ConfigError("target.line must be a positive integer")

        command_raw = raw.get("cmd")
        command = None if command_raw is None else str(command_raw).strip() or None
        test_command = str(raw.get("test_cmd", "")).strip()
        if not test_command:
            raise ConfigError("test_cmd is required")
        ready = _mapping(raw.get("ready"), "ready", nullable=True)
        stop = _mapping(raw.get("stop"), "stop", nullable=True)
        if command is None and (ready or stop):
            raise ConfigError("ready/stop are only valid when cmd is not null")

        env_raw = _mapping(raw.get("env"), "env") or {}
        env: dict[str, str] = {}
        for key, value in env_raw.items():
            if value is None:
                raise ConfigError(f"env.{key} cannot be null")
            env[str(key)] = str(value)

        selection = _mapping(raw.get("selection"), "selection") or {}
        selection = {
            "skip_invocations": int(selection.get("skip_invocations", 0)),
            "stages": list(selection.get("stages") or []),
            "sample_count_per_stage": int(selection.get("sample_count_per_stage", 1)),
            "sampling": str(selection.get("sampling", "unique_decomposition")),
            "aggregation": str(selection.get("aggregation", "single")),
        }
        if selection["skip_invocations"] < 0:
            raise ConfigError("selection.skip_invocations must be >= 0")
        if selection["sample_count_per_stage"] <= 0:
            raise ConfigError("selection.sample_count_per_stage must be > 0")
        if selection["sampling"] not in SAMPLING_STRATEGIES:
            raise ConfigError(
                "selection.sampling must be one of " + ", ".join(sorted(SAMPLING_STRATEGIES))
            )
        if selection["sampling"] == "single" and selection["sample_count_per_stage"] != 1:
            raise ConfigError("selection.sampling=single requires sample_count_per_stage=1")
        if selection["aggregation"] != "single":
            raise ConfigError("Runtime Capture v1 only supports aggregation=single")
        if not all(isinstance(item, str) and item for item in selection["stages"]):
            raise ConfigError("selection.stages must contain non-empty strings")

        profiling = _mapping(raw.get("profiling"), "profiling") or {}
        profiling = {
            **profiling,
            "nsys_bin": str(profiling.get("nsys_bin", "nsys")),
            "max_runtime_sec": float(profiling.get("max_runtime_sec", 1800)),
            "disable_cuda_graph": bool(profiling.get("disable_cuda_graph", True)),
            "min_capture_coverage": float(profiling.get("min_capture_coverage", 1.0)),
        }
        if profiling["max_runtime_sec"] <= 0:
            raise ConfigError("profiling.max_runtime_sec must be > 0")
        if not profiling["disable_cuda_graph"]:
            raise ConfigError("CUDA Graph discovery is unsupported; disable_cuda_graph must be true")
        if not 0 <= profiling["min_capture_coverage"] <= 1:
            raise ConfigError("profiling.min_capture_coverage must be in [0, 1]")

        return cls(
            path=config_path,
            schema_version=version,
            backend_name=backend_name,
            workdir=workdir,
            output_dir=output_dir,
            target_file=target_file,
            target_line=target_line,
            target_qualified_name=(
                str(target["qualified_name"]) if target.get("qualified_name") else None
            ),
            command=command,
            test_command=test_command,
            ready=ready,
            stop=stop,
            env=env,
            selection=selection,
            profiling=profiling,
        )

    def events_dir(self) -> Path:
        return self.output_dir / "capture_events"

    def trace_dir(self) -> Path:
        return self.output_dir / "trace"

    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    def sqlite_path(self) -> Path:
        return self.trace_dir() / "profile.sqlite"

    def schema_path(self) -> Path:
        return self.output_dir / "runtime_capture.schema.json"

    def runtime_config(self, *, events_dir: Path | None = None) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_name": self.backend_name,
            "workdir": str(self.workdir),
            "output_dir": str(self.output_dir),
            "target": {
                "file": str(self.target_file),
                "line": self.target_line,
                "qualified_name": self.target_qualified_name,
            },
            "events_dir": str(events_dir or self.events_dir()),
            "selection": self.selection,
        }


# Temporary import compatibility for callers being migrated in the same change.
DecomposerConfig = RuntimeCaptureConfig
