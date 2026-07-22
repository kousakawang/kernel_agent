"""Deterministic helpers for the KID Agent semantic phase.

The Agent owns semantic choices.  This module owns evidence preparation,
contract validation, metric aggregation, and final artifact publication.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from .config import ConfigError, RuntimeCaptureConfig, load_mapping
except ImportError:  # pragma: no cover - direct script/tool fallback
    from config import ConfigError, RuntimeCaptureConfig, load_mapping


RESOLVER_CONFIG_VERSION = "kid-semantic-resolver-config/v3"
CONTEXT_VERSION = "kid-semantic-context/v1"
DECISIONS_VERSION = "kid-semantic-decisions/v1"
FINAL_VERSION = "kernel-interface-decomposition/v3"
RUNTIME_VERSION = "kid-runtime-capture/v1"
SOURCE_SNAPSHOT_VERSION = "kid-source-snapshot/v1"
ATTRIBUTION_METHOD = "python_stack+execution_capture+cuda_correlation_id"
PUBLISHABLE_CONFIDENCE = frozenset({"high", "medium"})
LOW_LEVEL_ID_RE = re.compile(r"^[a-z0-9_]+$")


class SemanticResolverError(ValueError):
    """Raised when Agent decisions cannot be safely materialized."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SemanticResolverError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SemanticResolverError(f"{label} must contain a JSON object: {path}")
    return value


def _resolve_path(value: Any, *, base: Path, name: str) -> Path:
    if value in {None, ""}:
        raise ConfigError(f"{name} is required")
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_keys(value: Any, location: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            yield key, child_location
            yield from _iter_keys(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_keys(child, f"{location}[{index}]")


@dataclass(frozen=True)
class PathMapping:
    runtime_prefix: str
    local_prefix: str


@dataclass(frozen=True)
class SemanticResolverConfig:
    path: Path
    backend_name: str
    runtime_config: RuntimeCaptureConfig
    output_root: Path
    runtime_capture: Path
    third_party_manifest: Path
    path_mappings: tuple[PathMapping, ...]
    context_output: Path
    decisions_output: Path
    output: Path
    notes_output: Path

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "SemanticResolverConfig":
        config_path = Path(path).expanduser().resolve()
        raw = load_mapping(config_path)
        if raw.get("schema_version") != RESOLVER_CONFIG_VERSION:
            raise ConfigError(
                f"schema_version must be {RESOLVER_CONFIG_VERSION!r}; "
                f"got {raw.get('schema_version')!r}"
            )
        backend = str(raw.get("backend_name", "")).strip()
        if not backend or "/" in backend or "\\" in backend:
            raise ConfigError("backend_name must be a non-empty directory-safe name")
        base = config_path.parent
        allowed_top_level = {"schema_version", "backend_name", "source_context"}
        removed_top_level = sorted(set(raw) - allowed_top_level)
        if removed_top_level:
            raise ConfigError(
                "unsupported kid-semantic-resolver-config/v3 fields: "
                + ", ".join(removed_top_level)
            )
        source_context = raw.get("source_context") or {}
        if not isinstance(source_context, dict):
            raise ConfigError("source_context must be a mapping")
        allowed_source_fields = {
            "third_party_manifest",
            "runtime_to_local_path_mappings",
        }
        removed_source_fields = sorted(set(source_context) - allowed_source_fields)
        if removed_source_fields:
            raise ConfigError(
                "unsupported source_context fields: "
                + ", ".join(removed_source_fields)
            )
        manifest = _resolve_path(
            source_context.get("third_party_manifest"),
            base=base,
            name="source_context.third_party_manifest",
        )
        mappings_raw = source_context.get("runtime_to_local_path_mappings") or []
        if not isinstance(mappings_raw, list):
            raise ConfigError(
                "source_context.runtime_to_local_path_mappings must be a list"
            )
        mappings: list[PathMapping] = []
        for index, item in enumerate(mappings_raw):
            if not isinstance(item, dict):
                raise ConfigError(f"path mapping {index} must be a mapping")
            runtime_prefix = str(item.get("runtime_prefix", "")).rstrip("/")
            local_prefix = str(item.get("local_prefix", "")).rstrip("/")
            if not runtime_prefix or not local_prefix:
                raise ConfigError(f"path mapping {index} requires both prefixes")
            mappings.append(PathMapping(runtime_prefix, local_prefix))
        mappings.sort(key=lambda item: len(item.runtime_prefix), reverse=True)
        runtime_config_path = base / "runtime_capture_config.json"
        if not runtime_config_path.is_file():
            raise ConfigError(
                "semantic config requires sibling runtime_capture_config.json: "
                f"{runtime_config_path}"
            )
        runtime_config = RuntimeCaptureConfig.load(runtime_config_path)
        if runtime_config.backend_name != backend:
            raise ConfigError(
                "semantic/runtime backend_name mismatch: "
                f"{backend!r} != {runtime_config.backend_name!r}"
            )

        def map_path(value: str | Path) -> Path:
            normalized = str(value).rstrip("/")
            for mapping in mappings:
                prefix = mapping.runtime_prefix
                if normalized == prefix or normalized.startswith(prefix + "/"):
                    suffix = normalized[len(prefix) :].lstrip("/")
                    return (Path(mapping.local_prefix) / suffix).resolve()
            return Path(value).expanduser().resolve()

        output_root = map_path(runtime_config.output_dir)
        backend_root = output_root / backend
        return cls(
            path=config_path,
            backend_name=backend,
            runtime_config=runtime_config,
            output_root=output_root,
            runtime_capture=backend_root / "cli_log" / "runtime_capture.schema.json",
            third_party_manifest=manifest,
            path_mappings=tuple(mappings),
            context_output=backend_root / "ref" / "semantic_resolver_context.json",
            decisions_output=backend_root / "ref" / "semantic_resolver_decisions.json",
            output=backend_root / "output" / "decomposition.schema.json",
            notes_output=backend_root / "ref" / "kid_semantic_resolver_notes.md",
        )

    def map_runtime_path(self, value: str) -> str:
        normalized = str(value).rstrip("/")
        for mapping in self.path_mappings:
            prefix = mapping.runtime_prefix
            if normalized == prefix or normalized.startswith(prefix + "/"):
                suffix = normalized[len(prefix) :].lstrip("/")
                return str(Path(mapping.local_prefix) / suffix)
        return str(value)


class _SourceInspector:
    def __init__(self, config: SemanticResolverConfig) -> None:
        self.config = config
        self._text_cache: dict[Path, str | None] = {}
        self._ast_cache: dict[Path, ast.AST | None] = {}
        self._snapshot_cache: dict[Path, dict[str, Any]] = {}
        self.repo_hints = self._load_repo_hints()

    def _load_repo_hints(self) -> list[dict[str, Any]]:
        hints: list[dict[str, Any]] = []
        if self.config.third_party_manifest.is_file():
            manifest = _load_json(self.config.third_party_manifest, "third-party manifest")
            sglang_root = manifest.get("sglang_repo_root")
            if sglang_root:
                mapped_root = Path(self.config.map_runtime_path(str(sglang_root)))
                hints.append(
                    {
                        "name": "sglang",
                        "local_path": str(mapped_root),
                        "source": "third_party_manifest",
                        "available": mapped_root.exists(),
                    }
                )
            for repo in manifest.get("repos", []):
                if isinstance(repo, dict) and repo.get("local_path"):
                    mapped_path = Path(
                        self.config.map_runtime_path(str(repo.get("local_path")))
                    )
                    hints.append(
                        {
                            "name": repo.get("name"),
                            "local_path": str(mapped_path),
                            "source": "third_party_manifest",
                            "available": mapped_path.exists(),
                            "status": repo.get("status"),
                        }
                    )
        unique: dict[tuple[str, str], dict[str, Any]] = {}
        for hint in hints:
            unique[(str(hint.get("name")), hint["local_path"])] = hint
        return sorted(unique.values(), key=lambda item: (str(item.get("name")), item["local_path"]))

    def _repository_hint(self, published_path: str) -> str | None:
        candidates: list[tuple[int, str]] = []
        for hint in self.repo_hints:
            root = hint["local_path"].rstrip("/")
            if published_path == root or published_path.startswith(root + "/"):
                candidates.append((len(root), str(hint.get("name"))))
        return max(candidates)[1] if candidates else None

    def _snapshot_line(
        self, path: Path, line: int, published_file: str
    ) -> dict[str, Any] | None:
        if path not in self._snapshot_cache:
            snapshot = _load_json(path, "source snapshot")
            if snapshot.get("schema_version") != SOURCE_SNAPSHOT_VERSION:
                raise SemanticResolverError(f"unsupported source snapshot: {path}")
            self._snapshot_cache[path] = snapshot
        snapshot = self._snapshot_cache[path]
        if snapshot.get("published_file") != published_file:
            raise SemanticResolverError(
                f"source snapshot published_file mismatch: {path}"
            )
        value = snapshot.get("lines", {}).get(str(line))
        return value if isinstance(value, dict) else None

    def _text(self, path: Path) -> str | None:
        if path not in self._text_cache:
            try:
                self._text_cache[path] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                self._text_cache[path] = None
        return self._text_cache[path]

    def _call_expression(self, path: Path, line: int, text: str) -> str | None:
        if path not in self._ast_cache:
            try:
                self._ast_cache[path] = ast.parse(text, filename=str(path))
            except (SyntaxError, ValueError):
                self._ast_cache[path] = None
        tree = self._ast_cache[path]
        if tree is None:
            return None
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and node.lineno <= line <= getattr(node, "end_lineno", node.lineno)
        ]
        if not calls:
            return None
        calls.sort(
            key=lambda node: (
                getattr(node, "end_lineno", node.lineno) - node.lineno,
                getattr(node, "end_col_offset", 0) - node.col_offset,
            )
        )
        segment = ast.get_source_segment(text, calls[0])
        return " ".join(segment.split()) if segment else None

    def inspect_edge(
        self,
        frame: dict[str, Any],
        next_frame: dict[str, Any] | None,
        execution_interface: str,
        edge_index: int,
    ) -> dict[str, Any]:
        edge = frame.get("call_site_to_next") or {}
        runtime_file = str(edge.get("file", ""))
        line = int(edge.get("line", 0))
        published_file = self.config.map_runtime_path(runtime_file)
        analysis_path = Path(published_file)
        source_excerpt: str | None = None
        call_expression: str | None = None
        analysis_file: str | None = str(analysis_path) if analysis_path.exists() else None
        if analysis_path.exists() and analysis_path.suffix == ".json":
            record = self._snapshot_line(analysis_path, line, published_file)
            if record:
                source_excerpt = record.get("source")
                call_expression = record.get("call_expression")
        elif analysis_path.is_file():
            text = self._text(analysis_path)
            if text is not None:
                lines = text.splitlines()
                start = max(1, line - 1)
                end = min(len(lines), line + 1)
                source_excerpt = "\n".join(
                    f"{number}: {lines[number - 1]}" for number in range(start, end + 1)
                )
                call_expression = self._call_expression(analysis_path, line, text)
        return {
            "edge_index": edge_index,
            "caller": {
                "function": frame.get("function"),
                "qualname": frame.get("qualname"),
                "definition_file": self.config.map_runtime_path(str(frame.get("file", ""))),
                "definition_line": frame.get("definition_line"),
            },
            "callee": {
                "function": next_frame.get("function") if next_frame else execution_interface,
                "qualname": next_frame.get("qualname") if next_frame else execution_interface,
            },
            "runtime_call_site": {"file": runtime_file, "line": line},
            "call_site": {"file": published_file, "line": line},
            "analysis_file": analysis_file,
            "source_excerpt": source_excerpt,
            "call_expression": call_expression,
            "repository_hint": self._repository_hint(published_file),
        }


class SemanticResolver:
    def __init__(self, config: SemanticResolverConfig) -> None:
        self.config = config

    def _runtime(self) -> dict[str, Any]:
        runtime = _load_json(self.config.runtime_capture, "Runtime capture")
        if runtime.get("schema_version") != RUNTIME_VERSION:
            raise SemanticResolverError("Runtime capture schema_version mismatch")
        if runtime.get("backend_name") != self.config.backend_name:
            raise SemanticResolverError("Runtime capture backend_name mismatch")
        return runtime

    def _required_input_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.config.third_party_manifest.is_file():
            errors.append(
                f"third-party manifest is missing: {self.config.third_party_manifest}"
            )
        return errors

    @staticmethod
    def _owner_captures(runtime: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
        owners: dict[tuple[str, str], dict[str, Any]] = {}
        for invocation in runtime.get("invocations", []):
            call_id = str(invocation.get("high_level", {}).get("call_id"))
            for capture in invocation.get("execution_captures", []):
                if capture.get("kernel_ids"):
                    key = (call_id, str(capture.get("capture_id")))
                    if key in owners:
                        raise SemanticResolverError(f"duplicate owner capture: {key}")
                    owners[key] = capture
        return owners

    def prepare_context(self) -> dict[str, Any]:
        input_errors = self._required_input_errors()
        if input_errors:
            raise SemanticResolverError("; ".join(input_errors))
        runtime = self._runtime()
        inspector = _SourceInspector(self.config)
        kernel_by_id = {
            str(kernel.get("kernel_id")): kernel for kernel in runtime.get("kernels", [])
        }
        invocations: list[dict[str, Any]] = []
        for invocation in runtime.get("invocations", []):
            high = invocation.get("high_level", {})
            call_id = str(high.get("call_id"))
            captures = invocation.get("execution_captures", [])
            capture_by_id = {str(item.get("capture_id")): item for item in captures}
            owner_contexts: list[dict[str, Any]] = []
            for capture in captures:
                if not capture.get("kernel_ids"):
                    continue
                ancestors: list[dict[str, Any]] = []
                parent_id = capture.get("parent_capture_id")
                while parent_id is not None:
                    parent = capture_by_id.get(str(parent_id))
                    if parent is None:
                        raise SemanticResolverError(
                            f"capture {capture.get('capture_id')} has missing parent {parent_id}"
                        )
                    ancestors.append(
                        {
                            "capture_id": str(parent.get("capture_id")),
                            "archetype": parent.get("archetype"),
                            "common_interface": parent.get("common_interface"),
                            "execution_interface": parent.get("execution_interface"),
                        }
                    )
                    parent_id = parent.get("parent_capture_id")
                ancestors.reverse()
                stack = capture.get("python_stack") or []
                edges = [
                    inspector.inspect_edge(
                        frame,
                        stack[index + 1] if index + 1 < len(stack) else None,
                        str(capture.get("execution_interface")),
                        index,
                    )
                    for index, frame in enumerate(stack)
                ]
                owner_contexts.append(
                    {
                        "member_ref": {
                            "call_id": call_id,
                            "capture_id": str(capture.get("capture_id")),
                        },
                        "assignable": True,
                        "parent_capture_id": capture.get("parent_capture_id"),
                        "nested_depth": len(ancestors),
                        "ancestor_chain": ancestors,
                        "archetype": capture.get("archetype"),
                        "common_interface": capture.get("common_interface"),
                        "execution_interface": capture.get("execution_interface"),
                        "execution_leaf": capture.get("execution_leaf"),
                        "provider_hint": capture.get("provider_hint"),
                        "implementation_hint": capture.get("implementation_hint"),
                        "direct_gpu_kernel_sum_us": capture.get("metrics", {}).get(
                            "direct_gpu_kernel_sum_us"
                        ),
                        "kernels": [
                            kernel_by_id[kernel_id]
                            for kernel_id in capture.get("kernel_ids", [])
                            if kernel_id in kernel_by_id
                        ],
                        "stack_edges": edges,
                    }
                )
            invocations.append(
                {
                    "call_id": call_id,
                    "stage": high.get("stage"),
                    "total_gpu_us": high.get("gpu_kernel_sum_us"),
                    "runtime_coverage": invocation.get("coverage"),
                    "unattributed_kernel_ids": invocation.get(
                        "unattributed_kernel_ids", []
                    ),
                    "owner_captures": owner_contexts,
                }
            )
        target = dict(runtime.get("target") or {})
        if target.get("file"):
            target["file"] = self.config.map_runtime_path(str(target["file"]))
        return {
            "schema_version": CONTEXT_VERSION,
            "backend_name": self.config.backend_name,
            "runtime_capture": str(self.config.runtime_capture),
            "runtime_sha256": _sha256(self.config.runtime_capture),
            "target": target,
            "repository_hints": inspector.repo_hints,
            "invocations": invocations,
        }

    def prepare(self) -> dict[str, Any]:
        context = self.prepare_context()
        _atomic_json(self.config.context_output, context)
        return context

    def _validate_context(
        self, context: dict[str, Any], runtime: dict[str, Any] | None = None
    ) -> list[str]:
        errors: list[str] = []
        if context.get("schema_version") != CONTEXT_VERSION:
            errors.append("semantic context schema_version mismatch")
        if context.get("backend_name") != self.config.backend_name:
            errors.append("semantic context backend_name mismatch")
        if context.get("runtime_sha256") != _sha256(self.config.runtime_capture):
            errors.append("semantic context was prepared from a different Runtime capture")
        for key, location in _iter_keys(context):
            if key in {"low_level_id", "normalized_kernel_name", "semantic_target"}:
                errors.append(f"semantic context contains decision oracle {key!r} at {location}")
        runtime = runtime or self._runtime()
        expected_refs = set(self._owner_captures(runtime))
        context_refs = {
            (
                str(capture.get("member_ref", {}).get("call_id")),
                str(capture.get("member_ref", {}).get("capture_id")),
            )
            for invocation in context.get("invocations", [])
            for capture in invocation.get("owner_captures", [])
            if capture.get("assignable") is True
        }
        if context_refs != expected_refs:
            errors.append("semantic context direct-owner references differ from Runtime")
        return errors

    def _validate_decisions(
        self, runtime: dict[str, Any], decisions: dict[str, Any], notes: str
    ) -> tuple[list[str], dict[tuple[str, str], dict[str, Any]]]:
        errors: list[str] = []
        owners = self._owner_captures(runtime)
        if decisions.get("schema_version") != DECISIONS_VERSION:
            errors.append("semantic decisions schema_version mismatch")
        if decisions.get("backend_name") != self.config.backend_name:
            errors.append("semantic decisions backend_name mismatch")
        if set(decisions) != {"schema_version", "backend_name", "targets"}:
            errors.append("semantic decisions top-level fields do not match the contract")
        targets = decisions.get("targets")
        if not isinstance(targets, list) or not targets:
            errors.append("semantic decisions targets must be a non-empty list")
            targets = []
        allowed_target_keys = {
            "low_level_id",
            "interface",
            "provider",
            "normalized_kernel_name",
            "confidence",
            "members",
        }
        seen_ids: set[str] = set()
        seen_interfaces: set[str] = set()
        assigned: dict[tuple[str, str], dict[str, Any]] = {}
        for index, target in enumerate(targets):
            label = f"decisions.targets[{index}]"
            if not isinstance(target, dict):
                errors.append(f"{label} must be a mapping")
                continue
            extra = set(target) - allowed_target_keys
            if extra:
                errors.append(f"{label} contains non-contract fields: {sorted(extra)}")
            low_id = str(target.get("low_level_id", ""))
            interface = str(target.get("interface", ""))
            if not LOW_LEVEL_ID_RE.fullmatch(low_id):
                errors.append(f"invalid low_level_id: {low_id!r}")
            if low_id in seen_ids:
                errors.append(f"duplicate low_level_id: {low_id}")
            seen_ids.add(low_id)
            if not interface:
                errors.append(f"{label}.interface is required")
            if interface in seen_interfaces:
                errors.append(f"duplicate semantic interface/provider conflict: {interface}")
            seen_interfaces.add(interface)
            provider = target.get("provider")
            if provider is not None and (not isinstance(provider, str) or not provider.strip()):
                errors.append(f"{label}.provider must be a non-empty string or null")
            if not isinstance(target.get("normalized_kernel_name"), str) or not target.get(
                "normalized_kernel_name", ""
            ).strip():
                errors.append(f"{label}.normalized_kernel_name is required")
            if target.get("confidence") not in PUBLISHABLE_CONFIDENCE:
                errors.append(f"{label}.confidence is not publishable")
            members = target.get("members") or []
            if not isinstance(members, list) or not members:
                errors.append(f"{label}.members must be a non-empty list")
                continue
            archetypes: set[str] = set()
            for member_index, member in enumerate(members):
                member_label = f"{label}.members[{member_index}]"
                if not isinstance(member, dict) or set(member) != {
                    "call_id",
                    "capture_id",
                    "semantic_call_site",
                }:
                    errors.append(f"{member_label} has invalid fields")
                    continue
                key = (str(member.get("call_id")), str(member.get("capture_id")))
                capture = owners.get(key)
                if capture is None:
                    errors.append(f"{member_label} does not reference a direct kernel owner: {key}")
                    continue
                if key in assigned:
                    errors.append(f"direct kernel owner assigned more than once: {key}")
                    continue
                assigned[key] = target
                archetypes.add(str(capture.get("archetype")))
                call_site = member.get("semantic_call_site") or {}
                candidate_edges = {
                    (
                        self.config.map_runtime_path(
                            str(frame.get("call_site_to_next", {}).get("file", ""))
                        ),
                        frame.get("call_site_to_next", {}).get("line"),
                    )
                    for frame in capture.get("python_stack", [])
                }
                selected_edge = (call_site.get("file"), call_site.get("line"))
                if selected_edge not in candidate_edges:
                    errors.append(
                        f"{member_label}.semantic_call_site lacks Runtime stack evidence: {selected_edge}"
                    )
            if len(archetypes) > 1 and interface not in notes:
                errors.append(
                    f"mixed-archetype target must be disclosed in notes: {interface}"
                )
        missing = set(owners) - set(assigned)
        if missing:
            errors.append(f"direct kernel owners are unassigned: {sorted(missing)}")
        return errors, assigned

    def build_final(self) -> dict[str, Any]:
        runtime = self._runtime()
        decisions = _load_json(self.config.decisions_output, "semantic decisions")
        try:
            notes = self.config.notes_output.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise SemanticResolverError(f"cannot read resolver notes: {exc}") from exc
        if not notes.strip():
            raise SemanticResolverError("resolver notes must be non-empty")
        errors, _ = self._validate_decisions(runtime, decisions, notes)
        if errors:
            raise SemanticResolverError("; ".join(errors))

        invocations = runtime.get("invocations", [])
        capture_by_key = {
            (str(invocation.get("high_level", {}).get("call_id")), str(capture.get("capture_id"))): capture
            for invocation in invocations
            for capture in invocation.get("execution_captures", [])
        }
        kernel_by_id = {
            str(kernel.get("kernel_id")): kernel for kernel in runtime.get("kernels", [])
        }
        kernel_order = {kernel_id: index for index, kernel_id in enumerate(kernel_by_id)}
        invocation_totals = {
            str(item.get("high_level", {}).get("call_id")): sum(
                sorted(
                    float(kernel_by_id[str(kernel_id)].get("duration_us", 0.0))
                    for kernel_id in item.get("high_level", {}).get("kernel_ids", [])
                )
            )
            for item in invocations
        }
        total_gpu = sum(sorted(invocation_totals.values()))
        entries: list[dict[str, Any]] = []
        assigned_by_call: dict[str, set[str]] = {
            str(item.get("high_level", {}).get("call_id")): set() for item in invocations
        }
        for target in decisions.get("targets", []):
            contributions: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
            member_calls: set[str] = set()
            for member in target["members"]:
                call_id = str(member["call_id"])
                capture = capture_by_key[(call_id, str(member["capture_id"]))]
                member_calls.add(call_id)
                for kernel_id in capture.get("kernel_ids", []):
                    kernel = kernel_by_id[str(kernel_id)]
                    contributions.append((kernel, capture, member))
                    assigned_by_call[call_id].add(str(kernel_id))
            hottest_kernel, hottest_capture, hottest_member = max(
                contributions,
                key=lambda item: (
                    float(item[0].get("duration_us", 0.0)),
                    -kernel_order[str(item[0].get("kernel_id"))],
                ),
            )
            duration = round(
                sum(float(item[0].get("duration_us", 0.0)) for item in contributions),
                9,
            )
            entries.append(
                {
                    "rank": 0,
                    "low_level_id": target["low_level_id"],
                    "kernel": {
                        "raw_name": hottest_kernel.get("name"),
                        "normalized_name": target["normalized_kernel_name"],
                    },
                    "interface": target["interface"],
                    "archetype": hottest_capture.get("archetype"),
                    "provider": target.get("provider"),
                    "metrics": {
                        "duration_us": duration,
                        "share_in_invocation": duration / total_gpu if total_gpu else 0.0,
                    },
                    "measurement": {
                        "metric": "gpu_kernel_duration_us",
                        "aggregation": "sum",
                        "sample_count": len(member_calls),
                    },
                    "runtime_event": {
                        "call_site": hottest_member["semantic_call_site"],
                        "attribution": {
                            "method": ATTRIBUTION_METHOD,
                            "confidence": target["confidence"],
                        },
                    },
                }
            )
        entries.sort(key=lambda item: (-item["metrics"]["duration_us"], item["low_level_id"]))
        for rank, entry in enumerate(entries, 1):
            entry["rank"] = rank

        coverage_rows: list[dict[str, Any]] = []
        for invocation in invocations:
            high = invocation.get("high_level", {})
            call_id = str(high.get("call_id"))
            total = invocation_totals[call_id]
            covered = round(
                sum(
                    float(kernel_by_id[kernel_id].get("duration_us", 0.0))
                    for kernel_id in assigned_by_call[call_id]
                ),
                9,
            )
            high_kernel_ids = {
                str(kernel_id) for kernel_id in high.get("kernel_ids", [])
            }
            coverage = (
                1.0
                if assigned_by_call[call_id] == high_kernel_ids
                else (covered / total if total else 0.0)
            )
            coverage_rows.append(
                {
                    "call_id": call_id,
                    "stage": high.get("stage"),
                    "covered_us": covered,
                    "total_gpu_us": total,
                    "coverage": coverage,
                    "unattributed_kernel_ids": invocation.get(
                        "unattributed_kernel_ids", []
                    ),
                }
            )
        target = dict(runtime.get("target") or {})
        if target.get("file"):
            target["file"] = self.config.map_runtime_path(str(target["file"]))
        min_coverage = min((row["coverage"] for row in coverage_rows), default=0.0)
        selected_kernel_ids = {
            str(kernel_id)
            for invocation in invocations
            for kernel_id in invocation.get("high_level", {}).get("kernel_ids", [])
        }
        return {
            "schema_version": FINAL_VERSION,
            "backend_name": self.config.backend_name,
            "target": target,
            "coverage_report": {
                "per_invocation": coverage_rows,
                "min_coverage": min_coverage,
                "semantic_target_count": len(entries),
                "gpu_kernel_count": len(selected_kernel_ids),
                "uncaptured_hint": None
                if math.isclose(min_coverage, 1.0, rel_tol=1e-9, abs_tol=1e-9)
                else "Inspect Runtime unattributed_kernel_ids before source location.",
            },
            "kernels": entries,
        }

    def finalize(self) -> dict[str, Any]:
        input_errors = self._required_input_errors()
        if input_errors:
            raise SemanticResolverError("; ".join(input_errors))
        if not self.config.context_output.is_file():
            raise SemanticResolverError("semantic context is missing; run prepare first")
        context = _load_json(self.config.context_output, "semantic context")
        context_errors = self._validate_context(context, self._runtime())
        if context_errors:
            raise SemanticResolverError("; ".join(context_errors))
        final = self.build_final()
        _atomic_json(self.config.output, final)
        return final

    def validate(self) -> list[str]:
        errors: list[str] = []
        for label, path in (
            ("Runtime capture", self.config.runtime_capture),
            ("third-party manifest", self.config.third_party_manifest),
            ("semantic context", self.config.context_output),
            ("semantic decisions", self.config.decisions_output),
            ("resolver notes", self.config.notes_output),
            ("final decomposition", self.config.output),
        ):
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing or empty {label}: {path}")
        if errors:
            return errors
        try:
            context = _load_json(self.config.context_output, "semantic context")
            runtime = self._runtime()
            errors.extend(self._validate_context(context, runtime))
            expected = self.build_final()
            actual = _load_json(self.config.output, "final decomposition")
            if actual != expected:
                errors.append("final decomposition differs from deterministic materialization")
            forbidden = {
                "implementation",
                "source_files",
                "symbols",
                "source_locations",
                "kernel_sources_dir",
                "capture_id",
                "python_stack",
                "execution_capture_id",
                "semantic_target_hint",
                "workload_case",
                "alternatives",
                "reason",
            }
            for key, location in _iter_keys(actual):
                if key in forbidden:
                    errors.append(f"final decomposition contains forbidden field {key!r} at {location}")
        except (SemanticResolverError, OSError) as exc:
            errors.append(str(exc))
        return errors
