#!/usr/bin/env python3
"""Validate KID Runtime Capture outputs and complete two-stage golden workspaces."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from .config import RuntimeCaptureConfig
    from .sampling import decomposition_signature_hash
    from .semantic_resolver import SemanticResolver, SemanticResolverConfig
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import RuntimeCaptureConfig
    from sampling import decomposition_signature_hash
    from semantic_resolver import SemanticResolver, SemanticResolverConfig


RUNTIME_SCHEMA = "kid-runtime-capture/v1"
FINAL_SCHEMA = "kernel-interface-decomposition/v2"
CONFIG_RUNTIME_SCHEMA = "kid-runtime-config/v3"
CONFIG_RESOLVER_SCHEMA = "kid-semantic-resolver-config/v3"
ENVIRONMENT_SCHEMA = "kid-runtime-environment/v1"
CAPTURE_ARCHETYPES = {
    "pytorch_dispatch",
    "triton_launch",
    "cute_dsl_launch",
    "tilelang_launch",
    "tvm_ffi_call",
    "inductor_launch",
    "python_binding",
}
RUNTIME_ORACLE_FIELDS = {
    "workload_case",
    "semantic_target",
    "semantic_target_hint",
    "expected_archetype",
    "case_validation",
    "archetype_code",
    "binding_provider",
    "dry_run",
    "provider",
    "callsite",
}
FINAL_FORBIDDEN_FIELDS = {
    "archetype_code",
    "binding_provider",
    "dry_run",
    "execution_capture_id",
    "capture_id",
    "python_stack",
    "semantic_target_hint",
    "workload_case",
    "implementation",
    "source_files",
    "symbols",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_keys(value: Any, location: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            yield key, child_location
            yield from _iter_keys(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_keys(child, f"{location}[{index}]")


def _close(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False


def _parse_nvtx(label: str) -> dict[str, str]:
    if not label.startswith("KID:"):
        return {}
    fields: dict[str, str] = {}
    for item in label[4:].split("|"):
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    return fields


class RuntimeArtifactValidator:
    def __init__(self, cli_dir: Path) -> None:
        self.cli_dir = cli_dir.resolve()
        self.errors: list[str] = []
        self.runtime: dict[str, Any] = {}
        self.environment: dict[str, Any] = {}
        self.raw_events: list[dict[str, Any]] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def _require_file(self, path: Path) -> None:
        self.require(path.is_file() and path.stat().st_size > 0, f"missing or empty artifact: {path}")

    def load(self) -> bool:
        for relative in (
            "runtime_capture.schema.json",
            "environment_probe.json",
        ):
            self._require_file(self.cli_dir / relative)
        event_paths = sorted((self.cli_dir / "capture_events").glob("events_*.jsonl"))
        self.require(bool(event_paths), "no capture_events/events_<pid>.jsonl files found")
        if self.errors:
            return False
        try:
            self.runtime = _load_json(self.cli_dir / "runtime_capture.schema.json")
            self.environment = _load_json(self.cli_dir / "environment_probe.json")
            for path in event_paths:
                self.raw_events.extend(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self.errors.append(f"cannot load Runtime artifacts: {exc}")
        return not self.errors

    def _validate_no_oracles(self) -> None:
        for label, value in (
            ("runtime capture", self.runtime),
            ("environment probe", self.environment),
            ("capture events", self.raw_events),
        ):
            for key, location in _iter_keys(value):
                if key in RUNTIME_ORACLE_FIELDS:
                    self.errors.append(
                        f"{label} contains forbidden semantic/oracle field {key!r} at {location}"
                    )

    def _validate_stack(self, stack: Any, label: str) -> None:
        self.require(isinstance(stack, list) and bool(stack), f"{label} has empty python_stack")
        for index, frame in enumerate(stack or []):
            edge = frame.get("call_site_to_next")
            self.require(
                isinstance(frame.get("file"), str)
                and isinstance(frame.get("definition_line"), int)
                and isinstance(edge, dict)
                and isinstance(edge.get("file"), str)
                and isinstance(edge.get("line"), int),
                f"invalid stack frame {label}[{index}]",
            )

    def _validate_tree_and_metrics(self) -> None:
        self.require(self.runtime.get("schema_version") == RUNTIME_SCHEMA, "runtime schema_version mismatch")
        self.require(self.environment.get("schema_version") == ENVIRONMENT_SCHEMA, "environment schema_version mismatch")
        declared_sqlite = self.runtime.get("artifacts", {}).get("sqlite")
        if declared_sqlite is not None:
            self.require(
                declared_sqlite == "trace/profile.sqlite",
                f"unexpected Runtime SQLite artifact path: {declared_sqlite}",
            )
            self._require_file(self.cli_dir / str(declared_sqlite))
        raw_by_key: dict[tuple[Any, str], dict[str, Any]] = {}
        raw_by_call: Counter[str] = Counter()
        for event in self.raw_events:
            capture_id = str(event.get("capture_id"))
            key = (event.get("pid"), capture_id)
            self.require(event.get("event") == "execution_capture", f"invalid raw event type: {key}")
            self.require(capture_id not in {"", "None"}, f"invalid capture ID: {key}")
            self.require(key not in raw_by_key, f"duplicate capture key: {key}")
            raw_by_key[key] = event
            raw_by_call[str(event.get("parent_call_id", event.get("high_call_id")))] += 1
            self.require(event.get("archetype") in CAPTURE_ARCHETYPES, f"invalid raw archetype: {key}")
            self._validate_stack(event.get("python_stack"), f"raw capture {key}")
        for (pid, _), event in raw_by_key.items():
            parent = event.get("parent_capture_id")
            if parent is not None:
                self.require((pid, str(parent)) in raw_by_key, f"missing capture parent: {(pid, parent)}")
        diagnostics = self.runtime.get("diagnostics", {})
        self.require(
            diagnostics.get("capture_event_count") == len(self.raw_events),
            "diagnostics.capture_event_count mismatch",
        )
        invocations = self.runtime.get("invocations", [])
        self.require(
            diagnostics.get("selected_invocation_count") == len(invocations),
            "diagnostics.selected_invocation_count mismatch",
        )
        if self.runtime.get("invocation_selection", {}).get("sampling") == "unique_decomposition":
            groups = diagnostics.get("decomposition_groups", [])
            self.require(
                diagnostics.get("unique_decomposition_count") == len(groups),
                "diagnostics.unique_decomposition_count mismatch",
            )
            self.require(
                len(groups) == len(invocations),
                "unique decomposition groups do not match selected invocations",
            )
            selected_by_call = {
                str(item.get("high_level", {}).get("call_id")): item
                for item in invocations
            }
            grouped_call_ids: list[str] = []
            for group in groups:
                member_call_ids = [str(item) for item in group.get("member_call_ids", [])]
                representative = str(group.get("representative_call_id"))
                grouped_call_ids.extend(member_call_ids)
                self.require(
                    bool(member_call_ids) and representative == member_call_ids[-1],
                    f"invalid unique decomposition representative: {representative}",
                )
                invocation = selected_by_call.get(representative)
                self.require(
                    invocation is not None,
                    f"unique decomposition representative was not selected: {representative}",
                )
                if invocation is not None:
                    self.require(
                        group.get("signature_hash")
                        == decomposition_signature_hash(invocation),
                        f"unique decomposition signature mismatch: {representative}",
                    )
            self.require(
                Counter(grouped_call_ids)
                == Counter(
                    str(item) for item in diagnostics.get("eligible_call_ids", [])
                ),
                "unique decomposition members do not match eligible_call_ids",
            )

        kernels = self.runtime.get("kernels", [])
        kernel_by_id = {str(item.get("kernel_id")): item for item in kernels}
        self.require(len(kernel_by_id) == len(kernels), "kernel IDs are not unique")
        direct_owners: dict[str, str] = {}
        for invocation in self.runtime.get("invocations", []):
            high = invocation.get("high_level", {})
            call_id = str(high.get("call_id"))
            high_ids = set(high.get("kernel_ids", []))
            self.require(high_ids <= set(kernel_by_id), f"high {call_id} references unknown kernels")
            captures = invocation.get("execution_captures", [])
            capture_by_id = {str(item.get("capture_id")): item for item in captures}
            self.require(len(capture_by_id) == len(captures), f"duplicate materialized capture in high {call_id}")
            for capture_id, capture in capture_by_id.items():
                self.require(capture.get("archetype") in CAPTURE_ARCHETYPES, f"invalid archetype: {capture_id}")
                self._validate_stack(capture.get("python_stack"), f"capture {capture_id}")
                for kernel_id in capture.get("kernel_ids", []):
                    self.require(kernel_id not in direct_owners, f"kernel has multiple owners: {kernel_id}")
                    direct_owners[kernel_id] = capture_id
                direct_total = sum(
                    kernel_by_id[kernel_id]["duration_us"]
                    for kernel_id in capture.get("kernel_ids", [])
                    if kernel_id in kernel_by_id
                )
                inclusive_total = sum(
                    kernel_by_id[kernel_id]["duration_us"]
                    for kernel_id in capture.get("inclusive_kernel_ids", [])
                    if kernel_id in kernel_by_id
                )
                metrics = capture.get("metrics", {})
                self.require(_close(metrics.get("direct_gpu_kernel_sum_us"), direct_total), f"direct sum mismatch: {capture_id}")
                self.require(_close(metrics.get("inclusive_gpu_kernel_sum_us"), inclusive_total), f"inclusive sum mismatch: {capture_id}")
            self.require(
                invocation.get("raw_capture_event_count") == raw_by_call[call_id],
                f"raw capture count mismatch for high {call_id}",
            )
            self.require(
                invocation.get("capture_without_kernel_count")
                == invocation.get("raw_capture_event_count", 0) - len(captures),
                f"capture_without_kernel_count mismatch for high {call_id}",
            )
            high_total = sum(kernel_by_id[kernel_id]["duration_us"] for kernel_id in high_ids)
            self.require(_close(high.get("gpu_kernel_sum_us"), high_total), f"high GPU sum mismatch: {call_id}")
            unattributed = set(invocation.get("unattributed_kernel_ids", []))
            attributed_total = sum(
                kernel_by_id[kernel_id]["duration_us"] for kernel_id in high_ids - unattributed
            )
            expected_coverage = attributed_total / high_total if high_total else 0.0
            self.require(_close(invocation.get("coverage"), expected_coverage), f"coverage mismatch: {call_id}")

        self.require(
            diagnostics.get("materialized_execution_capture_count")
            == sum(
                len(item.get("execution_captures", []))
                for item in self.runtime.get("invocations", [])
            ),
            "diagnostics.materialized_execution_capture_count mismatch",
        )

        for kernel_id, kernel in kernel_by_id.items():
            self.require(kernel.get("duration_us", 0) > 0, f"non-positive duration: {kernel_id}")
            self.require(
                _close(
                    kernel.get("duration_us"),
                    (kernel.get("gpu_end_ns", 0) - kernel.get("gpu_start_ns", 0)) / 1000.0,
                ),
                f"timestamp/duration mismatch: {kernel_id}",
            )
            self.require(
                direct_owners.get(kernel_id) == kernel.get("owner_capture_id"),
                f"owner mismatch: {kernel_id}",
            )

    def _validate_sqlite(self) -> None:
        sqlite_path = self.cli_dir / "trace/profile.sqlite"
        if not sqlite_path.is_file():
            return
        try:
            connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            check = connection.execute("PRAGMA quick_check").fetchone()
            self.require(bool(check) and check[0] == "ok", f"SQLite quick_check failed: {check}")
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.require("NVTX_EVENTS" in tables, "SQLite is missing NVTX_EVENTS")
            strings: dict[int, str] = {}
            if "StringIds" in tables:
                strings = {
                    int(row[0]): str(row[1])
                    for row in connection.execute("SELECT id, value FROM StringIds")
                }
            labels = [
                row[0]
                for row in connection.execute(
                    "SELECT text FROM NVTX_EVENTS WHERE text LIKE 'KID:%' AND text IS NOT NULL"
                )
            ] if "NVTX_EVENTS" in tables else []
            parsed = [_parse_nvtx(label) for label in labels]
            sqlite_capture_ids = {
                fields.get("capture_id")
                for fields in parsed
                if fields.get("type") == "execution"
            }
            raw_capture_ids = {str(event.get("capture_id")) for event in self.raw_events}
            self.require(sqlite_capture_ids == raw_capture_ids, "SQLite execution ranges do not match raw events")
            runtime_high_ids = {
                str(item.get("high_level", {}).get("call_id"))
                for item in self.runtime.get("invocations", [])
            }
            sqlite_high_ids = {
                fields.get("call_id") for fields in parsed if fields.get("type") == "high"
            }
            self.require(runtime_high_ids <= sqlite_high_ids, "selected Runtime highs are absent from SQLite")

            kernel_tables = [
                table
                for table in tables
                if "cupti" in table.lower()
                and "kernel" in table.lower()
                and "runtime" not in table.lower()
            ]
            for kernel in self.runtime.get("kernels", []):
                matches: list[sqlite3.Row] = []
                for table in kernel_tables:
                    columns = {
                        row[1]
                        for row in connection.execute(f'PRAGMA table_info("{table}")')
                    }
                    required = {"start", "end", "correlationId"}
                    if not required <= columns:
                        continue
                    query = (
                        f'SELECT * FROM "{table}" WHERE correlationId = ? '
                        "AND start = ? AND end = ?"
                    )
                    values: list[Any] = [
                        kernel.get("correlation_id"),
                        kernel.get("gpu_start_ns"),
                        kernel.get("gpu_end_ns"),
                    ]
                    if "globalPid" in columns and kernel.get("global_pid") is not None:
                        query += " AND globalPid = ?"
                        values.append(kernel.get("global_pid"))
                    matches.extend(connection.execute(query, values).fetchall())
                kernel_id = kernel.get("kernel_id")
                self.require(len(matches) == 1, f"SQLite kernel match count is {len(matches)}: {kernel_id}")
                if len(matches) != 1:
                    continue
                row = matches[0]
                row_keys = set(row.keys())
                name_value = None
                for column in ("demangledName", "shortName", "mangledName", "name"):
                    if column in row_keys and row[column] is not None:
                        raw_name = row[column]
                        name_value = strings.get(int(raw_name), str(raw_name)) if isinstance(raw_name, int) else str(raw_name)
                        break
                self.require(name_value == kernel.get("name"), f"SQLite kernel name mismatch: {kernel_id}")
                if "deviceId" in row_keys:
                    self.require(row["deviceId"] == kernel.get("device_id"), f"SQLite device mismatch: {kernel_id}")
                if "streamId" in row_keys:
                    self.require(row["streamId"] == kernel.get("stream_id"), f"SQLite stream mismatch: {kernel_id}")

                launch = kernel.get("launch_api", {})
                launch_table = launch.get("table")
                self.require(launch_table in tables, f"SQLite launch table missing: {kernel_id}")
                if launch_table not in tables:
                    continue
                launch_rows = connection.execute(
                    f'SELECT * FROM "{launch_table}" WHERE correlationId = ? AND start = ?',
                    (kernel.get("correlation_id"), launch.get("start_ns")),
                ).fetchall()
                self.require(len(launch_rows) == 1, f"SQLite launch match count is {len(launch_rows)}: {kernel_id}")
                if len(launch_rows) == 1:
                    launch_row = launch_rows[0]
                    launch_name = None
                    if "nameId" in launch_row.keys() and launch_row["nameId"] is not None:
                        launch_name = strings.get(int(launch_row["nameId"]), str(launch_row["nameId"]))
                    elif "name" in launch_row.keys():
                        launch_name = launch_row["name"]
                    self.require(launch_name == launch.get("name"), f"SQLite launch name mismatch: {kernel_id}")
        except sqlite3.Error as exc:
            self.errors.append(f"SQLite validation failed: {exc}")
        finally:
            if "connection" in locals():
                connection.close()

    def validate(self) -> bool:
        if not self.load():
            return False
        self._validate_no_oracles()
        self._validate_tree_and_metrics()
        self._validate_sqlite()
        return not self.errors


class ArtifactValidator:
    """Validate the complete Runtime + Semantic Resolver golden workspace."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.errors: list[str] = []
        self.backend = ""
        self.runtime: dict[str, Any] = {}
        self.final: dict[str, Any] = {}
        self.raw_events: list[dict[str, Any]] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def _map_runtime_path(self, path: str, resolver: dict[str, Any]) -> str:
        mappings = resolver.get("source_context", {}).get("runtime_to_local_path_mappings", [])
        for mapping in sorted(mappings, key=lambda item: len(item["runtime_prefix"]), reverse=True):
            prefix = mapping["runtime_prefix"]
            if path == prefix or path.startswith(prefix + "/"):
                return str(Path(mapping["local_prefix"]) / path[len(prefix):].lstrip("/"))
        return path

    def validate(self) -> bool:
        config_root = self.root / "config"
        backends = sorted(item.name for item in config_root.iterdir() if item.is_dir()) if config_root.is_dir() else []
        self.require(len(backends) == 1, f"expected one backend config, found {backends}")
        if len(backends) != 1:
            return False
        self.backend = backends[0]
        backend_root = self.root / self.backend
        cli_dir = backend_root / "cli_log"
        runtime_validator = RuntimeArtifactValidator(cli_dir)
        if not runtime_validator.validate():
            self.errors.extend(runtime_validator.errors)
            return False
        self.runtime = runtime_validator.runtime
        self.raw_events = runtime_validator.raw_events

        runtime_config_path = config_root / self.backend / "runtime_capture_config.json"
        resolver_config_path = config_root / self.backend / "semantic_resolver_config.json"
        final_path = backend_root / "output" / "decomposition.schema.json"
        notes_path = backend_root / "ref" / "kid_semantic_resolver_notes.md"
        context_path = backend_root / "ref" / "semantic_resolver_context.json"
        decisions_path = backend_root / "ref" / "semantic_resolver_decisions.json"
        for path in (
            runtime_config_path,
            resolver_config_path,
            final_path,
            notes_path,
            context_path,
            decisions_path,
            self.root / "README.md",
            self.root / "ARTIFACT_GUIDE.md",
        ):
            self.require(path.is_file() and path.stat().st_size > 0, f"missing or empty golden artifact: {path}")
        if self.errors:
            return False
        try:
            runtime_config = _load_json(runtime_config_path)
            resolver = _load_json(resolver_config_path)
            self.final = _load_json(final_path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self.errors.append(f"cannot load golden JSON: {exc}")
            return False

        self.require(runtime_config.get("schema_version") == CONFIG_RUNTIME_SCHEMA, "runtime config version mismatch")
        self.require(resolver.get("schema_version") == CONFIG_RESOLVER_SCHEMA, "resolver config version mismatch")
        self.require(
            "stages" not in (runtime_config.get("selection") or {}),
            "runtime config must not contain removed selection.stages",
        )
        for field in ("ready", "stop", "env"):
            self.require(field in runtime_config, f"golden runtime config omits optional field {field}")
        for field in (
            "skip_invocations",
            "sample_count_per_stage",
            "sampling",
            "aggregation",
        ):
            self.require(
                field in (runtime_config.get("selection") or {}),
                f"golden runtime selection omits optional field {field}",
            )
        for field in (
            "nsys_bin",
            "max_runtime_sec",
            "disable_cuda_graph",
            "min_capture_coverage",
            "trace_retention",
        ):
            self.require(
                field in (runtime_config.get("profiling") or {}),
                f"golden runtime profiling omits optional field {field}",
            )
        self.require(
            set(resolver) == {"schema_version", "backend_name", "source_context"},
            "golden semantic config contains removed or unknown top-level fields",
        )
        self.require(
            set(resolver.get("source_context") or {})
            == {"third_party_manifest", "runtime_to_local_path_mappings"},
            "golden semantic source_context must explicitly contain only manifest and mappings",
        )
        target_config = runtime_config.get("target") or {}
        target_path = Path(str(target_config.get("file", "")))
        target_name = str(target_config.get("qualified_name") or "").rsplit(".", 1)[-1]
        target_line = target_config.get("line")
        self.require(target_path.is_file(), f"runtime target source does not exist: {target_path}")
        if target_path.is_file():
            try:
                tree = ast.parse(target_path.read_text(encoding="utf-8"), filename=str(target_path))
                matching_lines = {
                    node.lineno
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and (not target_name or node.name == target_name)
                }
                self.require(
                    target_line in matching_lines,
                    f"runtime target line {target_line} does not match current source {target_path}; "
                    f"candidate definition lines={sorted(matching_lines)}",
                )
            except (OSError, SyntaxError, UnicodeError) as exc:
                self.errors.append(f"cannot inspect runtime target source {target_path}: {exc}")
        for label, value in (
            ("runtime config", runtime_config.get("backend_name")),
            ("resolver config", resolver.get("backend_name")),
            ("runtime", self.runtime.get("backend_name")),
            ("final", self.final.get("backend_name")),
        ):
            self.require(value == self.backend, f"{label} backend mismatch: {value!r}")
        manifest = Path(resolver.get("source_context", {}).get("third_party_manifest", ""))
        self.require(manifest.is_file(), f"third-party manifest does not exist: {manifest}")
        self.require(not list(cli_dir.rglob("*.nsys-rep")), ".nsys-rep must not be retained")
        try:
            parsed_runtime_config = RuntimeCaptureConfig.load(runtime_config_path)
            self.require(
                parsed_runtime_config.cli_dir() == cli_dir.resolve(),
                "runtime config does not derive the golden backend cli_log directory",
            )
            resolver_config = SemanticResolverConfig.load(resolver_config_path)
            self.require(
                resolver_config.context_output == context_path.resolve(),
                "resolver context_output is outside the backend ref directory",
            )
            self.require(
                resolver_config.decisions_output == decisions_path.resolve(),
                "resolver decisions_output is outside the backend ref directory",
            )
            self.require(
                resolver_config.output == final_path.resolve(),
                "resolver output is outside the backend output directory",
            )
            self.require(
                resolver_config.notes_output == notes_path.resolve(),
                "resolver notes_output is outside the backend ref directory",
            )
            self.errors.extend(SemanticResolver(resolver_config).validate())
        except (ValueError, OSError) as exc:
            self.errors.append(f"cannot validate Semantic Resolver artifacts: {exc}")

        self.require(self.final.get("schema_version") == FINAL_SCHEMA, "final schema_version mismatch")
        mapped_target = dict(self.runtime.get("target") or {})
        if mapped_target.get("file"):
            mapped_target["file"] = self._map_runtime_path(mapped_target["file"], resolver)
        self.require(self.final.get("target") == mapped_target, "final target differs from mapped Runtime target")
        for key, location in _iter_keys(self.final):
            if key in FINAL_FORBIDDEN_FIELDS:
                self.errors.append(f"final contains forbidden field {key!r} at {location}")
        entries = self.final.get("kernels", [])
        self.require(bool(entries), "final has no semantic targets")
        self.require([item.get("rank") for item in entries] == list(range(1, len(entries) + 1)), "final ranks are not contiguous")
        durations = [item.get("metrics", {}).get("duration_us", -1) for item in entries]
        self.require(durations == sorted(durations, reverse=True), "final entries are not duration-sorted")
        runtime_names = {item.get("name") for item in self.runtime.get("kernels", [])}
        stack_edges = {
            (
                self._map_runtime_path(frame["call_site_to_next"]["file"], resolver),
                frame["call_site_to_next"]["line"],
            )
            for invocation in self.runtime.get("invocations", [])
            for capture in invocation.get("execution_captures", [])
            for frame in capture.get("python_stack", [])
        }
        for entry in entries:
            label = entry.get("low_level_id", "<unknown>")
            self.require(entry.get("archetype") in CAPTURE_ARCHETYPES, f"invalid final archetype: {label}")
            self.require(entry.get("kernel", {}).get("raw_name") in runtime_names, f"representative kernel absent from Runtime: {label}")
            call_site = entry.get("runtime_event", {}).get("call_site", {})
            self.require((call_site.get("file"), call_site.get("line")) in stack_edges, f"final call site lacks Runtime stack evidence: {label}")
            self.require(entry.get("metrics", {}).get("duration_us", 0) > 0, f"non-positive final duration: {label}")

        runtime_total = sum(
            item.get("covered_us", 0)
            for item in self.final.get("coverage_report", {}).get("per_invocation", [])
        )
        self.require(_close(sum(durations), runtime_total), "final semantic duration sum differs from Runtime")
        return not self.errors


def _runtime_cli_dir(path: Path) -> Path:
    path = path.resolve()
    if (path / "runtime_capture.schema.json").is_file():
        return path
    config_root = path / "config"
    if config_root.is_dir():
        backends = sorted(item.name for item in config_root.iterdir() if item.is_dir())
        if len(backends) == 1:
            return path / backends[0] / "cli_log"
    candidates = sorted(path.glob("*/cli_log/runtime_capture.schema.json"))
    if len(candidates) == 1:
        return candidates[0].parent
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate KID artifacts")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--runtime-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.runtime_only:
        validator: Any = RuntimeArtifactValidator(_runtime_cli_dir(args.workspace))
    else:
        validator = ArtifactValidator(args.workspace)
    if not validator.validate():
        print(f"KID artifact validation FAILED ({len(validator.errors)} errors)", file=sys.stderr)
        for error, count in Counter(validator.errors).items():
            suffix = f" (x{count})" if count > 1 else ""
            print(f"  - {error}{suffix}", file=sys.stderr)
        return 1
    if args.runtime_only:
        print(
            "KID Runtime artifact validation PASSED: "
            f"invocations={len(validator.runtime.get('invocations', []))} "
            f"raw_captures={len(validator.raw_events)} "
            f"gpu_kernels={len(validator.runtime.get('kernels', []))}"
        )
    else:
        print(
            "KID artifact validation PASSED: "
            f"backend={validator.backend} "
            f"semantic_targets={len(validator.final.get('kernels', []))} "
            f"gpu_kernels={len(validator.runtime.get('kernels', []))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
