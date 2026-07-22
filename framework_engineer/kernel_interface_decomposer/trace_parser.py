from __future__ import annotations

import json
import sqlite3
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import RuntimeCaptureConfig
from .sampling import select_invocations


RUNTIME_CAPTURE_VERSION = "kid-runtime-capture/v1"
LABEL_PREFIX = "KID:"


@dataclass(frozen=True)
class NvtxRange:
    start: int
    end: int
    fields: dict[str, str]
    global_tid: int | None
    global_pid: int | None

    @property
    def duration_us(self) -> float:
        return max(0, self.end - self.start) / 1000.0

    @property
    def process_id(self) -> int | None:
        return _os_pid(self.global_pid if self.global_pid is not None else self.global_tid)


@dataclass(frozen=True)
class ApiEvent:
    start: int
    end: int
    correlation_id: int
    name: str | None
    global_tid: int | None
    global_pid: int | None
    source_table: str

    @property
    def process_id(self) -> int | None:
        return _os_pid(self.global_pid if self.global_pid is not None else self.global_tid)


@dataclass(frozen=True)
class KernelEvent:
    start: int
    end: int
    correlation_id: int
    name: str
    global_pid: int | None
    device_id: int | None
    stream_id: int | None
    source_table: str

    @property
    def duration_us(self) -> float:
        return max(0, self.end - self.start) / 1000.0

    @property
    def process_id(self) -> int | None:
        return _os_pid(self.global_pid)


def _os_pid(value: int | None) -> int | None:
    if value is None:
        return None
    # Nsight encodes globalPid/globalTid as pid << 24 (plus tid for globalTid).
    return (value >> 24) & 0xFFFFFF if value >= (1 << 24) else value


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    ]


def _columns(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    return {
        str(row[1]).lower(): str(row[1])
        for row in conn.execute(f'PRAGMA table_info("{table}")')
    }


def _first(columns: dict[str, str], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        value = columns.get(candidate.lower())
        if value is not None:
            return value
    return None


def _row_int(row: sqlite3.Row, candidates: Iterable[str]) -> int | None:
    keys = {key.lower(): key for key in row.keys()}
    for candidate in candidates:
        actual = keys.get(candidate.lower())
        if actual is not None and row[actual] is not None:
            return int(row[actual])
    return None


def _load_string_ids(conn: sqlite3.Connection) -> dict[int, str]:
    strings: dict[int, str] = {}
    for table in _tables(conn):
        if table.lower() not in {"stringids", "string_ids"}:
            continue
        columns = _columns(conn, table)
        id_column = _first(columns, ("id", "stringId", "string_id"))
        value_column = _first(columns, ("value", "string", "text"))
        if id_column is None or value_column is None:
            continue
        for row in conn.execute(f'SELECT * FROM "{table}"'):
            if row[id_column] is not None and row[value_column] is not None:
                strings[int(row[id_column])] = str(row[value_column])
    return strings


def _row_text(row: sqlite3.Row, strings: dict[int, str]) -> str | None:
    keys = {key.lower(): key for key in row.keys()}
    for candidate in ("text", "jsonText", "name", "message"):
        actual = keys.get(candidate.lower())
        if actual is not None and row[actual] is not None:
            return str(row[actual])
    for candidate in ("textId", "jsonTextId", "nameId"):
        actual = keys.get(candidate.lower())
        if actual is not None and row[actual] is not None:
            return strings.get(int(row[actual]), str(row[actual]))
    return None


def _parse_label(text: str) -> dict[str, str]:
    position = text.find(LABEL_PREFIX)
    if position < 0:
        return {}
    fields: dict[str, str] = {}
    for component in text[position + len(LABEL_PREFIX) :].split("|"):
        if "=" not in component:
            continue
        key, value = component.split("=", 1)
        fields[key] = urllib.parse.unquote(value)
    return fields


def _load_nvtx_ranges(
    conn: sqlite3.Connection, strings: dict[int, str]
) -> tuple[list[NvtxRange], list[str]]:
    ranges: list[NvtxRange] = []
    used_tables: list[str] = []
    for table in _tables(conn):
        if "nvtx" not in table.lower():
            continue
        columns = _columns(conn, table)
        start_column = _first(columns, ("start", "startNs", "startTime"))
        end_column = _first(columns, ("end", "endNs", "endTime"))
        if start_column is None or end_column is None:
            continue
        used = False
        for row in conn.execute(f'SELECT * FROM "{table}"'):
            if row[start_column] is None or row[end_column] is None:
                continue
            text = _row_text(row, strings)
            if not text or LABEL_PREFIX not in text:
                continue
            fields = _parse_label(text)
            if fields.get("type") not in {"high", "execution"}:
                continue
            ranges.append(
                NvtxRange(
                    start=int(row[start_column]),
                    end=int(row[end_column]),
                    fields=fields,
                    global_tid=_row_int(row, ("globalTid", "tid", "threadId")),
                    global_pid=_row_int(row, ("globalPid", "pid", "processId")),
                )
            )
            used = True
        if used:
            used_tables.append(table)
    ranges.sort(key=lambda item: (item.start, item.end))
    return ranges, used_tables


def _load_api_events(
    conn: sqlite3.Connection, strings: dict[int, str]
) -> tuple[list[ApiEvent], list[str]]:
    events: list[ApiEvent] = []
    used_tables: list[str] = []
    for table in _tables(conn):
        lowered = table.lower()
        if "cupti" not in lowered or not ("runtime" in lowered or "driver" in lowered):
            continue
        columns = _columns(conn, table)
        start_column = _first(columns, ("start", "startNs", "startTime"))
        end_column = _first(columns, ("end", "endNs", "endTime"))
        correlation_column = _first(columns, ("correlationId", "correlation_id"))
        if start_column is None or end_column is None or correlation_column is None:
            continue
        used = False
        for row in conn.execute(f'SELECT * FROM "{table}"'):
            if row[start_column] is None or row[correlation_column] is None:
                continue
            end = row[end_column]
            events.append(
                ApiEvent(
                    start=int(row[start_column]),
                    end=int(end) if end is not None else int(row[start_column]),
                    correlation_id=int(row[correlation_column]),
                    name=_row_text(row, strings),
                    global_tid=_row_int(row, ("globalTid", "tid", "threadId")),
                    global_pid=_row_int(row, ("globalPid", "pid", "processId")),
                    source_table=table,
                )
            )
            used = True
        if used:
            used_tables.append(table)
    events.sort(key=lambda item: item.start)
    return events, used_tables


def _kernel_name(row: sqlite3.Row, strings: dict[int, str]) -> str | None:
    keys = {key.lower(): key for key in row.keys()}
    for candidate in (
        "demangledName",
        "shortName",
        "mangledName",
        "name",
        "demangledNameId",
        "shortNameId",
        "mangledNameId",
        "nameId",
    ):
        actual = keys.get(candidate.lower())
        if actual is None or row[actual] is None:
            continue
        value = row[actual]
        return strings.get(int(value), str(value)) if isinstance(value, int) else str(value)
    return None


def _load_kernel_events(
    conn: sqlite3.Connection, strings: dict[int, str]
) -> tuple[list[KernelEvent], list[str]]:
    events: list[KernelEvent] = []
    used_tables: list[str] = []
    seen: set[tuple[Any, ...]] = set()
    for table in _tables(conn):
        lowered = table.lower()
        if "cupti" not in lowered or "kernel" not in lowered or "runtime" in lowered:
            continue
        columns = _columns(conn, table)
        start_column = _first(columns, ("start", "startNs", "startTime"))
        end_column = _first(columns, ("end", "endNs", "endTime"))
        correlation_column = _first(columns, ("correlationId", "correlation_id"))
        if start_column is None or end_column is None or correlation_column is None:
            continue
        used = False
        for row in conn.execute(f'SELECT * FROM "{table}"'):
            if any(row[column] is None for column in (start_column, end_column, correlation_column)):
                continue
            name = _kernel_name(row, strings)
            if not name:
                continue
            event = KernelEvent(
                start=int(row[start_column]),
                end=int(row[end_column]),
                correlation_id=int(row[correlation_column]),
                name=name,
                global_pid=_row_int(row, ("globalPid", "pid", "processId")),
                device_id=_row_int(row, ("deviceId", "device", "device_id")),
                stream_id=_row_int(row, ("streamId", "stream", "stream_id")),
                source_table=table,
            )
            key = (
                event.start,
                event.end,
                event.correlation_id,
                event.name,
                event.global_pid,
                event.device_id,
                event.stream_id,
            )
            if key in seen:
                continue
            seen.add(key)
            events.append(event)
            used = True
        if used:
            used_tables.append(table)
    events.sort(key=lambda item: item.start)
    return events, used_tables


def _find_enclosing(
    ranges: Iterable[NvtxRange], timestamp: int, api: ApiEvent
) -> NvtxRange | None:
    candidates = [
        item
        for item in ranges
        if item.start <= timestamp <= item.end
        and (
            item.process_id is None
            or api.process_id is None
            or item.process_id == api.process_id
        )
    ]
    if api.global_tid is not None:
        same_thread = [
            item for item in candidates if item.global_tid in {None, api.global_tid}
        ]
        if same_thread:
            candidates = same_thread
    return min(candidates, key=lambda item: item.end - item.start) if candidates else None


def _load_capture_events(
    events_dir: Path,
) -> tuple[
    dict[tuple[int | None, str], dict[str, Any]],
    dict[tuple[int | None, str], dict[str, Any]],
    list[Path],
]:
    paths = sorted(events_dir.glob("events_*.jsonl")) if events_dir.is_dir() else [events_dir]
    execution_events: dict[tuple[int | None, str], dict[str, Any]] = {}
    high_events: dict[tuple[int | None, str], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            event_type = event.get("event")
            if event_type not in {"execution_capture", "high_invocation"}:
                continue
            identifier = str(
                event.get("capture_id")
                if event_type == "execution_capture"
                else event.get("call_id")
            )
            if identifier in {"", "None"}:
                raise RuntimeError(
                    f"missing identifier for {event_type} event at {path}:{line_number}"
                )
            key = (int(event["pid"]) if event.get("pid") is not None else None, identifier)
            destination = execution_events if event_type == "execution_capture" else high_events
            if key in destination:
                raise RuntimeError(f"duplicate {event_type} event id in process: {key}")
            destination[key] = event
    return execution_events, high_events, [path for path in paths if path.exists()]


def _capture_event_for(
    capture_events: dict[tuple[int | None, str], dict[str, Any]],
    execution: NvtxRange,
) -> dict[str, Any]:
    capture_id = str(execution.fields.get("capture_id"))
    exact = capture_events.get((execution.process_id, capture_id))
    if exact is not None:
        return exact
    matches = [event for (pid, cid), event in capture_events.items() if cid == capture_id]
    if len(matches) == 1:
        return matches[0]
    return {}


def _high_event_for(
    high_events: dict[tuple[int | None, str], dict[str, Any]],
    high: NvtxRange,
) -> dict[str, Any]:
    call_id = str(high.fields.get("call_id"))
    exact = high_events.get((high.process_id, call_id))
    if exact is not None:
        return exact
    matches = [event for (pid, cid), event in high_events.items() if cid == call_id]
    if len(matches) == 1:
        return matches[0]
    return {}


class RuntimeTraceParser:
    def __init__(self, config: RuntimeCaptureConfig):
        self.config = config

    def parse(self, sqlite_path: Path, events_dir: Path) -> dict[str, Any]:
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            strings = _load_string_ids(conn)
            ranges, nvtx_tables = _load_nvtx_ranges(conn, strings)
            api_events, api_tables = _load_api_events(conn, strings)
            kernel_events, kernel_tables = _load_kernel_events(conn, strings)
            table_names = _tables(conn)
        finally:
            conn.close()

        capture_events, high_events, event_paths = _load_capture_events(events_dir)
        high_ranges = [item for item in ranges if item.fields.get("type") == "high"]
        execution_ranges = [item for item in ranges if item.fields.get("type") == "execution"]
        if not high_ranges:
            raise RuntimeError(
                "No KID high-level NVTX ranges found. SQLite tables: "
                + ", ".join(table_names)
            )
        if not kernel_events:
            raise RuntimeError(
                "No CUDA GPU kernel activities found. SQLite tables: "
                + ", ".join(table_names)
            )

        graph_launches = [
            event
            for event in api_events
            if event.name and "graphlaunch" in event.name.replace("_", "").lower()
        ]
        if graph_launches:
            raise RuntimeError(
                "CUDA Graph launch activity was observed; Runtime Capture requires eager execution"
            )

        apis_by_key: dict[tuple[int | None, int], list[ApiEvent]] = defaultdict(list)
        for event in api_events:
            apis_by_key[(event.process_id, event.correlation_id)].append(event)

        high_index = {id(item): index for index, item in enumerate(high_ranges)}
        execution_index = {id(item): index for index, item in enumerate(execution_ranges)}
        kernel_rows: list[dict[str, Any]] = []
        kernels_by_high: dict[int, list[dict[str, Any]]] = defaultdict(list)
        kernels_by_execution: dict[int, list[dict[str, Any]]] = defaultdict(list)

        for ordinal, kernel in enumerate(kernel_events, start=1):
            apis = list(apis_by_key.get((kernel.process_id, kernel.correlation_id), []))
            if not apis:
                apis = [
                    item
                    for (pid, correlation), values in apis_by_key.items()
                    if correlation == kernel.correlation_id
                    and (pid is None or kernel.process_id is None)
                    for item in values
                ]
            matches: list[tuple[NvtxRange, NvtxRange | None, ApiEvent]] = []
            for api in apis:
                high = _find_enclosing(high_ranges, api.start, api)
                if high is None:
                    continue
                execution = _find_enclosing(execution_ranges, api.start, api)
                if execution is not None and str(execution.fields.get("parent_call_id")) != str(
                    high.fields.get("call_id")
                ):
                    execution = None
                matches.append((high, execution, api))
            if not matches:
                continue
            high, execution, api = max(
                matches, key=lambda item: (item[1] is not None, item[2].start)
            )
            global_pid = kernel.global_pid
            kernel_id = f"p{global_pid or 0}-c{kernel.correlation_id}-k{ordinal}"
            row = {
                "kernel_id": kernel_id,
                "correlation_id": kernel.correlation_id,
                "name": kernel.name,
                "global_pid": global_pid,
                "device_id": kernel.device_id,
                "stream_id": kernel.stream_id,
                "gpu_start_ns": kernel.start,
                "gpu_end_ns": kernel.end,
                "duration_us": kernel.duration_us,
                "launch_api": {
                    "name": api.name,
                    "start_ns": api.start,
                    "table": api.source_table,
                },
                "high_call_id": str(high.fields.get("call_id")),
                "owner_capture_id": (
                    str(execution.fields.get("capture_id")) if execution else None
                ),
            }
            kernel_rows.append(row)
            kernels_by_high[high_index[id(high)]].append(row)
            if execution is not None:
                kernels_by_execution[execution_index[id(execution)]].append(row)

        invocations: list[dict[str, Any]] = []
        for high_position, high in enumerate(high_ranges):
            high_event = _high_event_for(high_events, high)
            if self.config.command is not None:
                if not high_event:
                    raise RuntimeError(
                        "service capture is missing high_invocation evidence for "
                        f"call_id={high.fields.get('call_id')} pid={high.process_id}"
                    )
                if not high_event.get("entry_python_stack"):
                    raise RuntimeError(
                        "service high_invocation has an empty entry_python_stack for "
                        f"call_id={high.fields.get('call_id')}"
                    )
            high_kernels = kernels_by_high.get(high_position, [])
            high_total_us = sum(item["duration_us"] for item in high_kernels)
            child_ranges = [
                execution
                for execution in execution_ranges
                if str(execution.fields.get("parent_call_id"))
                == str(high.fields.get("call_id"))
                and execution.process_id == high.process_id
                and high.start <= execution.start <= execution.end <= high.end
            ]
            child_ranges.sort(key=lambda item: item.start)
            entries: list[dict[str, Any]] = []
            for execution in child_ranges:
                direct_kernels = kernels_by_execution.get(execution_index[id(execution)], [])
                event = _capture_event_for(capture_events, execution)
                capture_id = str(execution.fields.get("capture_id"))
                provider_hint = event.get("provider_hint", event.get("provider"))
                implementation_hint = event.get(
                    "implementation_hint", event.get("implementation", {})
                )
                direct_us = sum(item["duration_us"] for item in direct_kernels)
                entries.append(
                    {
                        "capture_id": capture_id,
                        "parent_capture_id": execution.fields.get(
                            "parent_capture_id", event.get("parent_capture_id")
                        ),
                        "parent_call_id": str(execution.fields.get("parent_call_id")),
                        "archetype": execution.fields.get(
                            "archetype", event.get("archetype", "unknown")
                        ),
                        "common_interface": event.get("common_interface"),
                        "execution_interface": execution.fields.get(
                            "interface", event.get("execution_interface", "unknown")
                        ),
                        "provider_hint": provider_hint,
                        "execution_leaf": event.get("execution_leaf"),
                        "implementation_hint": implementation_hint or {},
                        "python_stack": event.get("python_stack", []),
                        "child_capture_ids": [],
                        "kernel_ids": [item["kernel_id"] for item in direct_kernels],
                        "inclusive_kernel_ids": [],
                        "metrics": {
                            "nvtx_cpu_duration_us": execution.duration_us,
                            "direct_gpu_kernel_sum_us": direct_us,
                            "direct_share_of_high_gpu": (
                                direct_us / high_total_us if high_total_us else 0.0
                            ),
                        },
                        "_start_ns": execution.start,
                    }
                )

            by_capture = {str(entry["capture_id"]): entry for entry in entries}
            for entry in entries:
                parent = by_capture.get(str(entry.get("parent_capture_id")))
                if parent is not None:
                    parent["child_capture_ids"].append(entry["capture_id"])
            kernel_by_id = {item["kernel_id"]: item for item in high_kernels}

            def populate(entry: dict[str, Any], visiting: set[str]) -> list[str]:
                capture_id = str(entry["capture_id"])
                if capture_id in visiting:
                    raise RuntimeError(f"capture parent cycle detected at {capture_id}")
                inclusive = list(entry["kernel_ids"])
                for child_id in entry["child_capture_ids"]:
                    child = by_capture.get(str(child_id))
                    if child is None:
                        continue
                    for kernel_id in populate(child, {*visiting, capture_id}):
                        if kernel_id not in inclusive:
                            inclusive.append(kernel_id)
                entry["inclusive_kernel_ids"] = inclusive
                inclusive_us = sum(
                    kernel_by_id[kernel_id]["duration_us"]
                    for kernel_id in inclusive
                    if kernel_id in kernel_by_id
                )
                entry["metrics"]["inclusive_gpu_kernel_sum_us"] = inclusive_us
                entry["metrics"]["inclusive_share_of_high_gpu"] = (
                    inclusive_us / high_total_us if high_total_us else 0.0
                )
                entry["attribution_role"] = (
                    "kernel_owner" if entry["kernel_ids"] else "ancestor_context"
                )
                return inclusive

            for entry in entries:
                populate(entry, set())

            raw_for_call = [
                event
                for event in capture_events.values()
                if str(event.get("parent_call_id", event.get("high_call_id")))
                == str(high.fields.get("call_id"))
                and (event.get("pid") is None or int(event["pid"]) == high.process_id)
            ]
            materialized = [entry for entry in entries if entry["inclusive_kernel_ids"]]
            capture_without_kernel_count = max(0, len(raw_for_call) - len(materialized))
            ranked = sorted(
                materialized,
                key=lambda item: (
                    -float(item["metrics"]["direct_gpu_kernel_sum_us"]),
                    int(item["_start_ns"]),
                ),
            )
            for rank, entry in enumerate(ranked, start=1):
                entry["hotspot_rank"] = rank
            materialized.sort(key=lambda item: int(item["_start_ns"]))
            for entry in materialized:
                entry.pop("_start_ns", None)

            attributed_ids = {
                kernel_id for entry in materialized for kernel_id in entry["kernel_ids"]
            }
            unattributed = [
                item["kernel_id"]
                for item in high_kernels
                if item["kernel_id"] not in attributed_ids
            ]
            attributed_us = sum(
                item["duration_us"]
                for item in high_kernels
                if item["kernel_id"] in attributed_ids
            )
            high_level = {
                "call_id": str(high.fields.get("call_id")),
                "interface": high.fields.get("interface", "unknown"),
                "nvtx_cpu_duration_us": high.duration_us,
                "kernel_ids": [item["kernel_id"] for item in high_kernels],
                "gpu_kernel_sum_us": high_total_us,
                "stage": high.fields.get("stage", "unknown"),
            }
            if high_event:
                high_level["instrumentation_mode"] = high_event.get(
                    "instrumentation_mode", "unknown"
                )
                high_level["entry_python_stack"] = high_event.get(
                    "entry_python_stack", []
                )
            invocations.append(
                {
                    "high_level": high_level,
                    "execution_captures": materialized,
                    "raw_capture_event_count": len(raw_for_call),
                    "capture_without_kernel_count": capture_without_kernel_count,
                    "unattributed_kernel_ids": unattributed,
                    "coverage": attributed_us / high_total_us if high_total_us else 0.0,
                    "_nvtx_start_ns": high.start,
                }
            )

        selected_invocations, selection_diagnostics = select_invocations(
            invocations, self.config.selection
        )
        selected_call_ids = {
            str(item["high_level"]["call_id"]) for item in selected_invocations
        }
        selected_kernel_ids = {
            kernel_id
            for invocation in selected_invocations
            for kernel_id in invocation["high_level"]["kernel_ids"]
        }
        selected_kernels = [
            item
            for item in kernel_rows
            if item["kernel_id"] in selected_kernel_ids
            and str(item["high_call_id"]) in selected_call_ids
        ]
        min_coverage = float(self.config.profiling.get("min_capture_coverage", 1.0))
        insufficient = [
            item["high_level"]["call_id"]
            for item in selected_invocations
            if float(item["coverage"]) + 1e-12 < min_coverage
        ]
        if insufficient:
            raise RuntimeError(
                f"capture coverage below {min_coverage:.3f} for high call ids: {insufficient}"
            )

        return {
            "schema_version": RUNTIME_CAPTURE_VERSION,
            "backend_name": self.config.backend_name,
            "target": {
                "interface": self.config.target_qualified_name
                or high_ranges[0].fields.get("interface"),
                "file": str(self.config.target_file),
                "line": self.config.target_line,
            },
            "invocation_selection": dict(self.config.selection),
            "metric_definition": {
                "gpu_kernel_sum_us": "sum of Nsight GPU kernel activity durations; not end-to-end wall time",
                "nvtx_cpu_duration_us": "CPU-side NVTX push-to-pop duration",
                "coverage": "execution-capture-attributed GPU duration / all high-level GPU duration",
            },
            "invocations": selected_invocations,
            "kernels": selected_kernels,
            "diagnostics": {
                "nvtx_tables": nvtx_tables,
                "cuda_api_tables": api_tables,
                "kernel_tables": kernel_tables,
                "high_range_count": len(high_ranges),
                "execution_range_count": len(execution_ranges),
                "capture_event_count": len(capture_events),
                **(
                    {"high_invocation_event_count": len(high_events)}
                    if self.config.command is not None or high_events
                    else {}
                ),
                "capture_event_files": [str(path) for path in event_paths],
                "cuda_api_event_count": len(api_events),
                "trace_kernel_count": len(kernel_events),
                "high_related_kernel_count": len(kernel_rows),
                "materialized_execution_capture_count": sum(
                    len(item["execution_captures"]) for item in selected_invocations
                ),
                **selection_diagnostics,
            },
        }
