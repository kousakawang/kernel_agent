from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DecomposerConfig
from .source_resolver import SourceResolver


@dataclass
class RangeEvent:
    start: int
    end: int
    fields: dict[str, Any]
    pid: int | None = None
    tid: int | None = None


@dataclass
class KernelEvent:
    start: int
    end: int
    name: str
    correlation_id: int | None
    pid: int | None = None
    tid: int | None = None

    @property
    def duration_us(self) -> float:
        return max(0, self.end - self.start) / 1000.0


@dataclass
class RuntimeEvent:
    start: int
    end: int
    correlation_id: int
    name: str | None = None
    pid: int | None = None
    tid: int | None = None


class TraceParser:
    def __init__(self, config: DecomposerConfig, resolver: SourceResolver):
        self.config = config
        self.resolver = resolver
        self.selection = config.selection
        self.string_ids: dict[int, str] = {}

    def parse(self, sqlite_path: Path) -> list[dict[str, Any]]:
        conn = sqlite3.connect(sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            self.string_ids = self._load_string_ids(conn)
            ranges = self._load_pygpu_ranges(conn)
            kernels = self._load_kernels(conn)
            runtimes = self._load_runtimes(conn)
        finally:
            conn.close()

        targets = [event for event in ranges if event.fields.get("type") == "target"]
        wrappers = [event for event in ranges if event.fields.get("type") == "wrap"]
        targets.sort(key=lambda event: event.start)
        skip = int(self.config.profiling.get("skip_target_invocations", 0))
        if skip:
            targets = targets[skip:]
        runtime_by_corr = {event.correlation_id: event for event in runtimes}

        by_call: dict[str, dict[str, Any]] = {}
        all_kernel_records: dict[str, list[dict[str, Any]]] = {}
        for kernel in kernels:
            runtime = runtime_by_corr.get(kernel.correlation_id) if kernel.correlation_id is not None else None
            ts = runtime.start if runtime else kernel.start
            target = _find_enclosing(targets, ts, runtime.tid if runtime else kernel.tid)
            if target is None:
                continue
            stage = target.fields.get("stage", "unknown")
            allowed = self.selection.get("stages")
            if allowed and stage not in allowed:
                continue
            call_id = str(target.fields.get("call_id", "unknown"))
            wrapper = _find_enclosing(wrappers, ts, runtime.tid if runtime else kernel.tid)
            wrapper_fields = dict(wrapper.fields) if wrapper else {}
            record = {
                "kernel_event": kernel,
                "runtime_event": runtime,
                "wrapper": wrapper_fields,
            }
            all_kernel_records.setdefault(call_id, []).append(record)
            if call_id not in by_call:
                by_call[call_id] = {
                    "call_id": _to_int_or_raw(target.fields.get("call_id")),
                    "pid": target.pid,
                    "tid": target.tid,
                    "stage": stage,
                    "forward_mode": target.fields.get("forward_mode"),
                    "start_ns": target.start,
                    "end_ns": target.end,
                    "selected_kernels": [],
                }

        invocations: list[dict[str, Any]] = []
        for call_id, invocation in sorted(by_call.items(), key=lambda item: invocation_sort_key(item[1])):
            records = all_kernel_records.get(call_id, [])
            selected = self._select_records(records)
            invocation["selected_kernels"] = [
                self._record_to_schema(idx + 1, record, records) for idx, record in enumerate(selected)
            ]
            invocations.append(invocation)
        return invocations

    def _select_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        top_k = int(self.selection.get("top_k", 20))
        min_duration = float(self.selection.get("min_duration_us", 0))
        min_share = float(self.selection.get("min_share_in_invocation", 0.0))
        total_us = sum(item["kernel_event"].duration_us for item in records) or 1.0
        out = []
        for record in sorted(records, key=lambda item: item["kernel_event"].duration_us, reverse=True):
            duration = record["kernel_event"].duration_us
            share = duration / total_us
            if duration < min_duration or share < min_share:
                continue
            out.append(record)
            if top_k and len(out) >= top_k:
                break
        return out

    def _record_to_schema(self, rank: int, record: dict[str, Any], all_records: list[dict[str, Any]]) -> dict[str, Any]:
        kernel: KernelEvent = record["kernel_event"]
        total_us = sum(item["kernel_event"].duration_us for item in all_records) or 1.0
        normalized = normalize_kernel_name(kernel.name)
        category, wrapper_info, implementation = self.resolver.resolve(
            raw_kernel_name=kernel.name,
            normalized_kernel_name=normalized,
            wrapper=record.get("wrapper"),
        )
        return {
            "rank": rank,
            "selection_reason": "top_k",
            "kernel": {
                "raw_name": kernel.name,
                "normalized_name": normalized,
                "category": category,
            },
            "metrics": {
                "duration_us": kernel.duration_us,
                "share_in_invocation": kernel.duration_us / total_us,
            },
            "wrapper": wrapper_info,
            "implementation": implementation,
            "attribution": {
                "method": "cuda_correlation_id+nvtx" if record.get("runtime_event") else "nvtx_time_containment",
                "confidence": "high" if record.get("wrapper") else "medium",
            },
        }

    def _load_string_ids(self, conn: sqlite3.Connection) -> dict[int, str]:
        tables = _tables(conn)
        string_table = next((name for name in tables if name.lower() == "stringids"), None)
        if not string_table:
            return {}
        cols = _columns(conn, string_table)
        id_col = _first(cols, ["id", "Id"])
        value_col = _first(cols, ["value", "string", "name"])
        if not id_col or not value_col:
            return {}
        out = {}
        for row in conn.execute(f'SELECT "{id_col}", "{value_col}" FROM "{string_table}"'):
            try:
                out[int(row[0])] = str(row[1])
            except Exception:
                continue
        return out

    def _load_pygpu_ranges(self, conn: sqlite3.Connection) -> list[RangeEvent]:
        out: list[RangeEvent] = []
        for table in _tables(conn):
            if "nvtx" not in table.lower():
                continue
            cols = _columns(conn, table)
            start_col = _first(cols, ["start", "startNs", "startTime"])
            end_col = _first(cols, ["end", "endNs", "endTime"])
            if not start_col or not end_col:
                continue
            for row in conn.execute(f'SELECT * FROM "{table}"'):
                text = self._row_text(row)
                if not text or "PYGPU:type=" not in text:
                    continue
                fields = parse_pygpu_label(text)
                out.append(
                    RangeEvent(
                        start=int(row[start_col]),
                        end=int(row[end_col]),
                        fields=fields,
                        pid=_row_int(row, ["globalPid", "pid", "processId"]),
                        tid=_row_int(row, ["globalTid", "tid", "threadId"]),
                    )
                )
        return out

    def _row_text(self, row: sqlite3.Row) -> str | None:
        keys = set(row.keys())
        for key in ("text", "name", "message"):
            if key in keys and row[key] is not None:
                return str(row[key])
        for key in ("textId", "nameId", "messageId", "domainId"):
            if key in keys and row[key] is not None:
                value = self.string_ids.get(int(row[key]))
                if value:
                    return value
        return None

    def _load_kernels(self, conn: sqlite3.Connection) -> list[KernelEvent]:
        out: list[KernelEvent] = []
        for table in _tables(conn):
            lowered = table.lower()
            if "kernel" not in lowered or "runtime" in lowered:
                continue
            cols = _columns(conn, table)
            start_col = _first(cols, ["start", "startNs", "startTime"])
            end_col = _first(cols, ["end", "endNs", "endTime"])
            corr_col = _first(cols, ["correlationId", "correlation_id"])
            if not start_col or not end_col:
                continue
            for row in conn.execute(f'SELECT * FROM "{table}"'):
                name = self._kernel_name(row)
                if not name:
                    continue
                out.append(
                    KernelEvent(
                        start=int(row[start_col]),
                        end=int(row[end_col]),
                        name=name,
                        correlation_id=int(row[corr_col]) if corr_col and row[corr_col] is not None else None,
                        pid=_row_int(row, ["globalPid", "pid", "processId"]),
                        tid=_row_int(row, ["globalTid", "tid", "threadId"]),
                    )
                )
        return out

    def _kernel_name(self, row: sqlite3.Row) -> str | None:
        keys = set(row.keys())
        for key in ("demangledName", "shortName", "mangledName", "name"):
            if key in keys and row[key] is not None:
                value = row[key]
                if isinstance(value, int):
                    return self.string_ids.get(int(value), str(value))
                return str(value)
        for key in ("demangledNameId", "shortNameId", "mangledNameId", "nameId"):
            if key in keys and row[key] is not None:
                return self.string_ids.get(int(row[key]), str(row[key]))
        return None

    def _load_runtimes(self, conn: sqlite3.Connection) -> list[RuntimeEvent]:
        out: list[RuntimeEvent] = []
        for table in _tables(conn):
            lowered = table.lower()
            if "runtime" not in lowered or "cupti" not in lowered:
                continue
            cols = _columns(conn, table)
            start_col = _first(cols, ["start", "startNs", "startTime"])
            end_col = _first(cols, ["end", "endNs", "endTime"])
            corr_col = _first(cols, ["correlationId", "correlation_id"])
            if not start_col or not end_col or not corr_col:
                continue
            for row in conn.execute(f'SELECT * FROM "{table}"'):
                if row[corr_col] is None:
                    continue
                out.append(
                    RuntimeEvent(
                        start=int(row[start_col]),
                        end=int(row[end_col]),
                        correlation_id=int(row[corr_col]),
                        name=self._row_text(row),
                        pid=_row_int(row, ["globalPid", "pid", "processId"]),
                        tid=_row_int(row, ["globalTid", "tid", "threadId"]),
                    )
                )
        return out


def parse_pygpu_label(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    marker = "PYGPU:"
    start = text.find(marker)
    if start >= 0:
        text = text[start + len(marker) :]
    for part in text.split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    return fields


def normalize_kernel_name(name: str) -> str:
    text = name.strip()
    for prefix in ("void ", "__global__ "):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if "(" in text:
        text = text.split("(", 1)[0]
    return text.strip()


def _find_enclosing(events: list[RangeEvent], ts: int, tid: int | None = None) -> RangeEvent | None:
    candidates = [event for event in events if event.start <= ts <= event.end]
    if tid is not None:
        same_tid = [event for event in candidates if event.tid in {None, tid}]
        if same_tid:
            candidates = same_tid
    if not candidates:
        return None
    return min(candidates, key=lambda event: event.end - event.start)


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    ]


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]


def _first(cols: list[str], candidates: list[str]) -> str | None:
    lowered = {col.lower(): col for col in cols}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _row_int(row: sqlite3.Row, keys: list[str]) -> int | None:
    available = set(row.keys())
    for key in keys:
        if key in available and row[key] is not None:
            try:
                return int(row[key])
            except Exception:
                return None
    return None


def _to_int_or_raw(value: Any) -> Any:
    try:
        return int(value)
    except Exception:
        return value


def invocation_sort_key(invocation: dict[str, Any]) -> tuple[int, str]:
    start = invocation.get("start_ns")
    try:
        return int(start), str(invocation.get("call_id"))
    except Exception:
        return 0, str(invocation.get("call_id"))

