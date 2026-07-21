"""Private deterministic tools used by the source-locate Agent.

This module intentionally is not wired into ``source_location.cli``.  The two
public commands remain ``locate`` and ``extract``; these helpers provide config
preflight, read-only context/search, mechanical finalization/evaluation, and
complete-workspace validation for the Prompt-driven Agent.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .agent_config import load_agent_config
from .agent_contracts import stripped_layers, validate_decisions
from .contracts import (
    CANDIDATE_AMBIGUOUS,
    CANDIDATE_NOT_FOUND,
    CANDIDATE_RESOLVED,
    CANDIDATE_STATUSES,
    LAYERS,
    ContractError,
    kernel_entries,
    kid_projection,
    load_json_object,
    validate_agent_schema,
    validate_kid_schema,
    write_json_atomic,
)
from .extractor import _end_line_for
from .locator import LocateError, SearchRoot, load_search_roots


class AgentHelperError(ContractError):
    """A private Agent-helper input or filesystem error."""


_CANDIDATE_FIELDS = {
    "status",
    "candidates",
    "repo_hint",
    "evidence",
    "diagnostics",
}
_CANDIDATE_HIT_FIELDS = {"file", "def_line"}
_SEARCH_MODES = ("literal", "registration", "loader", "build")
_SEARCH_GLOBS = (
    "*.py",
    "*.pyi",
    "*.cc",
    "*.cpp",
    "*.cxx",
    "*.cu",
    "*.cuh",
    "*.h",
    "*.hpp",
    "*.hh",
    "*.cmake",
    "CMakeLists.txt",
    "*.toml",
    "*.json",
    "*.txt",
)
_CATEGORY_TOKENS: dict[str, tuple[str, ...]] = {
    "registration": (
        "torch_library",
        "torch_library_impl",
        "pybind11_module",
        "tvm_ffi_dll_export",
        "m.def(",
        "m.impl(",
        ".def(",
        ".impl(",
        "register_op",
        "register_operator",
    ),
    "loader": (
        "load_jit",
        "load_inline",
        "build_and_load",
        "gen_jit_spec",
        "torch.ops.",
        "getattr(torch.ops",
        "sources=",
        "source_paths",
    ),
    "build": (
        "fetchcontent",
        "add_subdirectory",
        "target_sources",
        "target_link_libraries",
        "target_include_directories",
        "find_package",
        "cudaextension",
        "cppextension",
        "cmakelists",
    ),
}


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _non_empty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentHelperError(f"{where} must be a non-empty string")
    return value


def _validate_candidate_schema(
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    entries = validate_kid_schema(schema, allow_locate_candidates=True)
    for index, entry in enumerate(entries):
        where = f"kernels[{index}].locate_candidates"
        candidates = entry.get("locate_candidates")
        if not isinstance(candidates, dict) or set(candidates) != {
            "interface_definition"
        }:
            raise AgentHelperError(
                f"{where} must contain exactly interface_definition"
            )
        result = candidates["interface_definition"]
        if not isinstance(result, dict) or set(result) != _CANDIDATE_FIELDS:
            raise AgentHelperError(
                f"{where}.interface_definition has invalid fields"
            )
        status = result.get("status")
        if status not in CANDIDATE_STATUSES:
            raise AgentHelperError(
                f"{where}.interface_definition.status is invalid: {status!r}"
            )
        hits = result.get("candidates")
        if not isinstance(hits, list):
            raise AgentHelperError(
                f"{where}.interface_definition.candidates must be an array"
            )
        if status == CANDIDATE_RESOLVED and len(hits) != 1:
            raise AgentHelperError(f"{where} resolved candidate requires one hit")
        if status == CANDIDATE_AMBIGUOUS and len(hits) < 2:
            raise AgentHelperError(
                f"{where} ambiguous candidate requires at least two hits"
            )
        if status == CANDIDATE_NOT_FOUND and hits:
            raise AgentHelperError(f"{where} not_found candidate must be empty")
        for hit_index, hit in enumerate(hits):
            hit_where = f"{where}.interface_definition.candidates[{hit_index}]"
            if not isinstance(hit, dict) or set(hit) != _CANDIDATE_HIT_FIELDS:
                raise AgentHelperError(
                    f"{hit_where} must contain exactly file and def_line"
                )
            file = _non_empty_string(hit.get("file"), f"{hit_where}.file")
            if not Path(file).is_absolute():
                raise AgentHelperError(f"{hit_where}.file must be absolute")
            line = hit.get("def_line")
            if isinstance(line, bool) or not isinstance(line, int) or line < 1:
                raise AgentHelperError(
                    f"{hit_where}.def_line must be a positive integer"
                )
        repo_hint = result.get("repo_hint")
        if repo_hint is not None and not isinstance(repo_hint, str):
            raise AgentHelperError(f"{where}.repo_hint must be a string or null")
        _non_empty_string(result.get("evidence"), f"{where}.evidence")
        diagnostics = result.get("diagnostics")
        if not isinstance(diagnostics, list) or not all(
            isinstance(item, str) for item in diagnostics
        ):
            raise AgentHelperError(f"{where}.diagnostics must be a string array")
    return entries


def _root_owner(path: Path, roots: list[SearchRoot]) -> SearchRoot | None:
    path = _absolute(path)
    try:
        real_path = path.resolve()
    except OSError:
        real_path = path
    owners: list[SearchRoot] = []
    for root in roots:
        try:
            path.relative_to(root.path)
        except ValueError:
            try:
                real_path.relative_to(root.path.resolve())
            except (OSError, ValueError):
                continue
        owners.append(root)
    return max(owners, key=lambda item: len(item.path.parts), default=None)


def prepare_agent_run(config_path: Path) -> dict[str, Any]:
    """Validate one config, create its workspace layout, and expose run paths."""

    config = load_agent_config(config_path)
    kid_schema = load_json_object(config.kid_schema, label="KID schema")
    entries = validate_kid_schema(kid_schema)
    roots, skipped = load_search_roots(
        config.third_party_manifest, config.sglang_repo_root
    )
    owner = _root_owner(config.run.workspace, roots)
    if owner is not None:
        raise AgentHelperError(
            "Agent config workspace must not be inside source root "
            f"{owner.path}: {config.run.workspace}"
        )
    config.run.create_directories()
    report = config.to_dict()
    report.update(
        {
            "kernels": [entry["low_level_id"] for entry in entries],
            "search_roots": [
                {"name": root.name, "path": str(root.path)} for root in roots
            ],
            "search_roots_skipped": skipped,
        }
    )
    return report


def _source_context(path: Path, line: int, *, max_lines: int) -> dict[str, Any]:
    path = _absolute(path)
    result: dict[str, Any] = {
        "file": str(path),
        "def_line": line,
        "exists": path.is_file(),
    }
    if not path.is_file():
        result["error"] = "file not found"
        return result
    try:
        text = path.read_text(errors="ignore")
    except OSError as exc:
        result["error"] = f"unreadable: {exc}"
        return result
    lines = text.splitlines(keepends=True)
    result["line_count"] = len(lines)
    if line < 1 or line > len(lines):
        result["error"] = f"line out of range: {line} not in 1-{len(lines)}"
        return result
    end_line = _end_line_for(path, lines, line)
    snippet_end = min(end_line, line + max_lines - 1)
    result.update(
        {
            "end_line": end_line,
            "snippet_start": line,
            "snippet_end": snippet_end,
            "truncated": snippet_end < end_line,
            "snippet": [
                {"line": number, "text": lines[number - 1].rstrip("\r\n")}
                for number in range(line, snippet_end + 1)
            ],
        }
    )
    return result


def _line_context(path: Path, line: int, radius: int = 3) -> dict[str, Any]:
    path = _absolute(path)
    result: dict[str, Any] = {"file": str(path), "line": line, "exists": path.is_file()}
    if not path.is_file():
        result["error"] = "file not found"
        return result
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError as exc:
        result["error"] = f"unreadable: {exc}"
        return result
    if line < 1 or line > len(lines):
        result["error"] = f"line out of range: {line} not in 1-{len(lines)}"
        return result
    start = max(1, line - radius)
    end = min(len(lines), line + radius)
    result["context"] = [
        {"line": number, "text": lines[number - 1]}
        for number in range(start, end + 1)
    ]
    return result


def inspect_target(
    schema_path: Path,
    *,
    kernel_id: str,
    manifest_path: Path,
    sglang_repo_root: Path,
    max_lines: int = 160,
) -> dict[str, Any]:
    """Return one target's deterministic context without choosing any layer."""

    if max_lines < 1 or max_lines > 1000:
        raise AgentHelperError("--max-lines must be between 1 and 1000")
    schema = load_json_object(_absolute(schema_path), label="candidate schema")
    entries = _validate_candidate_schema(schema)
    matches = [entry for entry in entries if entry["low_level_id"] == kernel_id]
    if not matches:
        raise AgentHelperError(f"unknown low_level_id: {kernel_id}")
    roots, skipped = load_search_roots(
        _absolute(manifest_path), _absolute(sglang_repo_root)
    )
    entry = matches[0]
    candidate = entry["locate_candidates"]["interface_definition"]
    call_site = entry["runtime_event"]["call_site"]
    candidate_contexts: list[dict[str, Any]] = []
    for hit in candidate["candidates"]:
        path = Path(hit["file"])
        context = _source_context(
            path, int(hit["def_line"]), max_lines=max_lines
        )
        owner = _root_owner(path, roots)
        context["allowed_source_root"] = (
            {"name": owner.name, "path": str(owner.path)}
            if owner is not None
            else None
        )
        candidate_contexts.append(context)
    return {
        "low_level_id": entry["low_level_id"],
        "interface": entry["interface"],
        "kernel": copy.deepcopy(entry["kernel"]),
        # These are opaque hints only; helpers never branch on their values.
        "archetype_hint": entry["archetype"],
        "provider_hint": entry["provider"],
        "call_site": copy.deepcopy(call_site),
        "call_site_context": _line_context(
            Path(call_site["file"]), int(call_site["line"])
        ),
        "interface_candidates": copy.deepcopy(candidate),
        "candidate_contexts": candidate_contexts,
        "search_roots": [
            {"name": root.name, "path": str(root.path)} for root in roots
        ],
        "skipped_roots": skipped,
    }


def _query_variants(queries: Iterable[str]) -> list[str]:
    variants: list[str] = []
    for raw in queries:
        query = raw.strip()
        if not query:
            raise AgentHelperError("--query values must be non-empty")
        variants.append(query)
        leaf = re.split(r"::|\.|/", query)[-1]
        if leaf and leaf != query:
            variants.append(leaf)
    if not variants:
        raise AgentHelperError("at least one --query is required")
    return list(dict.fromkeys(variants))


def _category_for(path: Path, context: str) -> str:
    lowered = f"{path.name}\n{context}".lower()
    for category in ("registration", "loader", "build"):
        if any(token in lowered for token in _CATEGORY_TOKENS[category]):
            return category
    return "literal"


def _match_context(path: Path, line: int, radius: int = 3) -> str:
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return ""
    start = max(1, line - radius)
    end = min(len(lines), line + radius)
    return "\n".join(
        f"{number}: {lines[number - 1]}" for number in range(start, end + 1)
    )


def search_sources(
    *,
    manifest_path: Path,
    sglang_repo_root: Path,
    mode: str,
    queries: list[str],
    limit: int = 200,
) -> dict[str, Any]:
    """Search allowed roots and return raw, unranked source candidates."""

    if mode not in _SEARCH_MODES:
        raise AgentHelperError(f"unsupported search mode: {mode}")
    if limit < 1 or limit > 5000:
        raise AgentHelperError("--limit must be between 1 and 5000")
    variants = _query_variants(queries)
    roots, skipped = load_search_roots(
        _absolute(manifest_path), _absolute(sglang_repo_root)
    )
    command = [
        "rg",
        "--json",
        "--line-number",
        "--with-filename",
        "--color",
        "never",
        "--fixed-strings",
    ]
    for variant in variants:
        command.extend(("-e", variant))
    command.extend(("--glob", "!**/.git/**"))
    for glob in _SEARCH_GLOBS:
        command.extend(("--glob", glob))
    command.extend(str(root.path) for root in roots)
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise AgentHelperError("rg is required by the private search helper") from exc
    if completed.returncode not in {0, 1}:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise AgentHelperError(f"rg search failed: {detail}")

    found: dict[tuple[str, int, str], dict[str, Any]] = {}
    for raw_line in completed.stdout.splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data", {})
        path_text = data.get("path", {}).get("text")
        line_number = data.get("line_number")
        line_text = data.get("lines", {}).get("text")
        if not isinstance(path_text, str) or not isinstance(line_number, int):
            continue
        if not isinstance(line_text, str):
            line_text = ""
        path = _absolute(Path(path_text))
        owner = _root_owner(path, roots)
        if owner is None:
            continue
        context = _match_context(path, line_number)
        category = _category_for(path, context)
        if mode != "literal" and category != mode:
            continue
        submatches = data.get("submatches") or []
        matched_query = ""
        if submatches and isinstance(submatches[0], dict):
            matched_query = str(submatches[0].get("match", {}).get("text", ""))
        key = (str(path), line_number, line_text.rstrip("\r\n"))
        found[key] = {
            "category": category,
            "matched_query": matched_query,
            "repo_name": owner.name,
            "repo_root": str(owner.path),
            "file": str(path),
            "line": line_number,
            "text": line_text.rstrip("\r\n"),
            "context": context,
        }

    ordered = sorted(found.values(), key=lambda item: (item["file"], item["line"]))
    return {
        "mode": mode,
        "queries": variants,
        "roots_searched": [
            {"name": root.name, "path": str(root.path)} for root in roots
        ],
        "skipped_roots": skipped,
        "total_matches": len(ordered),
        "truncated": len(ordered) > limit,
        "matches": ordered[:limit],
    }


def _validate_hit_file(
    hit: dict[str, Any],
    *,
    roots: list[SearchRoot],
    where: str,
    require_python: bool,
) -> SearchRoot:
    path = Path(hit["file"])
    owner = _root_owner(path, roots)
    if owner is None:
        raise AgentHelperError(
            f"{where}.file is outside the allowed source roots: {path}"
        )
    if not path.is_file():
        raise AgentHelperError(f"{where}.file not found: {path}")
    if require_python and path.suffix.lower() not in {".py", ".pyi"}:
        raise AgentHelperError(
            f"{where}.file must be Python source for interface_definition"
        )
    try:
        total = len(path.read_text(errors="ignore").splitlines())
    except OSError as exc:
        raise AgentHelperError(f"{where}.file is unreadable: {exc}") from exc
    line = int(hit["def_line"])
    if line > total:
        raise AgentHelperError(
            f"{where}.def_line {line} is out of range for {path} (1-{total})"
        )
    return owner


def _materialize_layers(
    decision: dict[str, Any],
    *,
    roots: list[SearchRoot],
) -> dict[str, dict[str, Any]]:
    layers = stripped_layers(decision)
    kernel_id = decision["low_level_id"]
    for layer_name in LAYERS:
        owners: set[str] = set()
        for index, raw_hit in enumerate(decision["layers"][layer_name]["hits"]):
            owner = _validate_hit_file(
                raw_hit,
                roots=roots,
                where=f"{kernel_id}/{layer_name}/hits[{index}]",
                require_python=layer_name == "interface_definition",
            )
            owners.add(str(owner.path))
        layers[layer_name]["repo_hint"] = (
            next(iter(owners)) if len(owners) == 1 else None
        )
    return layers


def _render_notes(
    entries: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]
) -> str:
    lines = [
        "# source_locate Agent notes",
        "",
        "本文件由 Agent decisions 机械生成；正式下游契约只存在于 schema 的",
        "四层结果中。",
        "",
    ]
    for entry in entries:
        kernel_id = entry["low_level_id"]
        decision = decisions[kernel_id]
        lines.extend(
            [
                f"## {kernel_id}",
                "",
                f"- Interface: `{entry['interface']}`",
                f"- Summary: {decision['summary']}",
                "",
                "### Layer evidence",
                "",
            ]
        )
        for layer_name in LAYERS:
            layer = decision["layers"][layer_name]
            lines.append(f"- `{layer_name}` — `{layer['status']}`: {layer['rationale']}")
            for hit in layer["hits"]:
                lines.append(
                    "  - "
                    f"`{hit['file']}:{hit['def_line']}` `{hit['symbol']}` — "
                    f"{hit['reason']}"
                )
        lines.extend(["", "### Gaps and follow-up", ""])
        if decision["gaps"]:
            lines.extend(f"- Gap: {gap}" for gap in decision["gaps"])
        else:
            lines.append("- Gaps: none.")
        followup = decision["manual_followup"]
        lines.append(
            f"- Manual follow-up: {followup}"
            if followup is not None
            else "- Manual follow-up: none."
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def finalize_agent_result(
    schema_path: Path,
    *,
    decisions_path: Path,
    manifest_path: Path,
    sglang_repo_root: Path,
    output_path: Path,
    notes_path: Path,
) -> dict[str, Any]:
    """Validate decisions and atomically materialize Agent-owned outputs."""

    schema_path = _absolute(schema_path)
    decisions_path = _absolute(decisions_path)
    output_path = _absolute(output_path)
    notes_path = _absolute(notes_path)
    protected = {schema_path, decisions_path, _absolute(manifest_path)}
    if output_path in protected:
        raise AgentHelperError("--out must differ from schema, decisions, and manifest")
    if notes_path in protected or notes_path == output_path:
        raise AgentHelperError(
            "--notes-out must differ from all inputs and from --out"
        )

    candidate_schema = load_json_object(schema_path, label="candidate schema")
    entries = _validate_candidate_schema(candidate_schema)
    decisions_payload = load_json_object(decisions_path, label="decisions")
    kernel_ids = [entry["low_level_id"] for entry in entries]
    decisions = validate_decisions(
        decisions_payload, expected_kernel_ids=kernel_ids
    )
    roots, skipped = load_search_roots(
        _absolute(manifest_path), _absolute(sglang_repo_root)
    )
    for label, path in (("--out", output_path), ("--notes-out", notes_path)):
        owner = _root_owner(path, roots)
        if owner is not None:
            raise AgentHelperError(
                f"{label} must not write inside source root {owner.path}: {path}"
            )

    output = copy.deepcopy(candidate_schema)
    output_entries = kernel_entries(output)
    for entry in output_entries:
        kernel_id = entry["low_level_id"]
        entry.pop("locate_candidates", None)
        entry["source_locations"] = {
            "layers": _materialize_layers(decisions[kernel_id], roots=roots)
        }
    validate_agent_schema(output)
    if any("kernel_sources_dir" in entry for entry in output_entries):
        raise AgentHelperError("finalized Agent schema must not contain kernel_sources_dir")
    if kid_projection(output) != kid_projection(candidate_schema):
        raise AgentHelperError("finalization changed KID-owned fields")

    notes = _render_notes(output_entries, decisions)
    # All contract, source-root, path, and line checks happen before either
    # existing output is replaced.
    write_json_atomic(output_path, output)
    _write_text_atomic(notes_path, notes)
    return {
        "schema": str(schema_path),
        "decisions": str(decisions_path),
        "output": str(output_path),
        "notes": str(notes_path),
        "kernels": len(output_entries),
        "search_roots_skipped": skipped,
    }


def _ordered_subsequence(
    required: list[dict[str, Any]], actual: list[dict[str, Any]]
) -> bool:
    position = 0
    for hit in actual:
        if position < len(required) and hit == required[position]:
            position += 1
    return position == len(required)


def evaluate_agent_result(
    actual_path: Path,
    *,
    decisions_path: Path,
    golden_path: Path,
    manifest_path: Path,
    sglang_repo_root: Path,
) -> dict[str, Any]:
    """Evaluate strict core-chain recall while permitting explained extras."""

    actual = load_json_object(_absolute(actual_path), label="actual Agent schema")
    golden = load_json_object(_absolute(golden_path), label="golden Agent schema")
    actual_entries = validate_agent_schema(actual)
    golden_entries = validate_agent_schema(golden)
    errors: list[str] = []
    if any("kernel_sources_dir" in entry for entry in actual_entries):
        errors.append("actual schema contains kernel_sources_dir before extract")
    if kid_projection(actual) != kid_projection(golden):
        errors.append("actual KID-owned fields differ from golden")

    actual_by_id = {entry["low_level_id"]: entry for entry in actual_entries}
    golden_by_id = {entry["low_level_id"]: entry for entry in golden_entries}
    if set(actual_by_id) != set(golden_by_id):
        errors.append("actual and golden low_level_id sets differ")

    decisions_payload = load_json_object(_absolute(decisions_path), label="decisions")
    decisions = validate_decisions(
        decisions_payload,
        expected_kernel_ids=[entry["low_level_id"] for entry in actual_entries],
    )
    roots, skipped = load_search_roots(
        _absolute(manifest_path), _absolute(sglang_repo_root)
    )

    for kernel_id in sorted(set(actual_by_id).intersection(golden_by_id)):
        actual_layers = actual_by_id[kernel_id]["source_locations"]["layers"]
        golden_layers = golden_by_id[kernel_id]["source_locations"]["layers"]
        decision_layers = _materialize_layers(decisions[kernel_id], roots=roots)
        if actual_layers != decision_layers:
            errors.append(
                f"{kernel_id}: actual layers differ from finalized decisions"
            )
            continue
        for layer_name in LAYERS:
            actual_layer = actual_layers[layer_name]
            golden_layer = golden_layers[layer_name]
            if actual_layer["status"] != golden_layer["status"]:
                errors.append(
                    f"{kernel_id}/{layer_name}: status "
                    f"{actual_layer['status']!r} != golden {golden_layer['status']!r}"
                )
            if not _ordered_subsequence(golden_layer["hits"], actual_layer["hits"]):
                errors.append(
                    f"{kernel_id}/{layer_name}: golden core hits are not an "
                    "ordered subsequence of actual hits"
                )

    return {
        "ok": not errors,
        "actual": str(_absolute(actual_path)),
        "golden": str(_absolute(golden_path)),
        "kernels": len(actual_entries),
        "errors": errors,
        "search_roots_skipped": skipped,
    }


def validate_agent_run(config_path: Path) -> dict[str, Any]:
    """Validate the complete config-driven locate→Agent→extract workspace."""

    config = load_agent_config(config_path)
    roots, skipped = load_search_roots(
        config.third_party_manifest, config.sglang_repo_root
    )
    owner = _root_owner(config.run.workspace, roots)
    if owner is not None:
        raise AgentHelperError(
            "Agent config workspace must not be inside source root "
            f"{owner.path}: {config.run.workspace}"
        )

    kid = load_json_object(config.kid_schema, label="KID schema")
    kid_entries = validate_kid_schema(kid)
    candidate = load_json_object(
        config.run.candidate_schema, label="locate candidate schema"
    )
    candidate_entries = _validate_candidate_schema(candidate)
    decisions_payload = load_json_object(config.run.decisions, label="decisions")
    kernel_ids = [entry["low_level_id"] for entry in kid_entries]
    decisions = validate_decisions(
        decisions_payload, expected_kernel_ids=kernel_ids
    )
    located = load_json_object(config.run.located_schema, label="located schema")
    located_entries = validate_agent_schema(located)
    extracted = load_json_object(
        config.run.extracted_schema, label="extracted schema"
    )
    extracted_entries = validate_agent_schema(extracted)

    projection = kid_projection(kid)
    for label, payload in (
        ("candidate", candidate),
        ("located", located),
        ("extracted", extracted),
    ):
        if kid_projection(payload) != projection:
            raise AgentHelperError(f"{label} schema changed KID-owned fields")

    if [entry["low_level_id"] for entry in candidate_entries] != kernel_ids:
        raise AgentHelperError("candidate schema changed KID kernel order")
    if [entry["low_level_id"] for entry in located_entries] != kernel_ids:
        raise AgentHelperError("located schema changed KID kernel order")
    if [entry["low_level_id"] for entry in extracted_entries] != kernel_ids:
        raise AgentHelperError("extracted schema changed KID kernel order")

    for entry in located_entries:
        kernel_id = entry["low_level_id"]
        expected_layers = _materialize_layers(decisions[kernel_id], roots=roots)
        if entry["source_locations"]["layers"] != expected_layers:
            raise AgentHelperError(
                f"{kernel_id}: located layers differ from finalized decisions"
            )
        if "kernel_sources_dir" in entry:
            raise AgentHelperError(
                f"{kernel_id}: located schema must not contain kernel_sources_dir"
            )

    expected_extracted = copy.deepcopy(located)
    expected_by_id = {
        entry["low_level_id"]: entry for entry in kernel_entries(expected_extracted)
    }
    for kernel_id in kernel_ids:
        expected_dir = config.run.kernel_sources / kernel_id
        expected_by_id[kernel_id]["kernel_sources_dir"] = str(expected_dir)
    if extracted != expected_extracted:
        raise AgentHelperError(
            "extracted schema must equal located schema plus kernel_sources_dir"
        )

    if not config.run.notes.is_file():
        raise AgentHelperError(f"Agent notes not found: {config.run.notes}")
    try:
        notes_text = config.run.notes.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise AgentHelperError(f"Agent notes are not valid UTF-8: {config.run.notes}") from exc
    if not notes_text.strip():
        raise AgentHelperError(f"Agent notes are empty: {config.run.notes}")

    for kernel_id in kernel_ids:
        target_dir = config.run.kernel_sources / kernel_id
        required = (
            target_dir / "interface_definition.py",
            target_dir / "kernel_impl",
            target_dir / "py_cpp_binding",
            target_dir / "kernel_header",
            target_dir / "read_hints.txt",
        )
        for path in required:
            if not path.exists():
                raise AgentHelperError(
                    f"{kernel_id}: extracted artifact missing: {path}"
                )

    layer_status_counts: dict[str, int] = {}
    for entry in located_entries:
        for layer in entry["source_locations"]["layers"].values():
            status = layer["status"]
            layer_status_counts[status] = layer_status_counts.get(status, 0) + 1
    extracted_files = sum(
        1 for path in config.run.kernel_sources.rglob("*") if path.is_file()
    )
    return {
        "ok": True,
        "config": str(config.path),
        "testcase_id": config.testcase_id,
        "kernels": len(kernel_ids),
        "layer_status_counts": layer_status_counts,
        "extracted_files": extracted_files,
        "artifacts": config.run.to_dict(),
        "search_roots_skipped": skipped,
    }


def _cmd_inspect_target(args: argparse.Namespace) -> int:
    report = inspect_target(
        args.schema,
        kernel_id=args.kernel_id,
        manifest_path=args.manifest,
        sglang_repo_root=args.sglang_repo_root,
        max_lines=args.max_lines,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_prepare_run(args: argparse.Namespace) -> int:
    report = prepare_agent_run(args.config)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    report = search_sources(
        manifest_path=args.manifest,
        sglang_repo_root=args.sglang_repo_root,
        mode=args.mode,
        queries=args.query,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_finalize(args: argparse.Namespace) -> int:
    report = finalize_agent_result(
        args.schema,
        decisions_path=args.decisions,
        manifest_path=args.manifest,
        sglang_repo_root=args.sglang_repo_root,
        output_path=args.out,
        notes_path=args.notes_out,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    report = evaluate_agent_result(
        args.actual,
        decisions_path=args.decisions,
        golden_path=args.golden,
        manifest_path=args.manifest,
        sglang_repo_root=args.sglang_repo_root,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


def _cmd_validate_run(args: argparse.Namespace) -> int:
    report = validate_agent_run(args.config)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m framework_engineer.source_location.agent_helper",
        description="Private deterministic tools for the Prompt-driven source-locate Agent.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-run")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.set_defaults(func=_cmd_prepare_run)

    inspect = subparsers.add_parser("inspect-target")
    inspect.add_argument("--schema", type=Path, required=True)
    inspect.add_argument("--kernel-id", required=True)
    inspect.add_argument("--manifest", type=Path, required=True)
    inspect.add_argument("--sglang-repo-root", type=Path, required=True)
    inspect.add_argument("--max-lines", type=int, default=160)
    inspect.set_defaults(func=_cmd_inspect_target)

    search = subparsers.add_parser("search")
    search.add_argument("--manifest", type=Path, required=True)
    search.add_argument("--sglang-repo-root", type=Path, required=True)
    search.add_argument("--mode", choices=_SEARCH_MODES, required=True)
    search.add_argument("--query", action="append", required=True)
    search.add_argument("--limit", type=int, default=200)
    search.set_defaults(func=_cmd_search)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--schema", type=Path, required=True)
    finalize.add_argument("--decisions", type=Path, required=True)
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--sglang-repo-root", type=Path, required=True)
    finalize.add_argument("--out", type=Path, required=True)
    finalize.add_argument("--notes-out", type=Path, required=True)
    finalize.set_defaults(func=_cmd_finalize)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--actual", type=Path, required=True)
    evaluate.add_argument("--decisions", type=Path, required=True)
    evaluate.add_argument("--golden", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--sglang-repo-root", type=Path, required=True)
    evaluate.set_defaults(func=_cmd_evaluate)

    validate = subparsers.add_parser("validate-run")
    validate.add_argument("--config", type=Path, required=True)
    validate.set_defaults(func=_cmd_validate_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (AgentHelperError, ContractError, LocateError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
