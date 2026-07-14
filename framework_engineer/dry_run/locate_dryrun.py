"""locate dry-run: enrich KID schemas with source_locations skeletons.

Does NOT locate anything. For each kernel it applies the (filled) archetype's
null-rules: form-decided layers become ``not_applicable`` automatically; every
other layer becomes ``missed`` with ``{file, def_line}`` ``<FILL>`` placeholders —
the real "agent could not locate, hand to human" shape. Then writes a ``ref/``
directory holding non-contract reference material: ``locate_report.json`` (a
CLI-derived summary) and a ``locate_agent_notes.md`` stub.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import templates
from .fill_scan import has_unfilled_keys

# Fields the KID step must have filled before locate can proceed.
_KID_REQUIRED_KEYS = {"interface", "archetype", "archetype_code", "low_level_id"}


class LocateDryRunError(ValueError):
    pass


@dataclass
class LocateDryRunResult:
    schemas: list[Path]
    report_path: Path | None
    notes_path: Path | None
    gate_blocked: list[str] = field(default_factory=list)  # abs schema paths with unfilled KID keys


def _kernel_entries(schema: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(schema.get("kernels"), list):
        return [k for k in schema["kernels"] if isinstance(k, dict)]
    out: list[dict[str, Any]] = []
    for inv in schema.get("invocations") or []:
        if isinstance(inv, dict):
            out.extend(k for k in (inv.get("selected_kernels") or []) if isinstance(k, dict))
    return out


def run_one(schema_path: Path) -> tuple[bool, dict[str, Any]]:
    """Enrich one schema. Returns (ok, report_fragment).

    ok=False means the KID step still has unfilled required keys (gate).
    """
    schema_path = Path(schema_path).resolve()

    # Gate: interface/archetype must be filled first.
    blocked = has_unfilled_keys(schema_path, _KID_REQUIRED_KEYS)
    if blocked:
        return False, {"schema": str(schema_path), "blocked_on": [b.as_line() for b in blocked]}

    schema = json.loads(schema_path.read_text())
    needs_agent_items: list[dict[str, Any]] = []
    for entry in _kernel_entries(schema):
        archetype = str(entry.get("archetype", "")).strip()
        entry["source_locations"] = templates.source_locations_skeleton(archetype)
        kid = entry.get("low_level_id") or (entry.get("kernel") or {}).get("normalized_name") or "kernel"
        for layer_name, layer in entry["source_locations"]["layers"].items():
            if layer.get("status") == "missed":
                needs_agent_items.append(
                    {
                        "interface": entry.get("interface"),
                        "kernel": kid,
                        "archetype": archetype,
                        "layer": layer_name,
                        "repo_hint": layer.get("repo_hint"),
                    }
                )

    schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")
    return True, {"schema": str(schema_path), "needs_agent": needs_agent_items}


def run(schema_paths: list[Path]) -> LocateDryRunResult:
    if not schema_paths:
        raise LocateDryRunError("no schema found to enrich")

    fragments: list[dict[str, Any]] = []
    blocked_paths: list[str] = []
    enriched: list[Path] = []
    for sp in schema_paths:
        ok, frag = run_one(sp)
        if not ok:
            blocked_paths.append(str(Path(sp).resolve()))
            fragments.append(frag)
            continue
        enriched.append(Path(sp).resolve())
        fragments.append(frag)

    # ref/ holds non-contract reference material (no fixed downstream consumer):
    # locate_report.json is a CLI-derived summary; locate_agent_notes.md is the
    # Layer 2 agent's evidence/result report (content is prompt-driven in the
    # real pipeline; a stub here). The authoritative product is the schema itself.
    report_path = None
    notes_path = None
    if enriched:
        ref_dir = enriched[0].parent / "ref"
        ref_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "dry_run": True,
            "schemas": [str(p) for p in enriched],
            "needs_agent": [
                item
                for frag in fragments
                for item in frag.get("needs_agent", [])
            ],
        }
        report_path = ref_dir / "locate_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

        notes_lines = [
            "# locate_agent_notes (dry-run stub)",
            "",
            "参考资料（无固定消费者）。真实链路里由 Layer 2 agent 产出：对 locate 结果的",
            "证据阐述 + 结果报告（写什么由 prompt 约束）。dry-run 下所有适用层都标 missed，",
            "交人工补 file/def_line。",
            "",
        ]
        for item in report["needs_agent"]:
            notes_lines.append(
                f"- [{item['archetype']}] {item['interface']} / {item['layer']}"
                f"  repo_hint={item.get('repo_hint')}  -> TODO: 填 file/def_line 或明确放空"
            )
        notes_path = ref_dir / "locate_agent_notes.md"
        notes_path.write_text("\n".join(notes_lines) + "\n")

    return LocateDryRunResult(
        schemas=enriched,
        report_path=report_path,
        notes_path=notes_path,
        gate_blocked=blocked_paths,
    )
