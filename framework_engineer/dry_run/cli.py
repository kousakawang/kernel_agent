"""Dry-run CLI: validate the Step 0.5 -> Step 1 delivery chain without a GPU.

Three steps, run in order (plan §4.5):
  1. kid     — generate per-backend decomposition schema skeletons.
  2. locate  — enrich each schema with source_locations skeletons.
  3. extract — passthrough to the REAL Layer 3 CLI (source_location.cli extract).

Each step prints the absolute path of every file it created and the exact lines
the user must fill next. Convention: progress -> stderr, JSON summary -> stdout,
rc 0 ok / 2 config error or gate/hard-stop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import kid_dryrun, locate_dryrun
from .fill_scan import scan_files


def _print_fill_points(paths: list[Path]) -> list[dict]:
    """Print the fill points across paths to stderr; return them structured."""
    points = scan_files(paths)
    if points:
        print("→ 请填写以下位置（绝对路径:行号）：", file=sys.stderr)
        for fp in points:
            print(f"  {fp.as_line()}", file=sys.stderr)
    return [{"path": fp.path, "lineno": fp.lineno, "key": fp.key, "hint": fp.hint.strip()} for fp in points]


def cmd_kid(args: argparse.Namespace) -> int:
    try:
        config = kid_dryrun.KidDryRunConfig.load(args.config)
    except kid_dryrun.DryRunConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    result = kid_dryrun.run(config, out=args.out)

    print(f"[dry-run kid] generated {len(result.schemas)} backend schema(s):", file=sys.stderr)
    for p in result.schemas:
        print(f"  {p}", file=sys.stderr)
    fill_required = _print_fill_points(result.schemas)
    print(
        json.dumps(
            {"step": "kid", "schemas": [str(p) for p in result.schemas], "fill_required": fill_required},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _collect_schemas(args: argparse.Namespace) -> list[Path]:
    if args.schema:
        return [Path(args.schema).resolve()]
    ws = Path(args.workspace).resolve()
    return sorted(ws.rglob("decomposition_*.schema.json"))


def cmd_locate(args: argparse.Namespace) -> int:
    schemas = _collect_schemas(args)
    if not schemas:
        print("error: no decomposition_*.schema.json found", file=sys.stderr)
        return 2
    result = locate_dryrun.run(schemas)

    if result.gate_blocked:
        print(
            "[dry-run locate] GATE: these schemas still have unfilled KID fields "
            "(fill interface/archetype first):",
            file=sys.stderr,
        )
        for p in result.gate_blocked:
            print(f"  {p}", file=sys.stderr)
        # still surface exactly which lines
        _print_fill_points([Path(p) for p in result.gate_blocked])
        print(json.dumps({"step": "locate", "gate_blocked": result.gate_blocked}, indent=2, ensure_ascii=False))
        return 2

    print(f"[dry-run locate] enriched {len(result.schemas)} schema(s).", file=sys.stderr)
    if result.report_path:
        print(f"  report: {result.report_path}", file=sys.stderr)
        print(f"  notes : {result.notes_path}", file=sys.stderr)
    fill_required = _print_fill_points(result.schemas)
    print(
        json.dumps(
            {
                "step": "locate",
                "schemas": [str(p) for p in result.schemas],
                "locate_report": str(result.report_path) if result.report_path else None,
                "fill_required": fill_required,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    """Passthrough to the real Layer 3 extract CLI (plan decision A)."""
    from framework_engineer.source_location import cli as sl_cli

    schemas = _collect_schemas(args)
    if not schemas:
        print("error: no decomposition_*.schema.json found", file=sys.stderr)
        return 2

    rc_total = 0
    for schema in schemas:
        argv = ["extract", "--schema", str(schema)]
        if args.workspace_out:
            argv += ["--workspace-out", str(args.workspace_out)]
        else:
            argv += ["--workspace-out", str(schema.parent)]
        if args.allow_empty:
            argv.append("--allow-empty")
        print(f"[dry-run extract] -> source_location.cli {' '.join(argv)}", file=sys.stderr)
        rc = sl_cli.main(argv)
        rc_total = rc_total or rc
    if rc_total == 0:
        print("dry-run 全链路完成 → 交付物在各 backend 的 kernel_sources/", file=sys.stderr)
    return rc_total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m framework_engineer.dry_run.cli",
        description="Dry-run the Step 0.5 -> Step 1 delivery chain (no GPU).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    k = sub.add_parser("kid", help="Step 1: generate KID schema skeletons.")
    k.add_argument("--config", type=Path, required=True)
    k.add_argument("--out", type=Path, default=None)
    k.set_defaults(func=cmd_kid)

    loc = sub.add_parser("locate", help="Step 2: enrich schemas with source_locations skeletons.")
    g = loc.add_mutually_exclusive_group(required=True)
    g.add_argument("--schema", type=Path)
    g.add_argument("--workspace", type=Path)
    loc.set_defaults(func=cmd_locate)

    ex = sub.add_parser("extract", help="Step 3: passthrough to real Layer 3 extract.")
    ge = ex.add_mutually_exclusive_group(required=True)
    ge.add_argument("--schema", type=Path)
    ge.add_argument("--workspace", type=Path)
    ex.add_argument("--workspace-out", type=Path, default=None)
    ex.add_argument("--allow-empty", action="store_true")
    ex.set_defaults(func=cmd_extract)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
