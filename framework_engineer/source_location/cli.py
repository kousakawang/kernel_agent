"""CLI for deterministic source location and Layer-3 extraction.

Convention (matches third_party_solver/cli.py): progress -> stderr, JSON summary
-> stdout, return code 0 on success / 2 on hard stop or config error.

    python3 -m framework_engineer.source_location.cli locate \
        --schema <kid-schema.json> \
        --manifest <third_party_manifest.json> \
        --sglang-repo-root <sglang-root> [--out <output-schema.json>]

    python3 -m framework_engineer.source_location.cli extract \
        --schema <decomposition_*.schema.json> [--workspace-out <dir>] [--allow-empty]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .extractor import extract_workspace
from .locator import LocateError, locate_schema


def cmd_extract(args: argparse.Namespace) -> int:
    schema = Path(args.schema)
    if not schema.exists():
        print(f"error: schema not found: {schema}", file=sys.stderr)
        return 2

    report = extract_workspace(
        schema,
        workspace_out=args.workspace_out,
        allow_empty=args.allow_empty,
    )

    if report.stopped:
        print(
            "[extract] HARD STOP: required layers unresolved "
            "(fill file/line in the schema, or re-run with --allow-empty):",
            file=sys.stderr,
        )
        for m in report.missing:
            hint = f"  (repo_hint: {m['repo_hint']})" if m.get("repo_hint") else ""
            reason = f"  [{m['reason']}]" if m.get("reason") else ""
            print(f"  - {m['kernel']} / {m['layer']}{reason}{hint}", file=sys.stderr)
        print(json.dumps(report.summary(), indent=2))
        return 2

    for i, ke in enumerate(report.kernels, 1):
        wrote = ",".join(ke.layers_written) or "-"
        ph = ",".join(ke.layers_placeholder) or "-"
        print(
            f"[{i}/{len(report.kernels)}] {ke.kernel_id}: wrote {wrote} / placeholder {ph}",
            file=sys.stderr,
            flush=True,
        )
    print(json.dumps(report.summary(), indent=2))
    return 0


def cmd_locate(args: argparse.Namespace) -> int:
    try:
        result = locate_schema(
            Path(args.schema),
            manifest_path=Path(args.manifest),
            sglang_repo_root=Path(args.sglang_repo_root),
            output_path=Path(args.out) if args.out is not None else None,
        )
    except (LocateError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = result.summary()
    print(
        "[locate] "
        f"{summary['total']} kernels: "
        f"resolved={summary['interface_resolved']}, "
        f"ambiguous={summary['interface_ambiguous']}, "
        f"not_found={summary['interface_not_found']}, "
        f"not_applicable={summary['interface_not_applicable']}",
        file=sys.stderr,
    )
    for skipped in result.skipped_roots:
        print(
            f"[locate] skipped manifest repo {skipped['name']}: "
            f"{skipped['reason']}",
            file=sys.stderr,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m framework_engineer.source_location.cli",
        description="Source-location Layer 1 locate and Layer 3 extraction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="Layer 3: slice source layers into kernel_sources/.")
    e.add_argument("--schema", type=Path, required=True)
    e.add_argument("--workspace-out", type=Path, default=None)
    e.add_argument(
        "--allow-empty",
        action="store_true",
        help="Do not hard-stop on unresolved required layers; emit placeholders "
        "(= user explicitly accepts the gap as a known risk).",
    )
    e.set_defaults(func=cmd_extract)

    loc = sub.add_parser(
        "locate", help="Layer 1: enrich a KID schema with source_locations."
    )
    loc.add_argument("--schema", type=Path, required=True)
    loc.add_argument("--manifest", type=Path, required=True)
    loc.add_argument("--sglang-repo-root", type=Path, required=True)
    loc.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write an enriched copy; defaults to atomically updating --schema.",
    )
    loc.set_defaults(func=cmd_locate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
