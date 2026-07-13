"""CLI for the source-location package.

Currently exposes the Layer 3 ``extract`` subcommand (formerly
`import-decomposition`). The Layer 1 ``locate`` subcommand is future work and is
declared here as a stub so the command surface is discoverable.

Convention (matches third_party_solver/cli.py): progress -> stderr, JSON summary
-> stdout, return code 0 on success / 2 on hard stop or config error.

    python -m framework_engineer.source_location.cli extract \
        --schema <decomposition_*.schema.json> [--workspace-out <dir>] [--allow-empty]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .extractor import extract_workspace


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


def cmd_locate(args: argparse.Namespace) -> int:  # pragma: no cover - stub
    print(
        "error: `locate` (Layer 1 deterministic locator) is not implemented yet; "
        "see KID_and_locate_source_desgin_v2.md §5. Use dry_run to produce a "
        "source_locations skeleton for now.",
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m framework_engineer.source_location.cli",
        description="Source-location Layer 3 extraction (+ future Layer 1 locate).",
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

    loc = sub.add_parser("locate", help="Layer 1 deterministic locator (not implemented).")
    loc.set_defaults(func=cmd_locate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
