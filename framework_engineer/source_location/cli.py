"""The two public source-location commands: ``locate`` and ``extract``.

Progress is written to stderr, the machine-readable summary to stdout, and
configuration/contract errors return exit code 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contracts import ContractError
from .extractor import ExtractError, extract_workspace
from .locator import LocateError, locate_schema


def cmd_locate(args: argparse.Namespace) -> int:
    try:
        result = locate_schema(
            Path(args.schema),
            manifest_path=Path(args.manifest),
            sglang_repo_root=Path(args.sglang_repo_root),
            output_path=Path(args.out),
        )
    except (LocateError, ContractError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = result.summary()
    print(
        "[locate] "
        f"{summary['total']} kernels: "
        f"resolved={summary['interface_resolved']}, "
        f"ambiguous={summary['interface_ambiguous']}, "
        f"not_found={summary['interface_not_found']}",
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


def cmd_extract(args: argparse.Namespace) -> int:
    try:
        report = extract_workspace(
            Path(args.schema),
            workspace_out=Path(args.workspace_out) if args.workspace_out else None,
        )
    except (ExtractError, ContractError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for index, kernel in enumerate(report.kernels, 1):
        wrote = ",".join(kernel.layers_written) or "-"
        placeholders = ",".join(kernel.layers_placeholder) or "-"
        print(
            f"[{index}/{len(report.kernels)}] {kernel.kernel_id}: "
            f"wrote {wrote} / placeholder {placeholders}",
            file=sys.stderr,
        )
    print(json.dumps(report.summary(), indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m framework_engineer.source_location.cli",
        description="Locate Python interface candidates and extract Agent-confirmed sources.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    locate = subparsers.add_parser(
        "locate",
        help="Add transient Python interface candidates to a KID v2 schema copy.",
    )
    locate.add_argument("--schema", type=Path, required=True)
    locate.add_argument("--manifest", type=Path, required=True)
    locate.add_argument("--sglang-repo-root", type=Path, required=True)
    locate.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output schema copy; must differ from --schema.",
    )
    locate.set_defaults(func=cmd_locate)

    extract = subparsers.add_parser(
        "extract",
        help="Copy Agent-confirmed source hits and write read_hints.txt.",
    )
    extract.add_argument("--schema", type=Path, required=True)
    extract.add_argument("--workspace-out", type=Path, default=None)
    extract.set_defaults(func=cmd_extract)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
