from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, DecomposerConfig
from .runner import analyze_existing_trace, run_workflow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.kernel_interface_decomposer",
        description="Profile a high-level Python API and resolve GPU kernels to wrappers/source.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run service/test under nsys and analyze the trace")
    run.add_argument("config", help="YAML/JSON config path")

    analyze = sub.add_parser("analyze", help="Analyze an existing nsys report")
    analyze.add_argument("config", help="YAML/JSON config path")
    analyze.add_argument("--nsys-rep", required=True, help="Path to .nsys-rep")
    analyze.add_argument(
        "--sqlite",
        default=None,
        help="Optional pre-exported Nsight Systems SQLite file",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = DecomposerConfig.load(args.config)
        if args.command == "run":
            schema = run_workflow(config)
        elif args.command == "analyze":
            schema = analyze_existing_trace(
                config,
                nsys_rep=Path(args.nsys_rep).expanduser().resolve(),
                sqlite_path=Path(args.sqlite).expanduser().resolve()
                if args.sqlite
                else None,
            )
        else:
            parser.error(f"unknown command: {args.command}")
            return 2
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"kernel_interface_decomposer failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"schema": str(config.schema_path()), "invocations": len(schema.get("invocations", []))}, indent=2))
    return 0

