from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, RuntimeCaptureConfig
from .runner import analyze_existing_trace, capture_runtime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m framework_engineer.kernel_interface_decomposer",
        description="Capture execution-level GPU kernel evidence for one high-level Python target.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser(
        "capture", help="Run one backend/test workflow under Nsight Systems"
    )
    capture.add_argument("config", help="kid-runtime-config/v2 JSON or YAML path")

    analyze = subparsers.add_parser(
        "analyze", help="Rebuild Runtime Capture output from existing SQLite/JSONL"
    )
    analyze.add_argument("config", help="kid-runtime-config/v2 JSON or YAML path")
    analyze.add_argument("--sqlite", required=True, help="Nsight Systems SQLite path")
    analyze.add_argument(
        "--events-dir", required=True, help="Directory containing events_<pid>.jsonl"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = RuntimeCaptureConfig.load(args.config)
        if args.command == "capture":
            result = capture_runtime(config)
        else:
            result = analyze_existing_trace(
                config,
                sqlite_path=Path(args.sqlite).expanduser().resolve(),
                events_dir=Path(args.events_dir).expanduser().resolve(),
            )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"KID Runtime Capture failed: {exc}", file=sys.stderr)
        return 1

    diagnostics = result.get("diagnostics", {})
    print(
        json.dumps(
            {
                "runtime_capture": str(config.schema_path()),
                "backend_name": config.backend_name,
                "selected_invocations": len(result.get("invocations", [])),
                "kernels": len(result.get("kernels", [])),
                "raw_capture_events": diagnostics.get("capture_event_count", 0),
            },
            indent=2,
        )
    )
    return 0
