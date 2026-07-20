"""CLI used by the prompt-driven KID Agent semantic phase."""

from __future__ import annotations

import argparse
import json
import sys

from .config import ConfigError
from .semantic_resolver import (
    SemanticResolver,
    SemanticResolverConfig,
    SemanticResolverError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "framework_engineer.kernel_interface_decomposer.semantic_resolver_tools"
        ),
        description="Prepare, materialize, and validate KID semantic decisions.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "finalize", "validate"):
        command = commands.add_parser(name)
        command.add_argument("config", help="kid-semantic-resolver-config/v2 path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = SemanticResolverConfig.load(args.config)
        resolver = SemanticResolver(config)
        if args.command == "prepare":
            context = resolver.prepare()
            owner_count = sum(
                len(item.get("owner_captures", []))
                for item in context.get("invocations", [])
            )
            report = {
                "context_output": str(config.context_output),
                "invocations": len(context.get("invocations", [])),
                "direct_owner_captures": owner_count,
            }
        elif args.command == "finalize":
            final = resolver.finalize()
            report = {
                "output": str(config.output),
                "semantic_targets": len(final.get("kernels", [])),
                "min_coverage": final.get("coverage_report", {}).get("min_coverage"),
            }
        else:
            errors = resolver.validate()
            if errors:
                print(
                    f"KID Semantic Resolver validation FAILED ({len(errors)} errors)",
                    file=sys.stderr,
                )
                for error in errors:
                    print(f"  - {error}", file=sys.stderr)
                return 1
            final = json.loads(config.output.read_text(encoding="utf-8"))
            report = {
                "valid": True,
                "output": str(config.output),
                "semantic_targets": len(final.get("kernels", [])),
            }
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except (SemanticResolverError, OSError, json.JSONDecodeError) as exc:
        print(f"KID Semantic Resolver failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
