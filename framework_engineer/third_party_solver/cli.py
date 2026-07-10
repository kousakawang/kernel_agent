"""CLI for the resolve-third-party skill.

Run as:

    python -m framework_engineer.third_party_solver.cli resolve --config <path> [--dry-run]

`--dry-run` resolves versions and writes the manifest (with re-runnable clone
commands) but performs no network clone. Use it to review versions / version
mismatches before committing to clones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import cloner, manifest as manifest_mod, version_resolver
from .config import ConfigError, load_config
from .flags import annotate_universe


def cmd_resolve(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if not config.sgl_kernel_src.exists():
        print(
            f"error: sgl-kernel source tree not found at {config.sgl_kernel_src}",
            file=sys.stderr,
        )
        return 2

    triggered_by_map = annotate_universe(config.service_cmds)

    resolutions, meta = version_resolver.resolve_all(
        sgl_kernel_src=config.sgl_kernel_src,
        triggered_by_map=triggered_by_map,
    )

    records: list[manifest_mod.RepoRecord] = []
    for res in resolutions:
        outcome = cloner.clone_repo(
            res,
            cache_root=config.third_party_cache,
            sgl_kernel_src=config.sgl_kernel_src,
            explicit_paths=config.explicit_paths,
            dry_run=args.dry_run,
            clone_timeout=args.clone_timeout,
        )
        records.append(manifest_mod.build_record(res, outcome))

    manifest = manifest_mod.build_manifest(
        sglang_repo_root=config.sglang_repo_root,
        third_party_cache=config.third_party_cache,
        meta=meta,
        records=records,
    )
    manifest_mod.write_manifest(manifest, config.manifest_path)
    wrote_missing = manifest_mod.write_missing_repos(manifest, config.missing_repos_path)

    ok = sum(1 for r in records if r.status == "ok")
    clone_failed = sum(1 for r in records if r.status == "clone_failed")
    failed = sum(1 for r in records if r.status == "failed")
    summary = {
        "manifest": str(config.manifest_path),
        "missing_repos": str(config.missing_repos_path) if wrote_missing else None,
        "dry_run": args.dry_run,
        "counts": {
            "total": len(records),
            "ok": ok,
            "clone_failed": clone_failed,
            "failed": failed,
        },
        "sgl_kernel_version_mismatch": meta.get("sgl_kernel_version_mismatch"),
    }
    print(json.dumps(summary, indent=2))
    # Non-zero only on hard config/env errors; clone_failed/failed are reported in
    # the manifest and are not treated as CLI failures (out-of-scope to fix here).
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m framework_engineer.third_party_solver.cli",
        description="Resolve + clone GPU-inference third-party repos (Step 0.5).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("resolve", help="Resolve versions and clone missing repos.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve + write manifest with clone commands, but do not clone.",
    )
    p.add_argument("--clone-timeout", type=int, default=600)
    p.set_defaults(func=cmd_resolve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
