"""KID dry-run: generate per-backend decomposition schema skeletons.

Input is a config *structurally identical* to what the real (V2) KID will
consume — ``service_cmds`` (multi-backend) + a unified ``target`` — so the
dry-run exercises the real delivery shape. It does NOT profile anything; it just
writes skeleton schemas whose value fields are auto-filled where deterministic
and ``<FILL: ...>`` where a human must decide.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import templates


class DryRunConfigError(ValueError):
    pass


@dataclass
class KidDryRunConfig:
    path: Path
    service_cmds: list[dict[str, str]]
    target: dict[str, Any]
    output_root: Path
    kernels_per_backend: int
    sglang_repo_root: Path | None

    @classmethod
    def load(cls, path: str | Path) -> "KidDryRunConfig":
        config_path = Path(path).resolve()
        if not config_path.exists():
            raise DryRunConfigError(f"config not found: {config_path}")
        raw = _load_py_or_json(config_path)

        service_cmds = raw.get("service_cmds")
        if not isinstance(service_cmds, list) or not service_cmds:
            raise DryRunConfigError("service_cmds (non-empty list of {backend_name, cmd}) is required")
        norm_cmds: list[dict[str, str]] = []
        for i, entry in enumerate(service_cmds):
            if isinstance(entry, str):
                norm_cmds.append({"backend_name": f"backend_{i}", "cmd": entry})
            elif isinstance(entry, dict):
                norm_cmds.append(
                    {"backend_name": str(entry.get("backend_name") or f"backend_{i}"), "cmd": str(entry.get("cmd", ""))}
                )
            else:
                raise DryRunConfigError(f"service_cmds[{i}] must be str or dict")

        target = raw.get("target") or {}
        if not isinstance(target, dict) or not target.get("file") or target.get("line") is None:
            raise DryRunConfigError("target.file and target.line are required")

        base = config_path.parent
        output_root = Path(str(raw.get("output_root", base / "dry_run_out"))).expanduser()
        if not output_root.is_absolute():
            output_root = (base / output_root).resolve()

        sglang_repo_root = raw.get("sglang_repo_root")
        return cls(
            path=config_path,
            service_cmds=norm_cmds,
            target=target,
            output_root=output_root,
            kernels_per_backend=int(raw.get("kernels_per_backend", 3)),
            sglang_repo_root=Path(str(sglang_repo_root)).expanduser() if sglang_repo_root else None,
        )


def _load_py_or_json(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text())
    if suffix == ".py":
        spec = importlib.util.spec_from_file_location("_kid_dryrun_config", path)
        if spec is None or spec.loader is None:
            raise DryRunConfigError(f"cannot import config module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {k: getattr(module, k) for k in dir(module) if not k.startswith("_")}
    # yaml subset: reuse KID's dependency-free parser.
    from framework_engineer.kernel_interface_decomposer.config import load_mapping

    return load_mapping(path)


@dataclass
class KidDryRunResult:
    schemas: list[Path]


def run(config: KidDryRunConfig, out: Path | None = None) -> KidDryRunResult:
    out_root = Path(out).resolve() if out else config.output_root
    workspaces = out_root / "workspaces"
    schemas: list[Path] = []
    for entry in config.service_cmds:
        backend = entry["backend_name"]
        ws = workspaces / backend
        ws.mkdir(parents=True, exist_ok=True)
        schema = templates.kid_schema_skeleton(
            backend_name=backend,
            target=config.target,
            num_kernels=config.kernels_per_backend,
        )
        schema_path = ws / f"decomposition_{backend}.schema.json"
        schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")
        schemas.append(schema_path)
    return KidDryRunResult(schemas=schemas)
