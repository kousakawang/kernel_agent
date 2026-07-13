"""Config loading for the resolve-third-party skill.

Accepts a Python config (preferred, matches framework_engineer convention) or a
JSON/YAML file. YAML uses a tiny subset parser (borrowed shape from KID's config)
so the target environment does not need PyYAML.

Config fields
-------------
service_cmds : list[dict]        # [{"backend_name": str, "cmd": str}, ...]  (required)
sglang_repo_root : str           # contains sgl-kernel/ source tree            (required)
third_party_cache : str          # clone destination, keyed by (name, version) (required)
output_root : str                # where manifest + missing_repos.md land      (required)
workload_cmds : list[str]        # optional, for agent-side runtime confirmation
explicit_paths : dict[str, str]  # optional, name -> P1 user-provided path
extra_env : dict[str, str]       # optional, e.g. {"PYTHONPATH": ...}
https_proxy : str                # optional, proxy for all git clones (https_proxy=...)
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass
class ResolveConfig:
    path: Path
    service_cmds: list[dict[str, str]]
    sglang_repo_root: Path
    third_party_cache: Path
    output_root: Path
    workload_cmds: list[str] = field(default_factory=list)
    explicit_paths: dict[str, str] = field(default_factory=dict)
    extra_env: dict[str, str] = field(default_factory=dict)
    https_proxy: str | None = None

    @property
    def manifest_path(self) -> Path:
        return self.output_root / "third_party_manifest.json"

    @property
    def missing_repos_path(self) -> Path:
        return self.output_root / "missing_repos.md"

    @property
    def sgl_kernel_src(self) -> Path:
        return self.sglang_repo_root / "sgl-kernel"


def _load_python_config(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("_resolve_third_party_config", path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"cannot import python config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: getattr(module, k) for k in dir(module) if not k.startswith("_")}


def _load_json_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ConfigError("top-level JSON config must be an object")
    return data


def _abspath(value: Any, *, base: Path, field_name: str) -> Path:
    if not value:
        raise ConfigError(f"'{field_name}' is required")
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _normalize_service_cmds(raw: Any) -> list[dict[str, str]]:
    if not raw:
        raise ConfigError("'service_cmds' is required and must be non-empty")
    out: list[dict[str, str]] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            out.append({"backend_name": f"cmd{i}", "cmd": item})
        elif isinstance(item, dict) and item.get("cmd"):
            out.append(
                {
                    "backend_name": str(item.get("backend_name") or f"cmd{i}"),
                    "cmd": str(item["cmd"]),
                }
            )
        else:
            raise ConfigError(f"service_cmds[{i}] must be a str or {{backend_name, cmd}}")
    return out


def load_config(path: str | Path) -> ResolveConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"config not found: {config_path}")
    base = config_path.parent

    suffix = config_path.suffix.lower()
    if suffix == ".py":
        raw = _load_python_config(config_path)
    elif suffix in {".json"}:
        raw = _load_json_config(config_path)
    else:
        # YAML subset: reuse KID's parser to avoid a PyYAML dependency.
        from framework_engineer.kernel_interface_decomposer.config import load_mapping

        raw = load_mapping(config_path)

    return ResolveConfig(
        path=config_path,
        service_cmds=_normalize_service_cmds(raw.get("service_cmds")),
        sglang_repo_root=_abspath(
            raw.get("sglang_repo_root"), base=base, field_name="sglang_repo_root"
        ),
        third_party_cache=_abspath(
            raw.get("third_party_cache"), base=base, field_name="third_party_cache"
        ),
        output_root=_abspath(raw.get("output_root"), base=base, field_name="output_root"),
        workload_cmds=list(raw.get("workload_cmds") or []),
        explicit_paths={str(k): str(v) for k, v in (raw.get("explicit_paths") or {}).items()},
        extra_env={str(k): str(v) for k, v in (raw.get("extra_env") or {}).items()},
        https_proxy=(str(raw["https_proxy"]) if raw.get("https_proxy") else None),
    )
