"""Standalone snapshot replay runtime copied into task packs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def _torch():
    try:
        import torch
    except Exception:
        return None
    return torch


CURRENT_SAMPLE = None


def load_manifest(root: Path = Path("snapshots")) -> dict[str, Any]:
    return json.loads((root / "manifest.json").read_text())


def list_groups(root: Path = Path("snapshots"), priority: str | None = "required") -> list[dict[str, Any]]:
    manifest = load_manifest(root)
    groups = manifest.get("case_groups", [])
    if priority is None:
        return groups
    return [group for group in groups if group.get("selection", {}).get("priority") == priority]


def list_samples(
    root: Path = Path("snapshots"),
    *,
    group_id: str | None = None,
    sample_id: str | None = None,
    priority: str | None = "required",
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out = []
    for group in list_groups(root, priority=priority):
        if group_id is not None and group["group_id"] != group_id:
            continue
        for sample in group.get("samples", []):
            if sample_id is not None and sample["sample_id"] != sample_id:
                continue
            out.append((group, sample))
    return out


def load_sample(group_id: str, sample_id: str, root: Path = Path("snapshots"), device: str = "cuda") -> dict[str, Any]:
    group_dir = root / "selected" / group_id
    sample_dir = group_dir / "samples" / sample_id
    group_meta = json.loads((group_dir / "group_meta.json").read_text())
    sample_meta = json.loads((sample_dir / "meta.json").read_text())
    pre_inputs = _load_payload(sample_dir / sample_meta["files"]["pre_inputs"])
    post_inputs = _load_payload(sample_dir / sample_meta["files"]["post_inputs"])
    outputs = _load_payload(sample_dir / sample_meta["files"]["outputs"])
    return {
        "group": group_meta,
        "sample_meta": sample_meta,
        "pre_inputs": tree_to_device(pre_inputs, device),
        "post_inputs": tree_to_device(post_inputs, device),
        "outputs": tree_to_device(outputs, device),
    }


def _load_payload(path: Path) -> Any:
    torch = _torch()
    if torch is not None:
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            pass
    with path.open("rb") as f:
        return pickle.load(f)


def set_current_sample(sample: dict[str, Any]) -> None:
    global CURRENT_SAMPLE
    CURRENT_SAMPLE = sample


def get_current_sample() -> dict[str, Any]:
    if CURRENT_SAMPLE is None:
        raise RuntimeError("No current snapshot sample is set.")
    return CURRENT_SAMPLE


def tree_clone(value: Any) -> Any:
    torch = _torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.clone()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return tuple(tree_clone(v) for v in value)
    if isinstance(value, list):
        return [tree_clone(v) for v in value]
    if isinstance(value, dict):
        return {k: tree_clone(v) for k, v in value.items()}
    raise TypeError(f"Unsupported snapshot value for clone: {type(value)!r}")


def tree_to_device(value: Any, device: str) -> Any:
    torch = _torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.to(device)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return tuple(tree_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [tree_to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: tree_to_device(v, device) for k, v in value.items()}
    raise TypeError(f"Unsupported snapshot value for device transfer: {type(value)!r}")


def get_path(tree: Any, path: str) -> Any:
    cur = tree
    for part in [p for p in path.split(".") if p]:
        if isinstance(cur, dict):
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            cur = cur[int(part)]
        else:
            raise KeyError(f"Cannot descend into {type(cur)!r} at {part!r}")
    return cur


def set_path(tree: Any, path: str, value: Any) -> None:
    parts = [p for p in path.split(".") if p]
    cur = tree
    for part in parts[:-1]:
        if isinstance(cur, dict):
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            cur = cur[int(part)]
        else:
            raise KeyError(f"Cannot descend into {type(cur)!r} at {part!r}")
    last = parts[-1]
    torch = _torch()
    if isinstance(cur, dict):
        old = cur.get(last)
        if torch is not None and isinstance(old, torch.Tensor) and isinstance(value, torch.Tensor):
            old.copy_(value)
        else:
            cur[last] = value
    elif isinstance(cur, list):
        index = int(last)
        old = cur[index]
        if torch is not None and isinstance(old, torch.Tensor) and isinstance(value, torch.Tensor):
            old.copy_(value)
        else:
            cur[index] = value
    elif isinstance(cur, tuple):
        index = int(last)
        old = cur[index]
        if torch is not None and isinstance(old, torch.Tensor) and isinstance(value, torch.Tensor):
            old.copy_(value)
        else:
            raise TypeError(f"Cannot assign non-tensor value into tuple path {path!r}")
    else:
        raise KeyError(f"Cannot set path {path!r}")


def apply_snapshot_mutations(call_tree: dict[str, Any], sample: dict[str, Any]) -> None:
    sample_meta = sample["sample_meta"]
    post_inputs = sample["post_inputs"]
    for path in sample_meta.get("mutation", {}).get("mutable_arg_paths", []):
        set_path(call_tree, path, tree_clone(get_path(post_inputs, path)))


def assert_tree_close(actual: Any, expected: Any, *, atol: float, rtol: float, path: str = "") -> None:
    torch = _torch()
    if torch is not None and isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f"{path}: actual is not a tensor")
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    if expected is None or isinstance(expected, (str, int, float, bool)):
        if actual != expected:
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, tuple):
        if not isinstance(actual, tuple) or len(actual) != len(expected):
            raise AssertionError(f"{path}: tuple mismatch")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_tree_close(a, e, atol=atol, rtol=rtol, path=f"{path}.{i}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list mismatch")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_tree_close(a, e, atol=atol, rtol=rtol, path=f"{path}.{i}")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{path}: dict key mismatch")
        for key in expected:
            assert_tree_close(actual[key], expected[key], atol=atol, rtol=rtol, path=f"{path}.{key}")
        return
    raise TypeError(f"Unsupported snapshot value for comparison: {type(expected)!r}")
