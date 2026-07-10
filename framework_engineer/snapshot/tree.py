"""Tree serialization helpers for snapshot capture and replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import SnapshotTensorMeta


PRIMITIVE_TYPES = (str, int, float, bool)


def _torch():
    try:
        import torch
    except Exception:  # pragma: no cover - torch may be absent on docs-only hosts.
        return None
    return torch


def is_tensor(value: Any) -> bool:
    torch = _torch()
    return bool(torch is not None and isinstance(value, torch.Tensor))


def tensor_meta(value: Any, path: str) -> SnapshotTensorMeta:
    return SnapshotTensorMeta(
        path=path,
        dtype=str(value.dtype).replace("torch.", ""),
        shape=list(value.shape),
        stride=list(value.stride()),
        device_type=value.device.type,
        device_index=value.device.index,
        layout=str(value.layout).replace("torch.", ""),
        is_contiguous=bool(value.is_contiguous()),
        requires_grad=bool(value.requires_grad),
        numel=int(value.numel()),
        storage_offset=int(value.storage_offset()),
    )


def tree_to_cpu(value: Any) -> Any:
    """Detach tensor leaves and copy them to CPU; preserve plain containers."""
    if is_tensor(value):
        return value.detach().cpu().clone()
    if value is None or isinstance(value, PRIMITIVE_TYPES):
        return value
    if isinstance(value, tuple):
        return tuple(tree_to_cpu(v) for v in value)
    if isinstance(value, list):
        return [tree_to_cpu(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): tree_to_cpu(v) for k, v in value.items()}
    raise TypeError(f"Unsupported snapshot value at capture time: {type(value)!r}")


def tree_clone(value: Any) -> Any:
    if is_tensor(value):
        return value.clone()
    if value is None or isinstance(value, PRIMITIVE_TYPES):
        return value
    if isinstance(value, tuple):
        return tuple(tree_clone(v) for v in value)
    if isinstance(value, list):
        return [tree_clone(v) for v in value]
    if isinstance(value, Mapping):
        return {k: tree_clone(v) for k, v in value.items()}
    raise TypeError(f"Unsupported snapshot value for clone: {type(value)!r}")


def tree_to_device(value: Any, device: str) -> Any:
    if is_tensor(value):
        return value.to(device)
    if value is None or isinstance(value, PRIMITIVE_TYPES):
        return value
    if isinstance(value, tuple):
        return tuple(tree_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [tree_to_device(v, device) for v in value]
    if isinstance(value, Mapping):
        return {k: tree_to_device(v, device) for k, v in value.items()}
    raise TypeError(f"Unsupported snapshot value for device transfer: {type(value)!r}")


def tree_meta(value: Any, path: str = "") -> dict[str, Any]:
    if is_tensor(value):
        return {"kind": "tensor", "meta": tensor_meta(value, path).to_dict()}
    if value is None:
        return {"kind": "none"}
    if isinstance(value, PRIMITIVE_TYPES):
        return {"kind": "primitive", "type": type(value).__name__, "value": value}
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [tree_meta(v, f"{path}.{i}" if path else str(i)) for i, v in enumerate(value)],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [tree_meta(v, f"{path}.{i}" if path else str(i)) for i, v in enumerate(value)],
        }
    if isinstance(value, Mapping):
        return {
            "kind": "dict",
            "items": {
                str(k): tree_meta(v, f"{path}.{k}" if path else str(k))
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            },
        }
    raise TypeError(f"Unsupported snapshot value for metadata: {type(value)!r}")


def flatten_tensor_meta(meta: dict[str, Any]) -> list[dict[str, Any]]:
    kind = meta.get("kind")
    if kind == "tensor":
        return [meta["meta"]]
    if kind in ("tuple", "list"):
        out: list[dict[str, Any]] = []
        for item in meta.get("items", []):
            out.extend(flatten_tensor_meta(item))
        return out
    if kind == "dict":
        out = []
        for item in meta.get("items", {}).values():
            out.extend(flatten_tensor_meta(item))
        return out
    return []


def get_path(tree: Any, path: str) -> Any:
    cur = tree
    for part in path.split("."):
        if part == "":
            continue
        if isinstance(cur, Mapping):
            cur = cur[part]
        elif isinstance(cur, Sequence) and not isinstance(cur, (str, bytes)):
            cur = cur[int(part)]
        else:
            raise KeyError(f"Cannot descend into {type(cur)!r} at {part!r} for path {path!r}")
    return cur


def set_path(tree: Any, path: str, value: Any) -> None:
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise KeyError("Cannot set empty path")
    cur = tree
    for part in parts[:-1]:
        if isinstance(cur, Mapping):
            cur = cur[part]
        elif isinstance(cur, list):
            cur = cur[int(part)]
        else:
            raise KeyError(f"Cannot descend into {type(cur)!r} at {part!r} for path {path!r}")
    last = parts[-1]
    if isinstance(cur, Mapping):
        if last in cur and is_tensor(cur[last]) and is_tensor(value):
            cur[last].copy_(value)
        else:
            cur[last] = value
    elif isinstance(cur, list):
        index = int(last)
        if is_tensor(cur[index]) and is_tensor(value):
            cur[index].copy_(value)
        else:
            cur[index] = value
    else:
        raise KeyError(f"Cannot set path {path!r} on {type(cur)!r}")


def assert_tree_close(actual: Any, expected: Any, *, atol: float, rtol: float, path: str = "") -> None:
    torch = _torch()
    if is_tensor(expected):
        if not is_tensor(actual):
            raise AssertionError(f"{path}: actual is not a tensor")
        assert torch is not None
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    if expected is None or isinstance(expected, PRIMITIVE_TYPES):
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
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise AssertionError(f"{path}: dict key mismatch")
        for key in expected:
            assert_tree_close(actual[key], expected[key], atol=atol, rtol=rtol, path=f"{path}.{key}")
        return
    raise TypeError(f"Unsupported snapshot value for comparison: {type(expected)!r}")
