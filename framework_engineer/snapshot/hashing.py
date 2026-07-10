"""Hash helpers for snapshot case grouping and integrity checks."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .tree import flatten_tensor_meta, get_path, is_tensor


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def short_hash(data: Any, length: int = 16) -> str:
    return hashlib.sha256(stable_json(data).encode("utf-8")).hexdigest()[:length]


def shape_hash(target: dict[str, Any], input_meta: dict[str, Any]) -> str:
    tensors = []
    for meta in flatten_tensor_meta(input_meta):
        tensors.append(
            {
                "path": meta["path"],
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "stride": meta["stride"],
                "storage_offset": meta["storage_offset"],
                "is_contiguous": meta["is_contiguous"],
            }
        )
    payload = {
        "logical_name": target.get("logical_name") or target.get("qualified_name"),
        "qualified_name": target.get("qualified_name"),
        "mode": target.get("mode"),
        "backend": target.get("backend"),
        "layer_id": target.get("layer_id"),
        "tree": _shape_tree(input_meta),
        "tensors": tensors,
    }
    return short_hash(payload, 24)


def group_key(target: dict[str, Any], input_meta: dict[str, Any]) -> str:
    """Return the Phase 1 grouping key.

    This intentionally excludes tensor values. Tensor leaves contribute dtype,
    shape, stride/layout, storage offset, and contiguity through shape_hash.
    Primitive values and None/presence are preserved by _shape_tree.
    """
    return short_hash(
        {
            "schema": "phase1.forward_windowed_shape_group.v1",
            "shape_hash": shape_hash(target, input_meta),
            "target": {
                "qualified_name": target.get("qualified_name"),
                "logical_name": target.get("logical_name"),
                "mode": target.get("mode"),
                "backend": target.get("backend"),
                "layer_id": target.get("layer_id"),
            },
        },
        24,
    )


def _shape_tree(meta: dict[str, Any]) -> Any:
    kind = meta.get("kind")
    if kind == "tensor":
        tensor = meta["meta"]
        return {
            "kind": "tensor",
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "stride": tensor["stride"],
            "is_contiguous": tensor["is_contiguous"],
        }
    if kind in ("none", "primitive"):
        return meta
    if kind in ("tuple", "list"):
        return {"kind": kind, "items": [_shape_tree(item) for item in meta.get("items", [])]}
    if kind == "dict":
        return {"kind": "dict", "items": {k: _shape_tree(v) for k, v in meta.get("items", {}).items()}}
    return meta


def semantic_features(saved_inputs: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    features: dict[str, Any] = {
        "mode": target.get("mode"),
        "backend": target.get("backend"),
    }
    if target.get("layer_id") is not None:
        features["layer_id"] = target.get("layer_id")

    kwargs = saved_inputs.get("kwargs", {}) if isinstance(saved_inputs, dict) else {}
    if "query_start_loc" in kwargs:
        features["query_start_loc"] = _sequence_feature(kwargs["query_start_loc"])
    if "cache_indices" in kwargs:
        features["cache_indices"] = _index_feature(kwargs["cache_indices"])
    if "ssm_states" in kwargs and is_tensor(kwargs["ssm_states"]):
        tensor = kwargs["ssm_states"]
        features["ssm_states"] = {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
    return features


def semantic_hash(shape_digest: str, saved_inputs: dict[str, Any], target: dict[str, Any]) -> str:
    return short_hash({"shape_hash": shape_digest, "features": semantic_features(saved_inputs, target)}, 24)


def value_hash(value: Any) -> str:
    hasher = hashlib.sha256()
    _update_value_hash(hasher, value)
    return hasher.hexdigest()[:24]


def case_key(schema_version: str, target: dict[str, Any], semantic_digest: str) -> str:
    return short_hash(
        {
            "schema_version": schema_version,
            "logical_name": target.get("logical_name") or target.get("qualified_name"),
            "semantic_hash": semantic_digest,
        },
        16,
    )


def _update_value_hash(hasher: "hashlib._Hash", value: Any) -> None:
    if is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        hasher.update(str(tensor.dtype).encode())
        hasher.update(str(list(tensor.shape)).encode())
        data = _tensor_bytes(tensor)
        if len(data) > 1_000_000:
            third = max(1, len(data) // 3)
            data = data[:third] + data[len(data) // 2 : len(data) // 2 + third] + data[-third:]
        hasher.update(data)
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        hasher.update(repr(value).encode())
        return
    if isinstance(value, tuple):
        hasher.update(b"tuple")
        for item in value:
            _update_value_hash(hasher, item)
        return
    if isinstance(value, list):
        hasher.update(b"list")
        for item in value:
            _update_value_hash(hasher, item)
        return
    if isinstance(value, dict):
        hasher.update(b"dict")
        for key in sorted(value):
            hasher.update(str(key).encode())
            _update_value_hash(hasher, value[key])
        return
    raise TypeError(f"Unsupported value for hashing: {type(value)!r}")


def _tensor_bytes(tensor: Any) -> bytes:
    """Return raw tensor bytes for value hashing.

    Some PyTorch dtypes, notably bfloat16, cannot be converted to NumPy
    directly on CPU. In those cases, hash the contiguous uint8 view instead.
    """
    try:
        return tensor.numpy().tobytes()
    except (TypeError, RuntimeError):
        try:
            return tensor.view(_torch_uint8(tensor)).numpy().tobytes()
        except Exception:
            import io
            import torch

            buffer = io.BytesIO()
            torch.save(tensor, buffer)
            return buffer.getvalue()


def _torch_uint8(tensor: Any) -> Any:
    import torch

    return torch.uint8


def _sequence_feature(value: Any) -> dict[str, Any]:
    if not is_tensor(value):
        return {"type": type(value).__name__, "repr": repr(value)[:128]}
    tensor = value.detach().cpu().flatten()
    if tensor.numel() <= 1:
        diffs = []
    else:
        diffs = (tensor[1:] - tensor[:-1]).tolist()
    preview = tensor[:16].tolist()
    return {
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "numel": int(tensor.numel()),
        "preview": preview,
        "diff_preview": diffs[:64],
        "diff_unique": sorted(set(int(x) for x in diffs))[:64],
    }


def _index_feature(value: Any) -> dict[str, Any]:
    if not is_tensor(value):
        return {"type": type(value).__name__, "repr": repr(value)[:128]}
    tensor = value.detach().cpu().flatten()
    if tensor.numel() == 0:
        return {"numel": 0}
    return {
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "numel": int(tensor.numel()),
        "min": int(tensor.min().item()),
        "max": int(tensor.max().item()),
        "has_minus_one": bool((tensor == -1).any().item()),
        "unique_count": int(tensor.unique().numel()),
        "preview": tensor[:32].tolist(),
    }
