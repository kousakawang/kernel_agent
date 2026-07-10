"""Snapshot replay benchmark harness."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict

import candidate_impl
import reference_impl
import snapshot_runtime


CANDIDATE_FUNCTION = "candidate"


def _torch():
    try:
        import torch
    except Exception:
        return None
    return torch


def sync() -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _call(fn, call_tree):
    return fn(*call_tree.get("args", ()), **call_tree.get("kwargs", {}))


def elapsed_us(fn, make_inputs, *, warmup: int, repeat: int, use_cuda_events: bool) -> dict:
    torch = _torch()
    for _ in range(warmup):
        _call(fn, make_inputs())
    sync()

    values = []
    for _ in range(repeat):
        inputs = make_inputs()
        sync()
        if use_cuda_events and torch is not None and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _call(fn, inputs)
            end.record()
            sync()
            values.append(float(start.elapsed_time(end) * 1000.0))
        else:
            start_t = time.perf_counter()
            _call(fn, inputs)
            sync()
            values.append((time.perf_counter() - start_t) * 1_000_000.0)
    return {
        "median_us": statistics.median(values),
        "mean_us": statistics.mean(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def benchmark_sample(group_meta, sample_meta, *, device: str, target: str, warmup: int, repeat: int) -> dict:
    sample = snapshot_runtime.load_sample(group_meta["group_id"], sample_meta["sample_id"], device=device)
    snapshot_runtime.set_current_sample(sample)

    def make_inputs():
        return snapshot_runtime.tree_clone(sample["pre_inputs"])

    result = {
        "record_type": "sample",
        "group_id": group_meta["group_id"],
        "sample_id": sample_meta["sample_id"],
        "warmup": warmup,
        "repeat": repeat,
    }
    use_events = device.startswith("cuda")
    candidate = getattr(candidate_impl, CANDIDATE_FUNCTION)
    if target in ("reference", "both"):
        result["reference"] = elapsed_us(reference_impl.reference, make_inputs, warmup=warmup, repeat=repeat, use_cuda_events=use_events)
    if target in ("candidate", "both"):
        result["candidate"] = elapsed_us(candidate, make_inputs, warmup=warmup, repeat=repeat, use_cuda_events=use_events)
    if "reference" in result and "candidate" in result and result["candidate"]["median_us"] > 0:
        result["speedup_median"] = result["reference"]["median_us"] / result["candidate"]["median_us"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", choices=["reference", "candidate", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--all-priorities", action="store_true")
    args = parser.parse_args()

    priority = None if args.all_priorities else "required"
    selected = snapshot_runtime.list_samples(group_id=args.group_id, sample_id=args.sample_id, priority=priority)
    by_group = defaultdict(list)
    for group, sample in selected:
        result = benchmark_sample(group, sample, device=args.device, target=args.target, warmup=args.warmup, repeat=args.repeat)
        by_group[group["group_id"]].append(result)
        print(json.dumps(result, sort_keys=True))

    for group_id, rows in sorted(by_group.items()):
        summary = {"record_type": "group_summary", "group_id": group_id, "sample_count": len(rows)}
        for target_name in ("reference", "candidate"):
            medians = [row[target_name]["median_us"] for row in rows if target_name in row]
            if medians:
                summary[target_name] = {
                    "median_of_sample_medians_us": statistics.median(medians),
                    "mean_of_sample_medians_us": statistics.mean(medians),
                    "min_sample_median_us": min(medians),
                    "max_sample_median_us": max(medians),
                }
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
