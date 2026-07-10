"""
Benchmark harness template for Qwen3.5 GDN linear attention core.

Framework Engineer owns this file. Kernel Engineer should not change timing
rules; optimize candidate_impl instead.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

import candidate_impl
from correctness_test import clone_inputs, load_shape_cases, make_inputs
import reference_impl


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_call(fn: Callable[[], object], *, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    sync()

    values = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        sync()
        values.append((time.perf_counter() - start) * 1_000_000)

    return {
        "median_us": statistics.median(values),
        "mean_us": statistics.mean(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def run_reference(inputs: dict[str, torch.Tensor]):
    return reference_impl.reference_extend(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        ssm_states=inputs["ssm_states"],
        cache_indices=inputs["cache_indices"],
        query_start_loc=inputs["query_start_loc"],
    )


def run_candidate(inputs: dict[str, torch.Tensor]):
    return candidate_impl.candidate_extend(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        ssm_states=inputs["ssm_states"],
        cache_indices=inputs["cache_indices"],
        query_start_loc=inputs["query_start_loc"],
    )


def benchmark_case(case: dict[str, Any], *, device: str, seed: int, warmup: int, repeat: int) -> None:
    inputs = make_inputs(case, device=device, seed=seed)
    ref_inputs = clone_inputs(inputs)
    cand_inputs = clone_inputs(inputs)

    ref = time_call(lambda: run_reference(ref_inputs), warmup=warmup, repeat=repeat)
    cand = time_call(lambda: run_candidate(cand_inputs), warmup=warmup, repeat=repeat)
    speedup = ref["median_us"] / cand["median_us"] if cand["median_us"] > 0 else 0.0
    print(
        json.dumps(
            {
                "case_id": case["case_id"],
                "reference": ref,
                "candidate": cand,
                "speedup_median": speedup,
                "warmup": warmup,
                "repeat": repeat,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-list", type=Path, default=Path("shape_list.json"))
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    for case in load_shape_cases(args.shape_list):
        if args.case_id is None or case["case_id"] == args.case_id:
            benchmark_case(case, device=args.device, seed=args.seed, warmup=args.warmup, repeat=args.repeat)


if __name__ == "__main__":
    main()
