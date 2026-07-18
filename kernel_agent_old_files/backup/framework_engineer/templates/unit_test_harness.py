"""
UnitTestHarness template for a kernel optimization task.

This file is intentionally a template. Framework Engineer fills in the golden
semantics and shape cases before handing it to Kernel Agent. Kernel Agent fills
or imports candidate_kernel from its implementation workspace.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class ShapeCase:
    name: str
    tokens: int
    batch_size: int
    heads: int
    kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.bfloat16


SHAPE_CASES: tuple[ShapeCase, ...] = (
    ShapeCase("prefill_hot_2k", tokens=2048, batch_size=1, heads=32, kv_heads=32, head_dim=128),
    ShapeCase("prefill_hot_16k", tokens=16384, batch_size=1, heads=32, kv_heads=32, head_dim=128),
    ShapeCase("decode_bs32", tokens=32, batch_size=32, heads=32, kv_heads=32, head_dim=128),
)


def make_inputs(case: ShapeCase, device: str = "cuda", seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn(case.tokens, case.heads, case.head_dim, device=device, dtype=case.dtype)
    k = torch.randn(case.tokens, case.kv_heads, case.head_dim, device=device, dtype=case.dtype)
    v = torch.randn(case.tokens, case.kv_heads, case.head_dim, device=device, dtype=case.dtype)

    # Replace this with real metadata from the framework path when needed.
    cu_seqlens = torch.tensor([0, case.tokens], device=device, dtype=torch.int32)

    return {
        "q": q,
        "k": k,
        "v": v,
        "cu_seqlens": cu_seqlens,
    }


def golden_impl(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """PyTorch golden implementation.

    Framework Engineer must replace this placeholder with the exact operator
    semantics for the task. Correctness is defined against this function.
    """
    raise NotImplementedError("Fill in PyTorch golden semantics before use.")


def candidate_kernel(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Candidate kernel interface.

    Kernel Agent should replace this function or import an implementation with
    the same signature.
    """
    raise NotImplementedError("Kernel Agent must provide candidate implementation.")


def iter_cases(selected: str | None) -> Iterable[ShapeCase]:
    for case in SHAPE_CASES:
        if selected is None or selected == case.name:
            yield case


def run_correctness(case_name: str | None, atol: float, rtol: float) -> None:
    for case in iter_cases(case_name):
        inputs = make_inputs(case)
        expected = golden_impl(inputs)
        actual = candidate_kernel(inputs)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        print(f"[correctness] {case.name}: PASS")


def benchmark_one(case: ShapeCase, warmup: int, repeat: int) -> None:
    inputs = make_inputs(case)

    for _ in range(warmup):
        candidate_kernel(inputs)
    torch.cuda.synchronize()

    timings_us: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        candidate_kernel(inputs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        timings_us.append((end - start) * 1_000_000)

    median_us = statistics.median(timings_us)
    mean_us = statistics.mean(timings_us)
    print(
        f"[benchmark] {case.name}: median_us={median_us:.3f}, "
        f"mean_us={mean_us:.3f}, repeat={repeat}, warmup={warmup}"
    )


def run_benchmark(case_name: str | None, warmup: int, repeat: int) -> None:
    for case in iter_cases(case_name):
        benchmark_one(case, warmup=warmup, repeat=repeat)


def print_ncu_command(case_name: str | None) -> None:
    selected_case = case_name or SHAPE_CASES[0].name
    print(
        "ncu --set full --target-processes all "
        f"python unit_test_harness.py --mode benchmark --case {selected_case} --warmup 5 --repeat 20"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["correctness", "benchmark", "ncu-command"], required=True)
    parser.add_argument("--case", default=None)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    if args.mode == "correctness":
        run_correctness(args.case, atol=args.atol, rtol=args.rtol)
    elif args.mode == "benchmark":
        run_benchmark(args.case, warmup=args.warmup, repeat=args.repeat)
    else:
        print_ncu_command(args.case)


if __name__ == "__main__":
    main()
