"""
Correctness harness template for Qwen3.5 GDN linear attention core.

Framework Engineer owns this file. It should load cases from shape_list.json,
construct framework-faithful inputs, and compare reference_impl against
candidate_impl.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

import candidate_impl
import reference_impl


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }[name]


def load_shape_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return data["shape_cases"]


def make_tensor(spec: dict[str, Any], *, device: str, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    dtype = _dtype(spec["dtype"])
    if "values" in spec:
        return torch.tensor(spec["values"], dtype=dtype, device=device)
    shape = spec["shape"]
    if dtype in (torch.int32, torch.int64):
        return torch.zeros(shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)


def make_inputs(case: dict[str, Any], *, device: str, seed: int) -> dict[str, torch.Tensor]:
    specs = case["inputs"]
    return {name: make_tensor(spec, device=device, seed=seed + i) for i, (name, spec) in enumerate(specs.items())}


def clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in inputs.items()}


def assert_outputs_close(actual, expected, *, atol: float, rtol: float) -> None:
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple), "candidate returned non-tuple output"
        assert len(actual) == len(expected), "candidate tuple length mismatch"
        pairs = zip(actual, expected)
    else:
        pairs = [(actual, expected)]

    for idx, (a, e) in enumerate(pairs):
        if a is None or e is None:
            assert a is e, f"output {idx} None mismatch"
            continue
        torch.testing.assert_close(a, e, atol=atol, rtol=rtol)


def run_case(case: dict[str, Any], *, device: str, seed: int) -> None:
    inputs = make_inputs(case, device=device, seed=seed)
    ref_inputs = clone_inputs(inputs)
    cand_inputs = clone_inputs(inputs)

    expected = reference_impl.reference_extend(
        ref_inputs["q"],
        ref_inputs["k"],
        ref_inputs["v"],
        ref_inputs["g"],
        ref_inputs["beta"],
        ssm_states=ref_inputs["ssm_states"],
        cache_indices=ref_inputs["cache_indices"],
        query_start_loc=ref_inputs["query_start_loc"],
    )
    actual = candidate_impl.candidate_extend(
        cand_inputs["q"],
        cand_inputs["k"],
        cand_inputs["v"],
        cand_inputs["g"],
        cand_inputs["beta"],
        ssm_states=cand_inputs["ssm_states"],
        cache_indices=cand_inputs["cache_indices"],
        query_start_loc=cand_inputs["query_start_loc"],
    )

    tol = case.get("tolerance", {})
    assert_outputs_close(actual, expected, atol=tol.get("atol", 2e-2), rtol=tol.get("rtol", 2e-2))
    print(f"[correctness] {case['case_id']}: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-list", type=Path, default=Path("shape_list.json"))
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cases = load_shape_cases(args.shape_list)
    for case in cases:
        if args.case_id is None or case["case_id"] == args.case_id:
            run_case(copy.deepcopy(case), device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
