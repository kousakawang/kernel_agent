"""Snapshot replay correctness harness."""

from __future__ import annotations

import argparse
import json

import candidate_impl
import reference_impl
import snapshot_runtime


CANDIDATE_FUNCTION = "candidate"


def _call(fn, call_tree):
    return fn(*call_tree.get("args", ()), **call_tree.get("kwargs", {}))


def run_sample(group_meta, sample_meta, *, device: str, mode: str) -> dict:
    sample = snapshot_runtime.load_sample(group_meta["group_id"], sample_meta["sample_id"], device=device)
    snapshot_runtime.set_current_sample(sample)
    tol = sample["sample_meta"].get("tolerance", {})
    atol = float(tol.get("atol", 2e-2))
    rtol = float(tol.get("rtol", 2e-2))

    ref_tree = snapshot_runtime.tree_clone(sample["pre_inputs"])
    cand_tree = snapshot_runtime.tree_clone(sample["pre_inputs"])

    if mode == "reference-replay":
        expected = _call(reference_impl.reference, ref_tree)
        expected_mut_tree = ref_tree
    else:
        expected = snapshot_runtime.tree_clone(sample["outputs"])
        expected_mut_tree = sample["post_inputs"]

    candidate = getattr(candidate_impl, CANDIDATE_FUNCTION)
    actual = _call(candidate, cand_tree)
    snapshot_runtime.assert_tree_close(actual, expected, atol=atol, rtol=rtol)

    for path in sample["sample_meta"].get("mutation", {}).get("mutable_arg_paths", []):
        snapshot_runtime.assert_tree_close(
            snapshot_runtime.get_path(cand_tree, path),
            snapshot_runtime.get_path(expected_mut_tree, path),
            atol=atol,
            rtol=rtol,
            path=path,
        )

    return {
        "group_id": group_meta["group_id"],
        "sample_id": sample_meta["sample_id"],
        "status": "PASS",
        "mode": mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["reference-replay", "snapshot-golden"], default="snapshot-golden")
    parser.add_argument("--all-priorities", action="store_true")
    args = parser.parse_args()

    priority = None if args.all_priorities else "required"
    selected = snapshot_runtime.list_samples(group_id=args.group_id, sample_id=args.sample_id, priority=priority)
    for group, sample in selected:
        print(json.dumps(run_sample(group, sample, device=args.device, mode=args.mode), sort_keys=True))


if __name__ == "__main__":
    main()
