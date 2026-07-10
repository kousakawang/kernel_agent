"""Generic snapshot-golden reference implementation template."""

from __future__ import annotations

import snapshot_runtime


def reference(*args, **kwargs):
    sample = snapshot_runtime.get_current_sample()
    call_tree = {"args": list(args), "kwargs": kwargs}
    snapshot_runtime.apply_snapshot_mutations(call_tree, sample)
    return snapshot_runtime.tree_clone(sample["outputs"])
