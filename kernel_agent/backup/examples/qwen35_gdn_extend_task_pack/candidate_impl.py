"""Candidate implementation generated for Kernel Engineer.

The initial candidate delegates to the original captured target when available
so benchmark.py starts with a real baseline. Kernel Engineer should replace
candidate().
"""

from __future__ import annotations

import original_impl
import reference_impl


def candidate(*args, **kwargs):
    try:
        return original_impl.original(*args, **kwargs)
    except original_impl.OriginalUnavailableError:
        return reference_impl.snapshot_reference(*args, **kwargs)


if "candidate" != "candidate":
    globals()["candidate"] = candidate
