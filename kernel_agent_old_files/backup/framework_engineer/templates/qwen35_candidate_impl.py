"""Generic candidate implementation template.

Generated task packs now use candidate(*args, **kwargs). Kernel Engineer should
replace candidate() with the optimized implementation for the selected target.
"""

from __future__ import annotations

import reference_impl


def candidate(*args, **kwargs):
    return reference_impl.reference(*args, **kwargs)
