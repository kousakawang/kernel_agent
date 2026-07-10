"""Third-party repository resolver for Step 0.5 (`resolve-third-party` skill).

Deterministic helpers that:
  1. know the fixed universe of GPU-inference third-party repos (registry.py),
  2. annotate which backend path triggered each repo (flags.py),
  3. resolve the correct version per repo (version_resolver.py + cmake_pins.py),
  4. clone missing ones into a (name, version) cache (cloner.py),
  5. emit third_party_manifest.json + missing_repos.md (manifest.py).

The `locate-kernel-source` skill (source_locator.py) is intentionally NOT part of
this package yet; it is the second Step 0.5 skill and lands separately.
"""

from __future__ import annotations

from .registry import UNIVERSE, RepoSpec, get_spec, iter_universe

__all__ = ["UNIVERSE", "RepoSpec", "get_spec", "iter_universe"]
