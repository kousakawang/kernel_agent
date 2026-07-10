"""Parse backend-selector flags from sglang launch commands.

IMPORTANT: the result is used ONLY to annotate ``triggered_by`` on each repo.
It NEVER prunes the clone set — default-path libraries are used even when no flag
names them, so pruning by flags would drop real dependencies (see plan doc).

We scan each ``service_cmds`` entry for known backend selector flags and collect
their values, then attribute each registry entry to the backends whose selector
value matches one of the entry's ``backend_flags``. Every entry also carries the
synthetic ``default_path`` tag when ``on_default_path`` is set.
"""

from __future__ import annotations

import re
import shlex

from .registry import RepoSpec, iter_universe

# Backend selector flags whose *values* name a kernel backend.
_BACKEND_FLAGS = (
    "--attention-backend",
    "--linear-attn-backend",
    "--mm-attention-backend",
    "--moe-runner-backend",
    "--moe-a2a-backend",
    "--grammar-backend",
    "--sampling-backend",
    "--decode-attention-backend",
    "--prefill-attention-backend",
)


def _parse_flag_values(cmd: str) -> set[str]:
    """Extract backend-selector values from one command string.

    Handles both ``--flag value`` and ``--flag=value`` forms.
    """

    values: set[str] = set()
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()
    for i, tok in enumerate(tokens):
        key, sep, inline = tok.partition("=")
        if key in _BACKEND_FLAGS:
            if sep and inline:
                values.add(inline.strip().lower())
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                values.add(tokens[i + 1].strip().lower())
    return values


def backends_from_service_cmds(service_cmds: list[dict[str, str]]) -> dict[str, set[str]]:
    """Return ``{backend_name: {selector_value, ...}}`` per launch command."""

    out: dict[str, set[str]] = {}
    for entry in service_cmds:
        name = entry.get("backend_name") or "cmd"
        out[name] = _parse_flag_values(entry.get("cmd", ""))
    return out


def triggered_by_for(spec: RepoSpec, backends: dict[str, set[str]]) -> list[str]:
    """Compute the ``triggered_by`` annotation list for one repo.

    A repo is attributed to a backend command when any of its ``backend_flags``
    appears among that command's selector values. Default-path repos always get
    the synthetic ``default_path`` tag so they are never mistaken for unused.
    """

    tags: list[str] = []
    flag_set = {f.lower() for f in spec.backend_flags}
    for backend_name, values in backends.items():
        if flag_set & values:
            tags.append(backend_name)
    if spec.on_default_path:
        tags.append("default_path")
    return sorted(dict.fromkeys(tags))


def annotate_universe(service_cmds: list[dict[str, str]]) -> dict[str, list[str]]:
    """Return ``{repo_name: triggered_by}`` for the whole source-bearing universe.

    Note: this covers ALL source-bearing repos, not a pruned subset.
    """

    backends = backends_from_service_cmds(service_cmds)
    return {
        spec.name: triggered_by_for(spec, backends)
        for spec in iter_universe(source_bearing_only=True)
    }
