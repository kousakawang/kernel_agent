"""Scan generated artifacts for unfilled ``<FILL: ...>`` placeholders.

Used two ways:
  * reporting  — after each dry-run step, tell the user exactly which absolute
    file + line numbers still need manual input.
  * gating     — before a step consumes the previous step's output, refuse to
    continue if required fields are still placeholders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FILL_RE = re.compile(r"<FILL:([^>]*)>")
# Extract the JSON key on a line like:  "archetype": "<FILL: ...>",
_KEY_RE = re.compile(r'"([A-Za-z0-9_]+)"\s*:')


@dataclass
class FillPoint:
    path: str          # absolute path
    lineno: int        # 1-based
    key: str           # nearest JSON key on the line ("" if none)
    hint: str          # text inside <FILL: ...>

    def as_line(self) -> str:
        loc = f"{self.path}:{self.lineno}"
        key = f"  {self.key}" if self.key else ""
        return f"{loc}{key}  <FILL: {self.hint.strip()}>"


def scan_file(path: Path) -> list[FillPoint]:
    path = Path(path).resolve()
    out: list[FillPoint] = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        for m in _FILL_RE.finditer(line):
            key_m = _KEY_RE.search(line)
            out.append(
                FillPoint(
                    path=str(path),
                    lineno=i,
                    key=key_m.group(1) if key_m else "",
                    hint=m.group(1),
                )
            )
    return out


def scan_files(paths: list[Path]) -> list[FillPoint]:
    out: list[FillPoint] = []
    for p in paths:
        out.extend(scan_file(p))
    return out


def has_unfilled_keys(path: Path, keys: set[str]) -> list[FillPoint]:
    """Return the subset of fill points whose JSON key is in ``keys``.

    Used as a gate: e.g. locate step requires ``interface``/``archetype`` filled.
    """
    return [fp for fp in scan_file(path) if fp.key in keys]
