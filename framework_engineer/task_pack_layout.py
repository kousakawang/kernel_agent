"""Canonical paths for Framework Engineer task packs."""

from __future__ import annotations

from pathlib import Path


TASK_DIR_NAME = "task"


def task_dir(task_pack: Path | str) -> Path:
    """Return the payload directory for an outer task-pack root."""

    return Path(task_pack) / TASK_DIR_NAME


def task_path(task_pack: Path | str, *parts: str) -> Path:
    """Return a path below the canonical payload directory."""

    return task_dir(task_pack).joinpath(*parts)
