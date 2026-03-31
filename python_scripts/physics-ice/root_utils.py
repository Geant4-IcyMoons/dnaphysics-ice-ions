"""Helpers for resolving one or many ROOT files (dna.root + splits)."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, List

from constants import PROJECT_ROOT


def _has_wildcards(path_str: str) -> bool:
    return any(ch in path_str for ch in "*?[]")


def _uniq_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def resolve_root_paths(root_arg: str | Path) -> List[str]:
    """Resolve a ROOT file argument into a sorted list of paths.

    - If root_arg contains wildcards, expand them.
    - If root_arg ends with .root, also include split files (dna_*.root).
    - If root_arg is relative, also try PROJECT_ROOT / root_arg.
    """
    raw = str(root_arg)
    expanded = os.path.expanduser(raw)
    matches: List[str] = []

    def add_glob(pattern: str) -> None:
        matches.extend(sorted(glob.glob(pattern)))

    if _has_wildcards(expanded):
        add_glob(expanded)
        if not os.path.isabs(expanded):
            add_glob(str(PROJECT_ROOT / expanded))
        return _uniq_keep_order(matches)

    path = Path(expanded)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(path)
        candidates.append(PROJECT_ROOT / path)

    for cand in candidates:
        # Always try pattern for split ROOT files
        if cand.suffix == ".root":
            pattern = str(cand.with_suffix("")) + "*.root"
            add_glob(pattern)
        elif cand.exists():
            matches.append(str(cand))

    return _uniq_keep_order(matches)


def resolve_first_root(root_arg: str | Path) -> str:
    paths = resolve_root_paths(root_arg)
    if not paths:
        raise FileNotFoundError(root_arg)
    return paths[0]
