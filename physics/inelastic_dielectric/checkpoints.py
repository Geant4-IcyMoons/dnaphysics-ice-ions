"""Atomic task products and strict provenance for interrupted generation."""
from pathlib import Path
from contextlib import contextmanager
import hashlib
import json
import os

import numpy as np


@contextmanager
def atomic_text(path):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_savez(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_digest():
    """Include executable physics and bundled input data, not generated products."""
    root = Path(__file__).resolve().parents[1]
    paths = sorted(p for p in root.rglob("*") if p.is_file()
                   and p.suffix in {".py", ".json", ".xlsx"}
                   and not {"output", "plots", "runs", "__pycache__"}.intersection(p.relative_to(root).parts))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def require_manifest(directory, settings):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    signature = json.dumps(settings, sort_keys=True)
    path = directory / "manifest.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(data["signature"]) != signature:
                raise ValueError(f"Incompatible checkpoint provenance: {directory}")
    else:
        atomic_savez(path, signature=signature)
    return signature


def load_task(directory, task, signature, T, W):
    if directory is None:
        return None
    path = Path(directory) / ("_".join(map(str, task)) + ".npz")
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        values = data["values"]
        if (str(data["signature"]) != signature or not np.array_equal(data["T"], T)
                or not np.array_equal(data["W"], W)):
            raise ValueError(f"Incompatible task checkpoint: {path}")
        if values.shape != W.shape or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"Invalid task checkpoint: {path}")
        return values.copy()


def save_task(directory, task, signature, T, W, values):
    if directory is not None:
        atomic_savez(Path(directory) / ("_".join(map(str, task)) + ".npz"),
                     signature=signature, T=T, W=W, values=values)


def save_energy(directory, index, T, sigma, signature):
    payload = json.dumps(sigma, default=lambda value: np.asarray(value).tolist())
    atomic_savez(Path(directory) / f"energy_{index:06d}.npz",
                 T=T, sigma=payload, signature=signature)


def load_energy(directory, index, T, signature):
    path = Path(directory) / f"energy_{index:06d}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["signature"]) != signature or float(data["T"]) != float(T):
            raise ValueError(f"Incompatible energy checkpoint: {path}")
        return json.loads(str(data["sigma"]))
