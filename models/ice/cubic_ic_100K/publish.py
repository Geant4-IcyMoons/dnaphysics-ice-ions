"""Publish the authorized, validated Ic material with an isolated Git index."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REQUEST = HERE / "runs/publish_request.json"
RECEIPT = HERE / "runs/publish_result.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str, input: bytes | None = None, env=None) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, input=input, env=env).decode().strip()


def tree_entry(commit: str, path: str) -> str:
    return git("ls-tree", commit, "--", path)


def verify(request: dict) -> dict[str, bytes]:
    """Return only reviewed sources and checksummed accepted outputs."""
    for relative, expected in request["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Reviewed source changed: {relative}")
    manifest = json.loads((HERE / "manifest.json").read_text())
    if manifest["status"] != "structurally_validated_density_proxy" or len(manifest["structures"]) != 3:
        raise ValueError("The cubic model is not fully validated")
    config = json.loads((HERE / "config.json").read_text())
    if manifest["config"] != config:
        raise ValueError("Model settings differ from reviewed settings")
    for record in manifest["structures"] + manifest["reports"] + manifest["preparations"]:
        if digest(HERE / record["path"]) != record["sha256"]:
            raise ValueError(f"Changed material product: {record['path']}")
    for record in manifest["reports"]:
        report = json.loads((HERE / record["path"]).read_text())
        if not report["passed"] or not all(report["gates"].values()):
            raise ValueError("A replica failed its scientific checks")
        if report["validator_sha256"] != digest(HERE / "validate.py"):
            raise ValueError("Unreviewed validator produced a report")
    figure = HERE.parent / "ice_structure_comparison.png"
    sidecar = json.loads(figure.with_suffix(".json").read_text())
    if [row["label"] for row in sidecar["rows"]] != [
        "Hexagonal ice Ih, 100 K", "Cubic ice Ic, 100 K", "Amorphous LDA, 80 K"]:
        raise ValueError("The requested three-row plot was not generated")
    for row in sidecar["rows"]:
        if digest(Path(row["path"])) != row["sha256"]:
            raise ValueError("Figure inputs changed")
    if digest(figure) != sidecar["png_sha256"]:
        raise ValueError("Comparison image changed after rendering")
    products = {relative: (ROOT / relative).read_bytes() for relative in request["paths"]}
    for relative, expected in request["source_sha256"].items():
        if hashlib.sha256(products[relative]).hexdigest() != expected:
            raise ValueError(f"Source changed while taking the publication snapshot: {relative}")
    for record in manifest["structures"] + manifest["reports"] + manifest["preparations"]:
        relative = str((HERE / record["path"]).relative_to(ROOT))
        if hashlib.sha256(products[relative]).hexdigest() != record["sha256"]:
            raise ValueError(f"Product changed while taking the publication snapshot: {relative}")
    if hashlib.sha256(products[str(figure.relative_to(ROOT))]).hexdigest() != sidecar["png_sha256"]:
        raise ValueError("Figure changed while taking the publication snapshot")
    return products


def publish() -> dict:
    request = json.loads(REQUEST.read_text())
    if git("remote", "get-url", "--push", "origin") != request["remote_url"]:
        raise ValueError("Push destination changed")
    products = verify(request)
    parent = git("ls-remote", "origin", "refs/heads/ion_modular").split()[0]
    git("fetch", "--no-tags", "--no-write-fetch-head", "origin", parent)
    for relative, expected in request["base_entries"].items():
        if tree_entry(parent, relative) != expected:
            raise ValueError(f"Remote task file changed; refusing to overwrite: {relative}")
    with tempfile.TemporaryDirectory(prefix="ic-publish-") as directory:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(directory) / "index"))
        git("read-tree", parent, env=env)
        for relative, content in products.items():
            blob = git("hash-object", "-w", "--stdin", input=content)
            git("update-index", "--add", "--cacheinfo", "100644", blob, relative, env=env)
        tree = git("write-tree", env=env)
        message = ("Add validated 100 K cubic ice and three-phase structure figure\n\n"
                   "Use the documented H2O-Ih molecular-volume approximation for Ic. "
                   "Retain three independent NEP-MB-pol configurations, numerical "
                   "structural checks, and restartable preparation. Update the "
                   "Ih/Ic/LDA figure. Cubic collision integration remains disabled.\n")
        commit = git("commit-tree", tree, "-p", parent, input=message.encode())
        # No force push. A concurrent remote update causes a safe rejection.
        git("push", "origin", f"{commit}:refs/heads/ion_modular")
    return {"status": "pushed", "commit": commit, "parent": parent,
            "branch": "ion_modular", "files": list(products),
            "local_workspace": "Shared HEAD, index, and unrelated files were not changed."}


if __name__ == "__main__":
    try:
        result = publish()
    except Exception as error:
        RECEIPT.write_text(json.dumps({"status": "failed", "error": str(error)}, indent=2) + "\n")
        raise
    RECEIPT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
