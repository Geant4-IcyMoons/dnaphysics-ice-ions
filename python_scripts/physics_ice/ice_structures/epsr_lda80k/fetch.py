#!/usr/bin/env python3
"""Fetch and checksum the primary 80 K LDA EPSR archive from STFC."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from urllib.request import Request, urlopen
import zipfile

from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
PHYSICS_ICE = HERE.parents[1]
sys.path.insert(0, str(PHYSICS_ICE))

from ice_structures.epsr_lda80k.model import file_sha256  # noqa: E402


DEFAULT_RUN_ROOT = (
    PHYSICS_ICE
    / "process_evidence"
    / "ice_structures"
    / "validation"
    / "runs"
    / "epsr_lda80k"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_ROOT / "source")
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an already-downloaded AmorIce.zip instead of accessing STFC.",
    )
    return parser.parse_args()


def _manifest() -> dict[str, object]:
    value = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("source_manifest.json must contain one object.")
    return value


def _download(url: str, output: Path, expected_bytes: int) -> None:
    request = Request(url, headers={"User-Agent": "dnaphysics-ice-ions/1"})
    with urlopen(request, timeout=120) as response, output.open("wb") as handle:
        size = int(response.headers.get("Content-Length", expected_bytes))
        with tqdm(total=size, unit="B", unit_scale=True, desc="AmorIce.zip") as progress:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                progress.update(len(block))


def fetch(output_dir: Path, supplied_archive: Path | None = None) -> dict[str, object]:
    manifest = _manifest()
    archive_record = manifest["archive"]
    dataset_record = manifest["dataset"]
    required = manifest["required_files"]
    if not isinstance(archive_record, dict) or not isinstance(dataset_record, dict):
        raise ValueError("Malformed source manifest.")
    if not isinstance(required, dict):
        raise ValueError("Malformed required-files manifest.")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / str(archive_record["filename"])
    expected_archive_hash = str(archive_record["sha256"])
    if supplied_archive is not None:
        source = supplied_archive.expanduser().resolve()
        if file_sha256(source) != expected_archive_hash:
            raise ValueError(f"Archive checksum mismatch: {source}")
        if not archive_path.is_file() or file_sha256(archive_path) != expected_archive_hash:
            temporary = archive_path.with_suffix(".tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, archive_path)
    elif not archive_path.is_file() or file_sha256(archive_path) != expected_archive_hash:
        with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            _download(
                str(archive_record["url"]),
                temporary,
                int(archive_record["bytes"]),
            )
            if file_sha256(temporary) != expected_archive_hash:
                raise ValueError("Downloaded AmorIce.zip checksum does not match manifest.")
            os.replace(temporary, archive_path)
        finally:
            temporary.unlink(missing_ok=True)

    prefix = PurePosixPath(str(dataset_record["archive_member_prefix"]))
    extracted = output_dir / "LDAneutron"
    extracted.mkdir(exist_ok=True)
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        for filename, expected_hash_value in tqdm(
            sorted(required.items()), desc="verified EPSR files"
        ):
            member = str(prefix / str(filename))
            if member not in names:
                raise FileNotFoundError(f"{member} is absent from {archive_path}.")
            payload = archive.read(member)
            output = extracted / str(filename)
            with tempfile.NamedTemporaryFile(dir=extracted, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
            os.replace(temporary, output)
            output.chmod(0o644)
            measured = file_sha256(output)
            expected_hash = str(expected_hash_value)
            if measured != expected_hash:
                raise ValueError(f"Checksum mismatch for {filename}.")
            hashes[str(filename)] = measured

    receipt: dict[str, object] = {
        "schema_version": 1,
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "archive": {
            "path": str(archive_path),
            "sha256": file_sha256(archive_path),
            "bytes": archive_path.stat().st_size,
        },
        "dataset": dataset_record,
        "files": hashes,
        "status": "verified",
    }
    receipt_path = output_dir / "source_receipt.json"
    temporary_receipt = receipt_path.with_suffix(".tmp")
    temporary_receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_receipt, receipt_path)
    archive_path.chmod(0o644)
    receipt_path.chmod(0o644)
    return receipt


def main() -> None:
    args = parse_args()
    receipt = fetch(args.output_dir, args.archive)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
