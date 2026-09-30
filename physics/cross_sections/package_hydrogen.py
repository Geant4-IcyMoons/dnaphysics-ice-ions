"""Create a verified, ready-to-unzip hydrogen DAT archive from the Git catalogue."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
from pathlib import Path
import zipfile

from tqdm import tqdm


CATALOGUE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = CATALOGUE.parents[2] / "Downloads/hydrogen_cross_sections_20260930.zip"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def archive_name(relative: Path) -> str:
    parts = relative.parts
    if len(parts) < 6 or parts[0] != "inelastic_dielectric":
        raise ValueError(f"Unexpected catalogue path: {relative}")
    _, kernel, charge, phase, setting, *rest = parts
    if kernel not in ("pwba", "rpwba") or charge not in ("H0", "H1"):
        raise ValueError(f"Unexpected catalogue path: {relative}")
    name = Path(*rest)
    if name.suffix == ".gz":
        name = name.with_suffix("")
    return str(Path(charge) / phase / kernel.upper() / setting / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not output.is_file():
        raise ValueError(f"Output is not a regular file: {output}")
    with (CATALOGUE / "MANIFEST.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 80:
        raise ValueError(f"Expected 80 hydrogen catalogue files, got {len(rows)}")
    if sum(row["catalogue_path"].endswith(".dat.gz") for row in rows) != 64:
        raise ValueError("Expected 64 hydrogen DAT files")
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Inspect interrupted archive before retrying: {temporary}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for row in tqdm(rows, desc="Packing hydrogen cross sections", unit="file"):
                relative = Path(row["catalogue_path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe catalogue path: {relative}")
                source = CATALOGUE / relative
                if digest(source) != row["catalogue_sha256"]:
                    raise ValueError(f"Catalogue checksum mismatch: {relative}")
                value = hashlib.sha256()
                size = 0
                with (gzip.open(source, "rb") if source.suffix == ".gz"
                      else source.open("rb")) as unpacked:
                    with archive.open(archive_name(relative), "w", force_zip64=True) as packed:
                        while chunk := unpacked.read(1024 * 1024):
                            value.update(chunk)
                            size += len(chunk)
                            packed.write(chunk)
                if size != int(row["original_bytes"]) or value.hexdigest() != row["original_sha256"]:
                    raise ValueError(f"Original checksum mismatch: {relative}")
            archive.write(CATALOGUE / "MANIFEST.csv", "MANIFEST.csv")
            archive.write(CATALOGUE / "README.md", "README.md")
            for source in sorted((CATALOGUE / "provenance").glob("*_submission.json")):
                archive.write(source, f"provenance/{source.name}")
        with zipfile.ZipFile(temporary) as archive:
            if len(archive.namelist()) != 86 or archive.testzip() is not None:
                raise ValueError("ZIP entry count or CRC verification failed")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Verified 86-entry hydrogen archive: {output}", flush=True)


if __name__ == "__main__":
    main()
