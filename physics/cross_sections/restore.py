"""Restore the catalogued cross sections, checking each original SHA-256."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tarfile

from tqdm import tqdm


CATALOGUE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def hydrogen(destination: Path | None):
    with (CATALOGUE / "MANIFEST.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in tqdm(rows, desc="Restoring hydrogen tables", unit="file"):
        relative = Path(row["catalogue_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid catalogue path: {relative}")
        source = CATALOGUE / relative
        if digest(source) != row["catalogue_sha256"]:
            raise ValueError(f"Compressed catalogue file changed: {source}")
        if destination is None:
            value = hashlib.sha256()
            size = 0
            with (gzip.open(source, "rb") if source.suffix == ".gz" else source.open("rb")) as packed:
                while chunk := packed.read(1024 * 1024):
                    value.update(chunk)
                    size += len(chunk)
            if size != int(row["original_bytes"]) or value.hexdigest() != row["original_sha256"]:
                raise ValueError(f"Catalogue does not reproduce original file: {relative}")
            continue
        output = destination / relative
        if output.suffix == ".gz":
            output = output.with_suffix("")
        if output.exists() and digest(output) == row["original_sha256"]:
            continue
        if output.exists():
            raise FileExistsError(f"Existing file differs from catalogue: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".partial")
        if temporary.exists():
            raise FileExistsError(f"Inspect interrupted restore before retrying: {temporary}")
        value = hashlib.sha256()
        try:
            with (gzip.open(source, "rb") if source.suffix == ".gz" else source.open("rb")) as packed:
                with temporary.open("wb") as restored:
                    while chunk := packed.read(1024 * 1024):
                        value.update(chunk)
                        restored.write(chunk)
            if (temporary.stat().st_size != int(row["original_bytes"])
                    or value.hexdigest() != row["original_sha256"]):
                raise ValueError(f"Restored file differs from original: {relative}")
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)


def elastic(destination: Path | None):
    folder = CATALOGUE / "elastic/carbon_nlh_pair"
    index = json.loads((folder / "index.json").read_text())
    products = {item["path"]: item["sha256"] for item in index["products"]}
    expected = set(products) | {"index.json"}
    index_sha = digest(folder / "index.json")
    with tarfile.open(folder / "pair_dcs.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
        if {item.name for item in members} != expected or any(not item.isfile() for item in members):
            raise ValueError("Elastic archive does not match its index")
        for item in tqdm(members, desc="Restoring carbon pair DCS", unit="map"):
            if Path(item.name).is_absolute() or ".." in Path(item.name).parts:
                raise ValueError(f"Invalid elastic path: {item.name}")
            expected_sha = index_sha if item.name == "index.json" else products[item.name]
            if destination is None:
                value = hashlib.sha256()
                with archive.extractfile(item) as original:
                    while chunk := original.read(1024 * 1024):
                        value.update(chunk)
                if value.hexdigest() != expected_sha:
                    raise ValueError(f"Elastic map differs from index: {item.name}")
                continue
            output = destination / "elastic/carbon_nlh_pair" / item.name
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() and digest(output) == expected_sha:
                continue
            if output.exists():
                raise FileExistsError(f"Existing file differs from archive: {output}")
            temporary = output.with_name(output.name + ".partial")
            if temporary.exists():
                raise FileExistsError(f"Inspect interrupted restore before retrying: {temporary}")
            value = hashlib.sha256()
            try:
                with archive.extractfile(item) as original, temporary.open("wb") as restored:
                    while chunk := original.read(1024 * 1024):
                        value.update(chunk)
                        restored.write(chunk)
                if value.hexdigest() != expected_sha:
                    raise ValueError(f"Elastic map differs from index: {item.name}")
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument("--family", choices=("hydrogen", "elastic", "all"), default="all")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only and args.destination is not None:
        parser.error("Do not provide a destination with --verify-only")
    if not args.verify_only and args.destination is None:
        parser.error("Provide a destination, or use --verify-only")
    target = args.destination.resolve() if args.destination else None
    if args.family in ("hydrogen", "all"):
        hydrogen(target)
    if args.family in ("elastic", "all"):
        elastic(target)
    print("Verified cross-section catalogue" if target is None
          else f"Verified cross sections restored under {target}", flush=True)


if __name__ == "__main__":
    main()
