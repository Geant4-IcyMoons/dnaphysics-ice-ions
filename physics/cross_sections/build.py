"""Copy verified hydrogen and elastic cross sections into the Git catalogue.

Run from the repository root after the referenced campaigns have completed.
The original output files are read only. Large tables are compressed separately
so regular Git can carry them without Git LFS.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "physics/inelastic_dielectric/output"
CATALOGUE = Path(__file__).resolve().parent
MAX_GIT_FILE = 100_000_000  # Stay below GitHub's 100 MB object limit.
CAMPAIGNS = (
    (1, False, OUTPUT / "runs/proton_20260907_yeqgpyaf/submission.json"),
    (0, False, OUTPUT / "runs/proton_q0_20260908_rdbe8gbb/submission.json"),
    (0, True, OUTPUT / "runs/proton_q0_20260914_gy7ofudd/submission.json"),
    (1, True, OUTPUT / "continuation_rtol05_20260929/proton_q1_20260914_s9w6fly3.json"),
)
ELASTIC = ROOT / "physics/elastic/hard_collisions/runs/carbon_ice/differential"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def compressed_copy(source: Path, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    value = hashlib.sha256()
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Inspect unfinished copy before retrying: {temporary}")
    try:
        with source.open("rb") as original, temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped:
                while chunk := original.read(1024 * 1024):
                    value.update(chunk)
                    zipped.write(chunk)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed while copying: {source}")
        if temporary.stat().st_size >= MAX_GIT_FILE:
            raise RuntimeError(f"Compressed file still exceeds Git object limit: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return before.st_size, value.hexdigest()


def dielectric_inventory():
    cases = set()
    for charge, polarization, manifest in CAMPAIGNS:
        campaign = json.loads(manifest.read_text())
        assert campaign.get("charge_state", charge) == charge
        for job in campaign["jobs"]:
            case = job["case"]
            if ("nobarkas" not in case) != polarization:
                continue
            phase = "amorphous" if "_amorphous_" in case else "hexagonal"
            kernel = "rpwba" if "_rpwba_" in case else "pwba"
            folder = (Path("inelastic_dielectric") / kernel / f"H{charge}" / phase
                      / ("polarization_on" if polarization else "polarization_off"))
            if folder in cases:
                raise ValueError(f"Duplicate case: {folder}")
            cases.add(folder)
            completion = Path(job["cache"]) / "completion.json"
            extra = []
            if polarization:
                report = json.loads(completion.read_text())
                if (report["status"] != "numerically_complete"
                        or report["source_sha256"] != campaign["source_sha256"]):
                    raise ValueError(f"Case not complete: {case}")
                products = []
                for item in report["products"]:
                    path = Path(item["path"])
                    stat = path.stat()
                    if (stat.st_size, stat.st_mtime_ns) != (item["size"], item["mtime_ns"]):
                        raise ValueError(f"Completion record differs from file: {path}")
                    products.append(path)
                extra.append((completion, folder / "completion.json", False))
            else:
                products = [p for p in Path(campaign["tables"]).glob("*.dat")
                            if (f"_{phase}_" in p.name and "barkas" not in p.name
                                and ("rpwba" in p.name) == (kernel == "rpwba"))]
            if len(products) != 4:
                raise ValueError(f"Expected four DAT products for {case}, got {len(products)}")
            for path in products:
                with path.open() as stream:
                    header = stream.readline()
                if not header.startswith("# ion_table_metadata: "):
                    raise ValueError(f"Missing table metadata: {path}")
                metadata = json.loads(header.split(": ", 1)[1])
                if (metadata["projectile_element"] != "H"
                        or metadata["projectile_charge_state"] != charge
                        or bool(metadata["include_barkas_dcs"]) != polarization
                        or metadata["projectile_kernel"] !=
                        ("pwba" if kernel == "pwba" else "dominguez-munoz-2022-finite-Q")):
                    raise ValueError(f"Table does not match its catalogue slot: {path}")
                yield path, folder / (path.name + ".gz"), True
            if charge == 0 and polarization:
                for name in ("DIAGNOSTIC_ONLY.json", "DIAGNOSTIC_ONLY.csv"):
                    path = products[0].parent / name
                    if not path.is_file():
                        raise ValueError(f"Missing neutral diagnostic flag data: {path}")
                    extra.append((path, folder / (name + (".gz" if name.endswith(".csv") else "")),
                                  name.endswith(".csv")))
            yield from extra
        provenance = CATALOGUE / "provenance" / f"H{charge}_polarization_{'on' if polarization else 'off'}_submission.json"
        provenance.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest, provenance)
    if len(cases) != 16:
        raise ValueError(f"Expected 16 hydrogen cases, got {len(cases)}")


def elastic_copy():
    index = json.loads((ELASTIC / "index.json").read_text())
    if index["status"] != "numerically_qualified_binary_differential_cross_sections":
        raise ValueError("Carbon elastic pair DCS is not qualified")
    products = index["products"]
    names = {item["path"] for item in products}
    if len(names) != len(products) or len(names) != 962:
        raise ValueError("Incomplete or duplicated elastic DCS index")
    for item in tqdm(products, desc="Verifying carbon pair DCS", unit="map"):
        path = ELASTIC / item["path"]
        if path.parent != ELASTIC or digest(path) != item["sha256"]:
            raise ValueError(f"Elastic map mismatch: {path}")
    folder = CATALOGUE / "elastic/carbon_nlh_pair"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ELASTIC / "index.json", folder / "index.json")
    archive = folder / "pair_dcs.tar.gz"
    temporary = archive.with_name(archive.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Inspect unfinished copy before retrying: {temporary}")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w|") as tar:
                    for name in sorted(names | {"index.json"}):
                        path = ELASTIC / name
                        info = tar.gettarinfo(str(path), arcname=name)
                        info.mtime = 0
                        with path.open("rb") as stream:
                            tar.addfile(info, stream)
        if temporary.stat().st_size >= MAX_GIT_FILE:
            raise RuntimeError("Elastic archive exceeds Git object limit")
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def source_snapshots():
    """Preserve the exact small code snapshots named by campaign submissions."""
    folder = CATALOGUE / "provenance"
    folder.mkdir(parents=True, exist_ok=True)
    for charge, polarization, manifest in CAMPAIGNS:
        campaign = json.loads(manifest.read_text())
        source = Path(campaign["source_snapshot"])
        if not source.is_dir():
            raise FileNotFoundError(source)
        archive = folder / f"H{charge}_polarization_{'on' if polarization else 'off'}_source.tar.gz"
        temporary = archive.with_name(archive.name + ".partial")
        if temporary.exists():
            raise FileExistsError(temporary)
        try:
            with temporary.open("wb") as raw:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped:
                    with tarfile.open(fileobj=zipped, mode="w|") as tar:
                        for path in sorted(p for p in source.rglob("*") if p.is_file()):
                            info = tar.gettarinfo(str(path), arcname=str(path.relative_to(source)))
                            info.mtime = 0
                            with path.open("rb") as stream:
                                tar.addfile(info, stream)
            if temporary.stat().st_size >= MAX_GIT_FILE:
                raise RuntimeError(f"Source snapshot archive exceeds Git object limit: {source}")
            os.replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    items = list(dielectric_inventory())
    if sum(dest.suffix == ".gz" and source.suffix == ".dat" for source, dest, _ in items) != 64:
        raise ValueError("Expected 64 hydrogen DAT files")
    rows = []
    for source, relative, compressed in tqdm(items, desc="Copying hydrogen tables", unit="file"):
        target = CATALOGUE / relative
        if compressed:
            size, sha = compressed_copy(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            size, sha = source.stat().st_size, digest(source)
        rows.append((str(relative), str(source), size, sha, digest(target)))
    with (CATALOGUE / "MANIFEST.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("catalogue_path", "original_path", "original_bytes", "original_sha256", "catalogue_sha256"))
        writer.writerows(rows)
    elastic_copy()
    source_snapshots()
    print(f"Catalogued {len(rows)} hydrogen files and 962 carbon elastic maps", flush=True)


if __name__ == "__main__":
    main()
