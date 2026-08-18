#!/usr/bin/env python3
"""Validate CP2K 2025.2's official density-Hirshfeld force regression."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


UPSTREAM_URL = (
    "https://github.com/cp2k/cp2k/tree/v2025.2/"
    "tests/QS/regtest-cdft-hirshfeld-3"
)
EXPECTED_TOTAL_ATOMIC_FORCE = 0.1552195046628
EXPECTED_RELATIVE_TOLERANCE = 1.0e-7
EXPECTED_INPUT_SHA256 = {
    "HeH-cdft-1.inp": "94f7aaa49ad5524efc321c42e83edb7d534c973f7a99c6b9d9100dcbdffb3046",
    "HeH-noconstraint.inp": "676404f41f8ad5cc5c6a08bd44f3bdce52f3434dedf83abdbf5050c90913ee5c",
    "HeH.xyz": "9f7f12ada81224979c68203769a66fb871d4b916e42dbd53b67f420321f0ebcb",
    "dft-common-params.inc": "40d2c56c2d740dd70cac269b9f728727a5c992d4c7aa6c028384010e0b418fd5",
    "hirshfeld_qs.inc": "82a83b7594064e9b5eada7a410f818a0b11ba1a9f8c9d88644ad7221df7f3973",
    "subsys.inc": "5994538acb4ce8982c025a7fb8582c297423e9adc2b708b490afbf2ab9ebd942",
}
EXPECTED_METADATA_SHA256 = {
    "TEST_FILES.toml": "081e975ade390fda44fe4ab9ea800612a83138d25fb78fe616723fb76af107fe"
}
VERSION_RE = re.compile(r"CP2K\|\s+version string:\s*(.+?)\s*$", re.MULTILINE)
REVISION_RE = re.compile(
    r"CP2K\|\s+source code revision number:\s*(.+?)\s*$", re.MULTILINE
)
FORCE_RE = re.compile(
    r"^\s*FORCES\|\s+Total atomic force\s+(?:=\s*)?"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)\s*$",
    re.MULTILINE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _parse_completed_output(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "PROGRAM ENDED AT" not in text:
        raise RuntimeError(f"CP2K did not terminate normally: {path}")
    versions = VERSION_RE.findall(text)
    revisions = REVISION_RE.findall(text)
    if not versions or "2025.2" not in versions[-1]:
        raise RuntimeError(f"Output is not from CP2K 2025.2: {path}")
    if not revisions:
        raise RuntimeError(f"CP2K source revision is absent: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "cp2k_version": versions[-1].strip(),
        "cp2k_source_revision": revisions[-1].strip(),
    }


def validate(
    input_directory: Path,
    run_directory: Path,
    cp2k_image: Path,
) -> dict[str, object]:
    if not cp2k_image.is_file() or cp2k_image.stat().st_size == 0:
        raise RuntimeError(f"Validated CP2K image is absent: {cp2k_image}")
    inputs: dict[str, dict[str, str]] = {}
    for name, expected in EXPECTED_INPUT_SHA256.items():
        path = input_directory / name
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Official CP2K input checksum mismatch: {path}")
        inputs[name] = {"sha256": actual}
    metadata: dict[str, dict[str, str]] = {}
    for name, expected in EXPECTED_METADATA_SHA256.items():
        path = input_directory / name
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Official CP2K metadata checksum mismatch: {path}")
        metadata[name] = {"sha256": actual}

    unconstrained = _parse_completed_output(run_directory / "HeH-noconstraint.out")
    constrained_path = run_directory / "HeH-cdft-1.out"
    constrained = _parse_completed_output(constrained_path)
    if unconstrained["cp2k_source_revision"] != constrained["cp2k_source_revision"]:
        raise RuntimeError("Reference stages used different CP2K revisions.")
    wavefunction = run_directory / "HeH-noconstraint-1_0.wfn"
    if not wavefunction.is_file() or wavefunction.stat().st_size == 0:
        raise RuntimeError("The official constrained stage lacks its paired WFN.")

    text = constrained_path.read_text(encoding="utf-8", errors="replace")
    values = FORCE_RE.findall(text)
    if not values:
        raise RuntimeError("CP2K M072 total-atomic-force observable is absent.")
    value = float(values[-1].replace("D", "E").replace("d", "e"))
    relative_error = abs(
        (value - EXPECTED_TOTAL_ATOMIC_FORCE) / EXPECTED_TOTAL_ATOMIC_FORCE
    )
    if not math.isfinite(relative_error) or relative_error > EXPECTED_RELATIVE_TOLERANCE:
        raise RuntimeError(
            "CP2K density-Hirshfeld force regression failed: "
            f"relative error {relative_error:.6e} exceeds "
            f"{EXPECTED_RELATIVE_TOLERANCE:.1e}."
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "upstream": {
            "url": UPSTREAM_URL,
            "tag": "v2025.2",
            "matcher": "M072",
            "matcher_definition": "FORCES| Total atomic force, column 5",
            "reference": EXPECTED_TOTAL_ATOMIC_FORCE,
            "relative_tolerance": EXPECTED_RELATIVE_TOLERANCE,
        },
        "inputs": inputs,
        "metadata": metadata,
        "outputs": {
            "unconstrained": unconstrained,
            "constrained": constrained,
            "restart_wavefunction": {
                "path": str(wavefunction.resolve()),
                "sha256": _sha256(wavefunction),
            },
        },
        "executable": {
            "cp2k_image_path": str(cp2k_image.resolve()),
            "cp2k_image_sha256": _sha256(cp2k_image),
            "cp2k_version": constrained["cp2k_version"],
            "cp2k_source_revision": constrained["cp2k_source_revision"],
        },
        "measured_total_atomic_force": value,
        "relative_error": relative_error,
        "scope": (
            "Exact-executable regression of CP2K density-Hirshfeld CDFT forces; "
            "not physical validation of an ion-water surface."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--cp2k-image", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.input_directory.resolve(),
        args.run_directory.resolve(),
        args.cp2k_image.resolve(),
    )
    _atomic_json(args.result_json.resolve(), result)
    print(f"CP2K density-Hirshfeld force regression: {result['status']}")
    print(f"Total atomic force: {result['measured_total_atomic_force']:.13f}")
    print(f"Relative error: {result['relative_error']:.6e}")


if __name__ == "__main__":
    main()
