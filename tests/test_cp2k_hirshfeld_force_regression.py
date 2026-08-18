from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = REPO_ROOT / "python_scripts" / "physics_ice" / "nep_mbpol"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from validate_cp2k_hirshfeld_force_regression import (  # noqa: E402
    EXPECTED_INPUT_SHA256,
    EXPECTED_METADATA_SHA256,
    EXPECTED_TOTAL_ATOMIC_FORCE,
    validate,
)


def _output(force: float) -> str:
    return (
        " CP2K| version string: CP2K version 2025.2\n"
        " CP2K| source code revision number: c3a8adfec5\n"
        f" FORCES| Total atomic force {force:.13f}\n"
        " PROGRAM ENDED AT 2026-08-09 00:00:00\n"
    )


def test_exact_v2025_2_force_regression_fixture(tmp_path: Path) -> None:
    source = MODULE_ROOT / "soft_dft" / "benchmarking" / "cp2k_2025_2_hirshfeld_force"
    run = tmp_path / "run"
    run.mkdir()
    (run / "HeH-noconstraint.out").write_text(_output(0.1), encoding="utf-8")
    (run / "HeH-cdft-1.out").write_text(
        _output(EXPECTED_TOTAL_ATOMIC_FORCE), encoding="utf-8"
    )
    (run / "HeH-noconstraint-1_0.wfn").write_bytes(b"paired")
    image = tmp_path / "cp2k.sif"
    image.write_bytes(b"exact image")

    result = validate(source, run, image)

    assert result["status"] == "passed"
    assert result["relative_error"] == pytest.approx(0.0)
    assert set(result["inputs"]) == set(EXPECTED_INPUT_SHA256)
    assert set(result["metadata"]) == set(EXPECTED_METADATA_SHA256)


def test_force_regression_rejects_wrong_value(tmp_path: Path) -> None:
    source = MODULE_ROOT / "soft_dft" / "benchmarking" / "cp2k_2025_2_hirshfeld_force"
    run = tmp_path / "run"
    run.mkdir()
    (run / "HeH-noconstraint.out").write_text(_output(0.1), encoding="utf-8")
    (run / "HeH-cdft-1.out").write_text(_output(0.2), encoding="utf-8")
    (run / "HeH-noconstraint-1_0.wfn").write_bytes(b"paired")
    image = tmp_path / "cp2k.sif"
    image.write_bytes(b"exact image")

    with pytest.raises(RuntimeError, match="force regression failed"):
        validate(source, run, image)
