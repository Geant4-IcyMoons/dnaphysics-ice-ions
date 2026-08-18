from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


NEP_DIR = (
    Path(__file__).resolve().parents[1]
    / "python_scripts"
    / "physics_ice"
    / "nep_mbpol"
)
if str(NEP_DIR) not in sys.path:
    sys.path.insert(0, str(NEP_DIR))

from soft_dft.gpaw_prepared_q1 import (  # noqa: E402
    Q1_COMPONENTS,
    _complete_metric_basis,
    build_multiplier_calibration_manifest,
    build_manifest,
    load_manifest,
)


def test_manifest_has_four_q1_components(tmp_path):
    path = build_manifest(tmp_path / "pilot")
    manifest = load_manifest(path)
    assert manifest["task_count"] == 4
    assert tuple(task["component"] for task in manifest["tasks"]) == Q1_COMPONENTS
    assert all(task["charge"] == 1 for task in manifest["tasks"])
    assert all(task["production_eligible"] is False for task in manifest["tasks"])


def test_manifest_fails_closed_after_tampering(tmp_path):
    path = build_manifest(tmp_path / "pilot")
    payload = json.loads(path.read_text())
    payload["tasks"][0]["charge"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="signature mismatch"):
        load_manifest(path)


def test_metric_completion_preserves_orthonormality():
    overlap = np.asarray([[1.0, 0.1, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 1.0]])
    coefficients = _complete_metric_basis([np.asarray([1.0, 0.0, 0.0])], overlap)
    assert np.allclose(coefficients @ overlap @ coefficients.T, np.eye(3), atol=1e-9)


def test_multiplier_calibration_manifest_has_requested_matrix(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    for name in ("prepared_complex.gpw", "carbon_fragment.gpw", "water_fragment.gpw"):
        (reference / name).write_bytes(name.encode())
    path = build_multiplier_calibration_manifest(tmp_path / "calibration", reference)
    manifest = load_manifest(path)
    assert manifest["task_count"] == 5
    assert [task["constraint_mode"] for task in manifest["tasks"]] == [
        "charge_only", "charge_only", "charge_spin", "charge_spin", "charge_spin"
    ]
    assert all(task["settings"]["constraint_tolerance_electrons"] == 1e-5
               for task in manifest["tasks"])
