from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest


NEP_MBPOL = (
    Path(__file__).resolve().parents[1]
    / "python_scripts"
    / "physics_ice"
    / "nep_mbpol"
)
if str(NEP_MBPOL) not in sys.path:
    sys.path.insert(0, str(NEP_MBPOL))

from bca.runtime import AdaptiveKernelTable  # noqa: E402
from export_nlh_geant4_table import export_table  # noqa: E402


SOURCE = NEP_MBPOL / "collision_kernels" / "nlh_collision_kernels.manifest.json"


def _read_export(path: Path):
    metadata: dict[str, str] = {}
    rows: dict[tuple[str, float], tuple[list[float], list[float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            fields = line[2:].split(maxsplit=1)
            if len(fields) == 2:
                metadata[fields[0]] = fields[1]
            continue
        target, energy, quantile, theta = line.split()
        values = rows.setdefault((target, float(energy)), ([], []))
        values[0].append(float(quantile))
        values[1].append(float(theta))
    return metadata, rows


def _linear(x: float, grid: list[float], values: list[float]) -> float:
    for upper in range(1, len(grid)):
        if x <= grid[upper]:
            fraction = (x - grid[upper - 1]) / (grid[upper] - grid[upper - 1])
            return values[upper - 1] + fraction * (values[upper] - values[upper - 1])
    return values[-1]


def test_exported_carbon_table_matches_verified_runtime(tmp_path):
    data_path, manifest_path = export_table(SOURCE, tmp_path, "C")
    source = AdaptiveKernelTable(SOURCE)
    metadata, rows = _read_export(data_path)

    assert metadata["projectile"] == "C"
    assert metadata["release_status"] == "atomistic_validation_pending"
    assert "evaluate exactly" in metadata["hard_cross_section"]
    assert metadata["pure_water_rate"] == (
        "Sigma_P_hard=n_H2O*(2*sigma_PH_hard+sigma_PO_hard)"
    )
    assert set(target for target, _ in rows) == {"H", "O"}

    for target in ("H", "O"):
        energies = sorted(energy for row_target, energy in rows if row_target == target)
        for energy in (energies[0], energies[len(energies) // 2], energies[-1]):
            quantiles, angles = rows[(target, energy)]
            for quantile in (0.0, 0.137, 0.5, 0.913, 1.0):
                actual = _linear(quantile, quantiles, angles)
                expected = source.theta_cm_rad("C", target, energy, quantile)
                assert actual == pytest.approx(expected, rel=2.0e-15, abs=1.0e-15)

        for energy in (1.1e3, 3.7e4, 2.3e6, 8.9e7):
            radius = source.turning_threshold_radius_angstrom("C", target)
            kinematics = source.pair_kinematics("C", target, energy)
            expected = (
                math.pi
                * radius**2
                * (
                    1.0
                    - source.minimum_turning_potential_ev
                    / kinematics.relative_kinetic_energy_ev
                )
                if kinematics.relative_kinetic_energy_ev
                > source.minimum_turning_potential_ev
                else 0.0
            )
            assert source.hard_cross_section_angstrom2(
                "C", target, energy
            ) == pytest.approx(expected, rel=2.0e-15)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (
        manifest["interpolation_contract"][
            "nominal_combined_relative_tolerance"
        ]
        == 0.005
    )
    assert "not a statement of physical accuracy" in manifest[
        "interpolation_contract"
    ]["interpretation"]
    assert manifest["physical_model_uncertainty"][
        "nlh_pair_potential_rms_error_percent"
    ] == {
        "H": {
            "rms_error_above_30_ev_percent": 3.56,
            "rms_error_above_10_ev_percent": 10.02,
        },
        "O": {
            "rms_error_above_30_ev_percent": 9.51,
            "rms_error_above_10_ev_percent": 25.06,
        },
    }


def test_export_rejects_unreleased_projectile(tmp_path):
    with pytest.raises(ValueError, match="Only carbon"):
        export_table(SOURCE, tmp_path, "O")
