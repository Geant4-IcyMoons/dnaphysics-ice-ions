from __future__ import annotations

from pathlib import Path
import csv
import json
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

from bca.tables import KernelTableConfig, generate_kernel_tables  # noqa: E402


def test_parallel_table_output_is_deterministic_and_resumable(tmp_path):
    common = dict(
        projectiles=("C",),
        energy_min_ev=1.0e4,
        energy_max_ev=1.0e5,
        base_energy_points=2,
        axis_relative_tolerance=0.05,
        max_energy_points=32,
        max_impact_points=256,
        quadrature_order=32,
    )
    serial = KernelTableConfig(**common, workers=1)
    parallel = KernelTableConfig(**common, workers=2)
    serial_csv, _ = generate_kernel_tables(
        serial, tmp_path / "serial", show_progress=False
    )
    parallel_csv, _ = generate_kernel_tables(
        parallel, tmp_path / "parallel", show_progress=False
    )
    assert serial_csv.read_bytes() == parallel_csv.read_bytes()
    with serial_csv.open(newline="") as handle:
        assert next(csv.reader(handle)) == [
            "projectile",
            "target",
            "projectile_energy_ev",
            "area_quantile",
            "theta_cm_rad",
        ]

    before = serial_csv.read_bytes()
    resumed_csv, _ = generate_kernel_tables(
        serial, tmp_path / "serial", resume=True, show_progress=False
    )
    assert resumed_csv.read_bytes() == before


def test_corrupt_energy_checkpoint_is_recomputed(tmp_path):
    config = KernelTableConfig(
        projectiles=("H",),
        energy_min_ev=1.0e4,
        energy_max_ev=2.0e4,
        base_energy_points=2,
        axis_relative_tolerance=0.05,
        max_energy_points=8,
        max_impact_points=128,
        quadrature_order=32,
        workers=1,
    )
    _, manifest_path = generate_kernel_tables(
        config, tmp_path / "recover", show_progress=False
    )
    checkpoint = next((tmp_path / "recover" / ".checkpoints").rglob("H_H.json"))
    checkpoint.write_text("{}\n", encoding="utf-8")
    generate_kernel_tables(
        config, tmp_path / "recover", resume=True, show_progress=False
    )
    recovered = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert recovered["point_count"] >= 2
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["schema_version"] == 4


@pytest.mark.parametrize(
    "field,value",
    (
        ("projectiles", ()),
        ("energy_min_ev", float("nan")),
        ("axis_relative_tolerance", float("nan")),
        ("minimum_turning_potential_ev", float("inf")),
    ),
)
def test_invalid_table_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        KernelTableConfig(**{field: value})
