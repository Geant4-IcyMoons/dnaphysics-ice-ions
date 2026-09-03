from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


NEP_MBPOL = (
    Path(__file__).resolve().parents[1]
    / "python_scripts"
    / "physics_ice"
    / "nep_mbpol"
)
if str(NEP_MBPOL) not in sys.path:
    sys.path.insert(0, str(NEP_MBPOL))

import adaptive_nlh_particle_transport as adaptive  # noqa: E402
import adaptive_nlh_particle_shards as hard_shards  # noqa: E402


def test_nested_simulator_progress_bypasses_pbs_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure_directory = tmp_path / "structures"
    structure_directory.mkdir()
    structure = structure_directory / "seed1000_final.xyz.gz"
    metadata = structure_directory / "seed1000_final.xyz.json"
    structure.write_text("", encoding="utf-8")
    metadata.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        structure_directory=structure_directory,
        kernels=tmp_path / "kernels",
        projectile="C",
        path_length_angstrom=100.0,
        trajectory_batch_size=100_000,
        meaningful_significant_digits=2,
        interpolation_tolerance=0.005,
        trajectory_cdf_tolerance=0.005,
        confidence=0.95,
        minimum_trajectories=200_000,
        maximum_trajectories=64_000_000,
        extended_maximum_trajectories=None,
        unlimited_trajectories=True,
        dry_run=False,
    )
    case = adaptive.Case(1000, "c_axis", 1.0e5)
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        kwargs["stdout"].write("progress\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(adaptive.subprocess, "run", fake_run)
    code, manifest, _ = adaptive._run_simulator(
        args,
        1,
        tmp_path / "particle",
        case,
        stage="production",
        fixed_trajectories=200_000,
        control_variate=False,
    )

    assert code == 0
    assert captured["check"] is False
    assert captured["stderr"] is subprocess.STDOUT
    assert manifest.parent.joinpath("hard_collision_progress.log").read_text(
        encoding="utf-8"
    ) == "progress\n"


@pytest.mark.parametrize("projectile", ("H", "He", "C", "O", "S"))
def test_seed_and_case_layout_are_generic_for_every_projectile(projectile):
    case = adaptive.Case(1000, "c_axis", 1.0e5)
    seed = adaptive._stable_seed("production", projectile, case)
    assert 0 <= seed < 2**32
    assert seed == adaptive._stable_seed("production", projectile, case)
    assert adaptive._case_key(case) == "seed1000/c_axis/100000eV"


def test_cdf_midpoint_interpolation_is_exact_for_identical_samples():
    sample = np.asarray((0.0, 0.1, 0.4, 1.0), dtype=float)
    assert adaptive._cdf_interpolation_error(sample, sample, sample) == 0.0


def test_cdf_midpoint_interpolation_detects_a_shift():
    lower = np.asarray((0.0, 0.0, 1.0, 1.0), dtype=float)
    upper = lower.copy()
    midpoint = np.asarray((0.0, 1.0, 1.0, 1.0), dtype=float)
    assert adaptive._cdf_interpolation_error(lower, upper, midpoint) == pytest.approx(
        0.25
    )


def test_restart_ceiling_extension_preserves_signed_configuration() -> None:
    args = SimpleNamespace(
        implementation_version=adaptive.IMPLEMENTATION_VERSION,
        projectile="H",
        structure_directory=Path("structures"),
        structure_seeds=[1000, 2000, 3000],
        orientations=list(adaptive.ORIENTATIONS),
        base_energies_ev=list(adaptive.DEFAULT_BASE_ENERGIES_EV),
        kernels=Path("kernels"),
        meaningful_significant_digits=2,
        interpolation_tolerance=0.005,
        trajectory_cdf_tolerance=0.005,
        confidence=0.95,
        calibration_trajectories=10_000,
        minimum_trajectories=200_000,
        maximum_trajectories=64_000_000,
        extended_maximum_trajectories=512_000_000,
        trajectory_batch_size=10_000,
        path_length_angstrom=100.0,
        maximum_refinement_depth=8,
        maximum_energy_points=257,
    )

    configuration = adaptive._controller_configuration(args)
    assert configuration["maximum_trajectories"] == 64_000_000
    assert "extended_maximum_trajectories" not in configuration
    assert adaptive._runtime_trajectory_ceiling(args) == 512_000_000

    args.unlimited_trajectories = True
    args.extended_maximum_trajectories = None
    assert adaptive._controller_configuration(args) == configuration
    assert adaptive._runtime_trajectory_ceiling(args) == sys.maxsize


def test_hard_case_shards_are_disjoint_and_complete() -> None:
    cases = [
        adaptive.Case(seed, orientation, energy)
        for energy in adaptive.DEFAULT_BASE_ENERGIES_EV
        for seed in adaptive.DEFAULT_STRUCTURE_SEEDS
        for orientation in adaptive.ORIENTATIONS
    ]
    selections = [
        hard_shards._cases_for_shard(cases, 17, index) for index in range(17)
    ]
    flattened = [case for selection in selections for case in selection]
    assert len(flattened) == len(cases)
    assert len(set(flattened)) == len(cases)


def test_hard_shard_prepare_is_restart_stable(tmp_path: Path) -> None:
    args = SimpleNamespace(
        projectile="C",
        structure_directory=NEP_MBPOL.parent / "ice_structures" / "hexagonal_ih_100K_experimental",
        structure_seeds=list(adaptive.DEFAULT_STRUCTURE_SEEDS),
        orientations=list(adaptive.ORIENTATIONS),
        base_energies_ev=list(adaptive.DEFAULT_BASE_ENERGIES_EV),
        kernels=NEP_MBPOL / "collision_kernels",
        output_root=tmp_path,
        meaningful_significant_digits=2,
        interpolation_tolerance=0.005,
        trajectory_cdf_tolerance=0.005,
        confidence=0.95,
        calibration_trajectories=10_000,
        minimum_trajectories=200_000,
        maximum_trajectories=64_000_000,
        extended_maximum_trajectories=None,
        trajectory_batch_size=100_000,
        path_length_angstrom=100.0,
        maximum_refinement_depth=8,
        maximum_energy_points=257,
        unlimited_trajectories=True,
    )

    status, manifest_path, count = hard_shards.prepare_wave(args)
    assert status == "prepared"
    assert count == 54
    manifest = hard_shards.serial._manifest(manifest_path)
    assert len(manifest["cases"]) == 54
    assert len({item["case_key"] for item in manifest["cases"]}) == 54

    resumed_status, resumed_path, resumed_count = hard_shards.prepare_wave(args)
    assert resumed_status == "awaiting_calculations"
    assert resumed_path == manifest_path
    assert resumed_count == 54


def test_pending_wave_can_remove_only_its_restart_ceiling(tmp_path: Path) -> None:
    args = SimpleNamespace(
        projectile="C",
        structure_directory=NEP_MBPOL.parent / "ice_structures" / "hexagonal_ih_100K_experimental",
        structure_seeds=list(adaptive.DEFAULT_STRUCTURE_SEEDS),
        orientations=list(adaptive.ORIENTATIONS),
        base_energies_ev=list(adaptive.DEFAULT_BASE_ENERGIES_EV),
        kernels=NEP_MBPOL / "collision_kernels",
        output_root=tmp_path,
        meaningful_significant_digits=2,
        interpolation_tolerance=0.005,
        trajectory_cdf_tolerance=0.005,
        confidence=0.95,
        calibration_trajectories=10_000,
        minimum_trajectories=200_000,
        maximum_trajectories=64_000_000,
        extended_maximum_trajectories=None,
        trajectory_batch_size=100_000,
        path_length_angstrom=100.0,
        maximum_refinement_depth=8,
        maximum_energy_points=257,
        unlimited_trajectories=False,
    )
    _, wave_path, _ = hard_shards.prepare_wave(args)
    finite = hard_shards.serial._manifest(wave_path)
    assert finite["runtime"]["maximum_trajectories"] == 64_000_000

    args.unlimited_trajectories = True
    status, resumed_path, count = hard_shards.prepare_wave(args)
    resumed = hard_shards.serial._manifest(resumed_path)
    assert status == "awaiting_calculations"
    assert count == 54
    assert resumed_path == wave_path
    assert resumed["runtime"]["unlimited_trajectories"] is True
    assert resumed["runtime_history"][-1]["previous"] == finite["runtime"]


def test_hard_shard_runner_writes_only_its_owned_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(
        projectile="C",
        structure_directory=NEP_MBPOL.parent / "ice_structures" / "hexagonal_ih_100K_experimental",
        structure_seeds=list(adaptive.DEFAULT_STRUCTURE_SEEDS),
        orientations=list(adaptive.ORIENTATIONS),
        base_energies_ev=list(adaptive.DEFAULT_BASE_ENERGIES_EV),
        kernels=NEP_MBPOL / "collision_kernels",
        output_root=tmp_path,
        meaningful_significant_digits=2,
        interpolation_tolerance=0.005,
        trajectory_cdf_tolerance=0.005,
        confidence=0.95,
        calibration_trajectories=10_000,
        minimum_trajectories=200_000,
        maximum_trajectories=64_000_000,
        extended_maximum_trajectories=None,
        trajectory_batch_size=100_000,
        path_length_angstrom=100.0,
        maximum_refinement_depth=8,
        maximum_energy_points=257,
        unlimited_trajectories=True,
    )
    _, wave_path, _ = hard_shards.prepare_wave(args)
    visited: list[adaptive.Case] = []

    def fake_calibration(_args, _workers, _root, case):
        visited.append(case)
        return {"use_control_variate": False}

    def fake_production(_args, _workers, root, case, _calibration):
        path = hard_shards._expected_production_manifest(root, case)
        hard_shards._atomic_json(
            path,
            {
                "configuration": {"completed_trajectories": 200_000},
                "statistical_convergence": {"converged": True},
            },
        )
        return path

    monkeypatch.setattr(hard_shards.serial, "_calibrate_case", fake_calibration)
    monkeypatch.setattr(hard_shards.serial, "_run_production_case", fake_production)
    receipt = hard_shards.run_shard(wave_path, 2, 1, 1)
    payload = hard_shards.serial._manifest(receipt)
    assert len(visited) == 27
    assert {item["case_key"] for item in payload["cases"]} == {
        adaptive._case_key(case) for case in visited
    }
