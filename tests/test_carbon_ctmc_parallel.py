from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

PHYSICS_SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "python_scripts" / "physics_ice"
)
sys.path.insert(0, str(PHYSICS_SCRIPT_DIR))

import charge_exchange_ctmc as ctmc  # noqa: E402


@pytest.fixture
def lithium_projectile():
    ctmc.select_projectile("lithium")
    try:
        yield ctmc.PROJECTILE
    finally:
        ctmc.select_projectile("carbon")


@pytest.fixture
def oxygen_projectile():
    ctmc.select_projectile("oxygen")
    try:
        yield ctmc.PROJECTILE
    finally:
        ctmc.select_projectile("carbon")


@pytest.fixture
def sulfur_projectile():
    ctmc.select_projectile("sulfur")
    try:
        yield ctmc.PROJECTILE
    finally:
        ctmc.select_projectile("carbon")


def _config(*, trajectories: int = 2, chunk_size: int = 1) -> ctmc.CTMCConfig:
    return ctmc.CTMCConfig(
        backend="scipy",
        trajectories=trajectories,
        trajectory_chunk_size=chunk_size,
        target_bmax_au=(1.0,) * len(ctmc.WATER_ORBITALS),
        loss_bmax_au=(1.0,) * len(ctmc.CARBON_OUTER_ORBITAL),
        radial_grid_points=128,
        start_separation_au=(2.0,),
        boundary_extension_factor=2.0,
        minimum_integration_time_au=(1.0,),
        rtol=1.0e-5,
        atol=1.0e-7,
        retry_rtol=3.0e-6,
        retry_atol=3.0e-8,
        max_step_au=float("inf"),
        minimum_radius_au=1.0e-10,
        maximum_relative_energy_drift=1.0e-3,
        max_failure_fraction=0.01,
        maximum_integration_steps=1000,
        seed=12345,
    )


def test_channel_rng_is_chunk_size_independent() -> None:
    full = ctmc._channel_rng(7, 3, 4, 5, 2, 0).random(50)
    split_trajectories = 4
    split_draws = (
        split_trajectories * ctmc.RANDOM_DRAWS_PER_TRAJECTORY
    )
    split = np.concatenate(
        (
            ctmc._channel_rng(7, 3, 4, 5, 2, 0).random(split_draws),
            ctmc._channel_rng(
                7, 3, 4, 5, 2, split_trajectories
            ).random(50 - split_draws),
        )
    )
    np.testing.assert_array_equal(full, split)


def test_published_initial_momentum_is_perpendicular_to_radius() -> None:
    core = ctmc.CorePotential.from_zn(
        ctmc.WATER_PSEUDO_NUCLEAR_CHARGE,
        9,
    )
    position, velocity = ctmc.sample_bound_electron(
        core,
        ctmc.WATER_MASS_AU,
        ctmc.WATER_ORBITALS[0].binding_eV,
        np.zeros(3),
        np.zeros(3),
        128,
        np.random.default_rng(123),
    )
    assert abs(float(np.dot(position, velocity))) < 1.0e-12


@pytest.mark.parametrize("shard_count", [1, 2, 4, 7, 14, 21, 256])
def test_node_shards_are_disjoint_and_cover_grid(shard_count: int) -> None:
    shape = (3, 7, 5)
    masks = [
        ctmc.owned_point_mask(shape, shard_count, index)
        for index in range(shard_count)
    ]
    coverage = np.sum(masks, axis=0)
    np.testing.assert_array_equal(coverage, np.ones(shape, dtype=int))


def test_seven_shards_assign_one_carbon_charge_to_each_job() -> None:
    shape = (3, 7, 5)
    for shard_index in range(7):
        mask = ctmc.owned_point_mask(shape, 7, shard_index)
        assert np.all(mask[:, shard_index, :])
        assert not np.any(np.delete(mask, shard_index, axis=1))


def test_fourteen_shards_assign_two_jobs_to_each_charge() -> None:
    shape = (3, 7, 5)
    masks = [ctmc.owned_point_mask(shape, 14, index) for index in range(14)]
    for charge_index in range(7):
        first = masks[charge_index]
        second = masks[charge_index + 7]
        assert not np.any(np.delete(first, charge_index, axis=1))
        assert not np.any(np.delete(second, charge_index, axis=1))
        coverage = (
            first[:, charge_index, :].astype(int)
            + second[:, charge_index, :].astype(int)
        )
        np.testing.assert_array_equal(
            coverage,
            np.ones((shape[0], shape[2]), dtype=int),
        )


def test_twenty_one_shards_assign_three_jobs_to_each_charge() -> None:
    shape = (4, 7, 5)
    masks = [ctmc.owned_point_mask(shape, 21, index) for index in range(21)]
    for charge_index in range(7):
        charge_masks = [
            masks[charge_index + 7 * replica][:, charge_index, :]
            for replica in range(3)
        ]
        np.testing.assert_array_equal(
            np.sum(charge_masks, axis=0),
            np.ones((shape[0], shape[2]), dtype=int),
        )
        for replica in range(3):
            assert not np.any(
                np.delete(
                    masks[charge_index + 7 * replica],
                    charge_index,
                    axis=1,
                )
            )


def test_chunk_aggregation_finalizes_point() -> None:
    energies = np.asarray([1000.0])
    charges = np.asarray([6])
    impact = np.asarray([0.0])
    shape = (1, 1, 1)
    config = _config()
    accumulators = ctmc.create_trajectory_accumulators(shape)
    pi = np.full(shape + (len(ctmc.WATER_ORBITALS),), np.nan)
    pc = np.full_like(pi, np.nan)
    pl = np.full(shape, np.nan)
    done = np.zeros(shape, dtype=bool)
    failures = np.zeros(shape, dtype=np.int64)
    successes = np.zeros(shape, dtype=np.int64)
    drift = np.zeros(shape)

    finalized = False
    for channel in range(len(ctmc.WATER_ORBITALS)):
        result = (0, 0, 0, channel, 0, 2, 1, 1, 0, 2, 1.0e-9)
        finalized = ctmc._commit_trajectory_chunk(
            result,
            energies=energies,
            charges=charges,
            impact_au=impact,
            config=config,
            accumulators=accumulators,
            pi=pi,
            pc=pc,
            pl=pl,
            done=done,
            failures=failures,
            successes=successes,
            maximum_energy_drift=drift,
        )

    assert finalized
    assert done.item()
    np.testing.assert_array_equal(pi[0, 0, 0], np.full(5, 0.5))
    np.testing.assert_array_equal(pc[0, 0, 0], np.full(5, 0.5))
    assert pl.item() == 0.0
    assert successes.item() == 10
    assert failures.item() == 0
    assert drift.item() == 1.0e-9


def test_signature_excludes_safe_scheduler_tuning() -> None:
    energies = np.asarray([1000.0])
    charges = np.asarray([6])
    impact = np.asarray([0.0, 1.0])
    config = _config(chunk_size=1)
    tuned = dataclasses.replace(config, trajectory_chunk_size=2048)

    assert ctmc.configuration_signature(
        energies, charges, impact, config
    ) == ctmc.configuration_signature(energies, charges, impact, tuned)


def test_checkpoint_roundtrip_preserves_partial_trajectory_counts(
    tmp_path: Path,
) -> None:
    shape = (1, 1, 1)
    accumulators = ctmc.create_trajectory_accumulators(shape)
    accumulators.primary_events[0, 0, 0, 1] = 3
    accumulators.secondary_events[0, 0, 0, 1] = 2
    accumulators.successes[0, 0, 0, 1] = 7
    accumulators.completed[0, 0, 0, 1] = 7
    path = tmp_path / "checkpoint.npz"
    signature = "roundtrip"
    arrays = {
        "energies": np.asarray([1.0]),
        "charges": np.asarray([0]),
        "impact": np.asarray([0.25]),
        "pi": np.full(shape + (len(ctmc.WATER_ORBITALS),), np.nan),
        "pc": np.full(shape + (len(ctmc.WATER_ORBITALS),), np.nan),
        "pl": np.full(shape, np.nan),
        "done": np.zeros(shape, dtype=bool),
        "failures": np.zeros(shape, dtype=np.int64),
        "successes": np.zeros(shape, dtype=np.int64),
        "drift": np.zeros(shape),
    }

    ctmc.save_checkpoint(
        path,
        signature=signature,
        energies=arrays["energies"],
        charges=arrays["charges"],
        impact_au=arrays["impact"],
        pi=arrays["pi"],
        pc=arrays["pc"],
        pl=arrays["pl"],
        done=arrays["done"],
        failures=arrays["failures"],
        successes=arrays["successes"],
        maximum_energy_drift=arrays["drift"],
        accumulators=accumulators,
        execution_workers=256,
    )
    loaded = ctmc.load_checkpoint(path, signature)

    assert loaded is not None
    assert loaded["channel_primary_events"][0, 0, 0, 1] == 3
    assert loaded["channel_secondary_events"][0, 0, 0, 1] == 2
    assert loaded["channel_successes"][0, 0, 0, 1] == 7
    assert loaded["channel_completed"][0, 0, 0, 1] == 7
    assert int(loaded["execution_workers"]) == 256


def test_zero_failure_policy_identifies_exact_uncommitted_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = dataclasses.replace(_config(), max_failure_fraction=0.0)
    monkeypatch.setattr(ctmc, "_WORKER_CONFIG", config)
    monkeypatch.setattr(
        ctmc,
        "simulate_one_trajectory",
        lambda **_: ("integration_failure", False, float("inf")),
    )
    task = (
        2,
        3,
        1.0,
        4,
        5,
        1.25,
        0,
        7,
        2,
        10_000.0,
        10_000.0,
    )

    with pytest.raises(ctmc.TrajectoryIntegrationFailure) as captured:
        ctmc._compute_trajectory_chunk(task)

    detail = captured.value.args[0]
    assert detail["energy_index"] == 2
    assert detail["charge_state"] == 4
    assert detail["impact_index"] == 5
    assert detail["channel_index"] == 0
    assert detail["trajectory_index"] == 7
    assert detail["failure_outcome"] == "integration_failure"
    assert np.isinf(detail["relative_total_energy_drift"])
    assert detail["rtol"] == config.rtol
    assert detail["atol"] == config.atol


def test_failure_diagnostic_records_deterministic_resume(
    tmp_path: Path,
) -> None:
    path = tmp_path / "failure.json"
    checkpoint = tmp_path / "checkpoint.npz"
    detail = {"trajectory_index": 19, "channel_index": 5}

    ctmc.save_failure_diagnostic(
        path,
        signature="signature",
        checkpoint_path=checkpoint,
        detail=detail,
    )

    payload = json.loads(path.read_text())
    assert payload["detail"] == detail
    assert payload["configuration_signature"] == "signature"
    assert payload["checkpoint"] == str(checkpoint)
    assert "not committed" in payload["resume_behavior"]


def test_default_failure_policy_is_strict() -> None:
    assert ctmc.build_parser().get_default("max_failure_fraction") == 0.0


def test_default_tolerances_are_ensemble_converged() -> None:
    parser = ctmc.build_parser()
    assert parser.get_default("rtol") == 1.0e-11
    assert parser.get_default("atol") == 1.0e-13
    assert parser.get_default("retry_rtol") == 1.0e-12
    assert parser.get_default("retry_atol") == 1.0e-14
    assert parser.get_default("maximum_relative_energy_drift") == 1.0e-3


def test_published_oxygen_garvey_rows_are_reindexed_by_spectators() -> None:
    seven = ctmc.SCREENING_BY_SPECTATORS[7]
    eight = ctmc.SCREENING_BY_SPECTATORS[8]
    assert seven == ctmc.ScreeningParameters(
        1.360, 0.4613, 2.410, 0.3925
    )
    assert eight == ctmc.ScreeningParameters(
        1.508, 0.4602, 2.590, 0.3755
    )


def test_nist_oxygen_outer_shell_data(oxygen_projectile) -> None:
    expected = {
        0: (13.618055, 4),
        1: (35.12112, 3),
        2: (54.93554, 2),
        3: (77.41350, 1),
        4: (113.8990, 2),
        5: (138.1189, 1),
        6: (739.32697, 2),
        7: (871.4099138, 1),
    }
    assert {
        q: (orbital.binding_eV, orbital.active_electrons)
        for q, orbital in oxygen_projectile.outer_orbitals.items()
    } == expected


def test_nist_oxygen_bare_nuclear_mass(oxygen_projectile) -> None:
    assert oxygen_projectile.mass_au == pytest.approx(29148.9497)


def test_oxygen_signature_fingerprints_projectile_mass(
    oxygen_projectile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    energies = np.asarray([1000.0])
    charges = np.asarray([8])
    impact = np.asarray([0.0, 1.0])
    config = dataclasses.replace(
        _config(),
        projectile="oxygen",
        loss_bmax_au=(1.0,) * len(ctmc.OXYGEN_OUTER_ORBITAL),
    )
    original = ctmc.configuration_signature(
        energies,
        charges,
        impact,
        config,
    )
    monkeypatch.setitem(
        ctmc.PROJECTILES,
        "oxygen",
        dataclasses.replace(
            oxygen_projectile,
            mass_au=oxygen_projectile.mass_au + 1.0,
        ),
    )
    changed = ctmc.configuration_signature(
        energies,
        charges,
        impact,
        config,
    )
    assert changed != original


def test_oxygen_trajectory_cores_use_published_spectator_rows(
    oxygen_projectile,
) -> None:
    _, neutral_target_projectile = ctmc._trajectory_cores(0, "target")
    _, singly_charged_target_projectile = ctmc._trajectory_cores(1, "target")
    _, neutral_loss_projectile = ctmc._trajectory_cores(0, "projectile")
    assert neutral_target_projectile.spectators == 8
    assert singly_charged_target_projectile.spectators == 7
    assert neutral_loss_projectile.spectators == 7


def test_oxygen_many_electron_loss_uses_outer_shell_occupancy(
    oxygen_projectile,
) -> None:
    pi = np.zeros((2, len(ctmc.WATER_ORBITALS)))
    pc = np.zeros_like(pi)
    pl = np.asarray([0.1, 0.4])
    channels = ctmc.many_electron_probabilities(pi, pc, pl, 0)
    np.testing.assert_allclose(channels["SL"], 4.0 * pl * (1.0 - pl) ** 3)
    stripped = ctmc.many_electron_probabilities(pi, pc, pl, 8)
    np.testing.assert_array_equal(stripped["SL"], np.zeros_like(pl))


def test_eighty_four_shards_cover_nine_oxygen_charges(
    oxygen_projectile,
) -> None:
    shape = (3, 9, 5)
    coverage = np.sum(
        [
            ctmc.owned_point_mask(shape, 84, shard_index)
            for shard_index in range(84)
        ],
        axis=0,
    )
    np.testing.assert_array_equal(coverage, np.ones(shape, dtype=int))


def test_nist_lithium_outer_shell_data(lithium_projectile) -> None:
    expected = {
        0: (5.391714996, 1),
        1: (75.6400970, 2),
        2: (122.45435913, 1),
    }
    assert {
        q: (orbital.binding_eV, orbital.active_electrons)
        for q, orbital in lithium_projectile.outer_orbitals.items()
    } == expected


def test_nist_lithium_bare_nuclear_mass(lithium_projectile) -> None:
    assert lithium_projectile.mass_au == pytest.approx(12786.3922820)


def test_lithium_extension_preserves_paper_grid_and_adds_100_mev(
    lithium_projectile,
) -> None:
    args = ctmc.build_parser().parse_args([])
    energies = ctmc.build_energy_grid(args)
    assert energies.size == 11
    assert energies[0] == pytest.approx(1.0)
    assert energies[9] == pytest.approx(1.0e4)
    assert energies[10] == pytest.approx(100_000.0 / 7.0)
    np.testing.assert_allclose(
        energies[1:10] / energies[:9],
        np.full(9, 10.0 ** (4.0 / 9.0)),
    )
    np.testing.assert_array_equal(
        ctmc.parse_charges(
            lithium_projectile.default_charges,
            1,
            lithium_projectile.nuclear_charge,
        ),
        np.arange(4),
    )


def test_lithium_trajectory_cores_use_published_spectator_rows(
    lithium_projectile,
) -> None:
    _, neutral_target_projectile = ctmc._trajectory_cores(0, "target")
    _, singly_charged_target_projectile = ctmc._trajectory_cores(1, "target")
    _, neutral_loss_projectile = ctmc._trajectory_cores(0, "projectile")
    _, stripped_target_projectile = ctmc._trajectory_cores(3, "target")
    assert neutral_target_projectile.spectators == 3
    assert singly_charged_target_projectile.spectators == 2
    assert neutral_loss_projectile.spectators == 2
    assert stripped_target_projectile.spectators == 0


def test_lithium_many_electron_loss_uses_outer_shell_occupancy(
    lithium_projectile,
) -> None:
    pi = np.zeros((2, len(ctmc.WATER_ORBITALS)))
    pc = np.zeros_like(pi)
    pl = np.asarray([0.1, 0.4])
    neutral = ctmc.many_electron_probabilities(pi, pc, pl, 0)
    np.testing.assert_allclose(neutral["SL"], pl)
    singly_charged = ctmc.many_electron_probabilities(pi, pc, pl, 1)
    np.testing.assert_allclose(
        singly_charged["SL"],
        2.0 * pl * (1.0 - pl),
    )
    stripped = ctmc.many_electron_probabilities(pi, pc, pl, 3)
    np.testing.assert_array_equal(stripped["SL"], np.zeros_like(pl))


def test_eighty_four_shards_cover_four_lithium_charges_evenly(
    lithium_projectile,
) -> None:
    shape = (11, 4, 101)
    masks = [
        ctmc.owned_point_mask(shape, 84, shard_index)
        for shard_index in range(84)
    ]
    np.testing.assert_array_equal(
        np.sum(masks, axis=0),
        np.ones(shape, dtype=int),
    )
    for charge_index in range(4):
        charge_masks = [
            masks[charge_index + 4 * replica][:, charge_index, :]
            for replica in range(21)
        ]
        np.testing.assert_array_equal(
            np.sum(charge_masks, axis=0),
            np.ones((shape[0], shape[2]), dtype=int),
        )


def test_lithium_loss_cutoffs_require_three_values(
    lithium_projectile,
) -> None:
    assert ctmc.parse_loss_bmax("20;20;20") == (20.0, 20.0, 20.0)
    with pytest.raises(ValueError, match="requires 3"):
        ctmc.parse_loss_bmax("20;20")


def test_sulfur_garvey_rows_are_direct_table_values() -> None:
    expected = {
        11: (1.492, 0.3452, 3.010, 0.3269),
        12: (1.170, 0.3191, 3.170, 0.3087),
        13: (1.012, 0.2933, 3.260, 0.2958),
        14: (0.954, 0.2659, 3.330, 0.2857),
        15: (0.926, 0.2478, 3.392, 0.2739),
        16: (0.933, 0.2368, 3.447, 0.2633),
    }
    assert {
        spectators: dataclasses.astuple(
            ctmc.SCREENING_BY_SPECTATORS[spectators]
        )
        for spectators in expected
    } == expected


def test_nist_sulfur_outer_shell_data(sulfur_projectile) -> None:
    expected = {
        0: (10.3600167, 4),
        1: (23.33788, 3),
        2: (34.86, 2),
        3: (47.222, 1),
        4: (72.5945, 2),
        5: (88.0529, 1),
        6: (280.954, 6),
        7: (328.794, 5),
        8: (379.84, 4),
        9: (447.7, 3),
        10: (504.55, 2),
        11: (564.41, 1),
        12: (651.96, 2),
        13: (706.994, 1),
        14: (3223.78057, 2),
        15: (3494.188518, 1),
    }
    assert {
        q: (orbital.binding_eV, orbital.active_electrons)
        for q, orbital in sulfur_projectile.outer_orbitals.items()
    } == expected


def test_nist_sulfur_bare_nuclear_mass(sulfur_projectile) -> None:
    assert sulfur_projectile.mass_au == pytest.approx(58265.5417)


def test_sulfur_trajectory_cores_use_published_spectator_rows(
    sulfur_projectile,
) -> None:
    _, neutral_target_projectile = ctmc._trajectory_cores(0, "target")
    _, singly_charged_target_projectile = ctmc._trajectory_cores(1, "target")
    _, neutral_loss_projectile = ctmc._trajectory_cores(0, "projectile")
    _, stripped_target_projectile = ctmc._trajectory_cores(16, "target")
    assert neutral_target_projectile.spectators == 16
    assert singly_charged_target_projectile.spectators == 15
    assert neutral_loss_projectile.spectators == 15
    assert stripped_target_projectile.spectators == 0


def test_sulfur_many_electron_loss_supports_six_2p_electrons(
    sulfur_projectile,
) -> None:
    pi = np.zeros((2, len(ctmc.WATER_ORBITALS)))
    pc = np.zeros_like(pi)
    pl = np.asarray([0.1, 0.4])
    channels = ctmc.many_electron_probabilities(pi, pc, pl, 6)
    np.testing.assert_allclose(channels["SL"], 6.0 * pl * (1.0 - pl) ** 5)
    stripped = ctmc.many_electron_probabilities(pi, pc, pl, 16)
    np.testing.assert_array_equal(stripped["SL"], np.zeros_like(pl))


def test_eighty_five_shards_cover_seventeen_sulfur_charges(
    sulfur_projectile,
) -> None:
    shape = (3, 17, 5)
    masks = [
        ctmc.owned_point_mask(shape, 85, shard_index)
        for shard_index in range(85)
    ]
    np.testing.assert_array_equal(
        np.sum(masks, axis=0),
        np.ones(shape, dtype=int),
    )
    for charge_index in range(17):
        charge_masks = [
            masks[charge_index + 17 * replica][:, charge_index, :]
            for replica in range(5)
        ]
        np.testing.assert_array_equal(
            np.sum(charge_masks, axis=0),
            np.ones((shape[0], shape[2]), dtype=int),
        )


def test_sulfur_loss_cutoffs_require_sixteen_values(
    sulfur_projectile,
) -> None:
    values = ctmc.parse_loss_bmax(";".join(["20"] * 16))
    assert values == (20.0,) * 16
    with pytest.raises(ValueError, match="requires 16"):
        ctmc.parse_loss_bmax(";".join(["20"] * 15))


def test_sulfur_additions_preserve_oxygen_production_signature(
    oxygen_projectile,
) -> None:
    energies = np.geomspace(1.0, 1.0e4, 41)
    charges = np.arange(9, dtype=int)
    impact = np.linspace(0.0, 25.0, 101)
    config = ctmc.CTMCConfig(
        backend="numba",
        trajectories=10_000,
        trajectory_chunk_size=8,
        target_bmax_au=(25.0,) * 5,
        loss_bmax_au=(20.0,) * 8,
        radial_grid_points=4096,
        start_separation_au=(20_000.0,) + (10_000.0,) * 39 + (1_000.0,),
        boundary_extension_factor=2.0,
        minimum_integration_time_au=tuple(
            ctmc.paper_minimum_integration_time_au(energy)
            for energy in energies
        ),
        rtol=1.0e-11,
        atol=1.0e-13,
        retry_rtol=1.0e-12,
        retry_atol=1.0e-14,
        max_step_au=float("inf"),
        minimum_radius_au=1.0e-10,
        maximum_relative_energy_drift=1.0e-3,
        max_failure_fraction=0.0,
        maximum_integration_steps=0,
        seed=20130641,
        projectile="oxygen",
    )
    assert ctmc.configuration_signature(
        energies,
        charges,
        impact,
        config,
    ) == "0fb3dae255288ef1ada699228c6a1a404fb2781ab9c92fa6806307d7d441b526"


@pytest.mark.parametrize("projectile", ("carbon", "oxygen", "sulfur"))
def test_all_ctmc_projectiles_use_material_density_at_runtime(
    projectile: str,
) -> None:
    ctmc.select_projectile(projectile)
    try:
        scaling = ctmc.phase_density_scaling_metadata()
    finally:
        ctmc.select_projectile("carbon")

    assert not scaling["microscopic_tables_scaled"]
    phases = scaling["phases"]
    assert phases["ice_am"]["mass_density_g_cm3"] == pytest.approx(0.940)
    assert phases["ice_hex"]["mass_density_g_cm3"] == pytest.approx(0.917)
    assert phases["water"]["mass_density_g_cm3"] == pytest.approx(1.000)
    assert phases["ice_am"]["molecular_number_density_cm3"] == pytest.approx(
        3.142228327508648e22
    )
    assert phases["ice_hex"]["molecular_number_density_cm3"] == pytest.approx(
        3.0653440173674787e22
    )
    assert (
        phases["ice_am"]["molecular_number_density_cm3"]
        / phases["ice_hex"]["molecular_number_density_cm3"]
    ) == pytest.approx(0.940 / 0.917)


def test_cpp_and_python_ice_phase_densities_match() -> None:
    project_root = PHYSICS_SCRIPT_DIR.parents[1]
    for relative_path in (
        Path("include/IcePhaseProperties.hh"),
        Path("proton-pipeline/include/IcePhaseProperties.hh"),
    ):
        header = (project_root / relative_path).read_text(encoding="utf-8")
        assert "kAmorphousIceDensityGPerCm3 = 0.94;" in header
        assert "kHexagonalIceDensityGPerCm3 = 0.917;" in header
        assert "kWaterDensityGPerCm3 = 1.0;" in header


def test_ctmc_output_metadata_records_runtime_density_scaling(
    tmp_path: Path,
) -> None:
    energies = np.asarray([1.0])
    charges = np.asarray([0])
    impact = np.asarray([0.0, 1.0])
    shape = (1, 1, 2)
    pi = np.zeros(shape + (len(ctmc.WATER_ORBITALS),))
    pc = np.zeros_like(pi)
    pl = np.zeros(shape)
    _, _, _, metadata_path = ctmc.write_outputs(
        tmp_path,
        energies=energies,
        charges=charges,
        impact_au=impact,
        pi=pi,
        pc=pc,
        pl=pl,
        failures=np.zeros(shape, dtype=np.int64),
        successes=np.ones(shape, dtype=np.int64),
        maximum_energy_drift=np.zeros(shape),
        config=_config(),
        workers=1,
        shard_count=1,
        signature="density-test",
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert not metadata["density_applied"]
    assert metadata["cross_section_unit"] == "cm2 per H2O molecule"
    assert not metadata["phase_density_scaling"]["microscopic_tables_scaled"]
    assert (
        metadata["phase_density_scaling"]["phases"]["ice_am"][
            "geant4_material"
        ]
        == "G4_WATER_ICE_AM"
    )
    assert (
        metadata["phase_density_scaling"]["phases"]["ice_hex"][
            "geant4_material"
        ]
        == "G4_WATER_ICE_HEX"
    )


def test_paper_boundary_extension_is_allowed_for_both_bound_convergence() -> None:
    parser = ctmc.build_parser()
    args = parser.parse_args(
        [
            "--start-separation-au",
            "20000;1000",
            "--target-bmax-au",
            "25;25;25;25;1",
            "--loss-bmax-au",
            "20;20;20;20;20;20",
        ]
    )
    separations, minimum_times = ctmc.resolve_paper_trajectory_boundaries(
        args,
        np.asarray([1.0, 1.0e4]),
    )

    assert separations == (20_000.0, 1_000.0)
    assert minimum_times == (10_000.0, 1_000.0)


def test_failed_integration_retries_identical_state_at_tighter_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[np.ndarray, float, float]] = []

    def fake_integrator(
        initial: np.ndarray,
        _minimum_time: float,
        _target: np.ndarray,
        _projectile: np.ndarray,
        rtol: float,
        atol: float,
        *_: object,
    ) -> tuple[bool, np.ndarray, int, int]:
        calls.append((initial.copy(), rtol, atol))
        return len(calls) == 3, initial.copy(), 1, 0

    monkeypatch.setattr(ctmc, "integrate_relative_dop853", fake_integrator)
    config = dataclasses.replace(_config(), backend="numba")
    outcome, success, _ = ctmc.simulate_one_trajectory(
        energy_keV_u=1.0,
        charge_state=0,
        impact_parameter_au=0.0,
        bound_to="target",
        binding_eV=ctmc.WATER_ORBITALS[0].binding_eV,
        start_separation_au=100.0,
        minimum_integration_time_au=1.0,
        config=config,
        rng=np.random.default_rng(7),
    )

    assert success
    assert outcome == "retained_target"
    assert len(calls) == 3
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    np.testing.assert_array_equal(calls[0][0], calls[2][0])
    assert calls[0][1:] == (config.rtol, config.atol)
    assert calls[1][1:] == (config.retry_rtol, config.retry_atol)
    assert calls[2][1:] == (
        0.1 * config.retry_rtol,
        0.1 * config.retry_atol,
    )


def test_failed_integration_uses_identical_regularized_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_states: list[np.ndarray] = []
    regularized_states: list[np.ndarray] = []

    def failed_integrator(
        initial: np.ndarray,
        *_: object,
    ) -> tuple[bool, np.ndarray, int, int]:
        direct_states.append(initial.copy())
        return False, initial.copy(), 1, 1

    def regularized_integrator(
        initial: np.ndarray,
        *_: object,
    ) -> tuple[bool, np.ndarray, int, int]:
        regularized_states.append(initial.copy())
        return True, initial.copy(), 1, 0

    monkeypatch.setattr(ctmc, "integrate_relative_dop853", failed_integrator)
    monkeypatch.setattr(
        ctmc,
        "integrate_relative_dop853_regularized",
        regularized_integrator,
    )
    config = dataclasses.replace(_config(), backend="numba")
    outcome, success, drift = ctmc.simulate_one_trajectory(
        energy_keV_u=1.0,
        charge_state=0,
        impact_parameter_au=0.0,
        bound_to="target",
        binding_eV=ctmc.WATER_ORBITALS[0].binding_eV,
        start_separation_au=100.0,
        minimum_integration_time_au=1.0,
        config=config,
        rng=np.random.default_rng(37),
    )

    assert success
    assert outcome == "retained_target"
    assert drift == pytest.approx(0.0)
    assert len(direct_states) == 3
    assert len(regularized_states) == 1
    for state in direct_states + regularized_states:
        np.testing.assert_array_equal(state, direct_states[0])


def test_excess_energy_drift_retries_identical_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_integrator(
        initial: np.ndarray,
        *_: object,
    ) -> tuple[bool, np.ndarray, int, int]:
        calls.append(initial.copy())
        return True, initial.copy(), 1, 0

    energies = iter((100.0, 100.2, 100.01))
    monkeypatch.setattr(ctmc, "integrate_relative_dop853", fake_integrator)
    monkeypatch.setattr(
        ctmc,
        "relative_three_body_energy",
        lambda *_: next(energies),
    )
    config = dataclasses.replace(_config(), backend="numba")
    _, success, drift = ctmc.simulate_one_trajectory(
        energy_keV_u=1.0,
        charge_state=0,
        impact_parameter_au=0.0,
        bound_to="target",
        binding_eV=ctmc.WATER_ORBITALS[0].binding_eV,
        start_separation_au=100.0,
        minimum_integration_time_au=1.0,
        config=config,
        rng=np.random.default_rng(29),
    )

    assert success
    assert drift == pytest.approx(1.0e-4)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], calls[1])


def test_unconserved_finite_endpoint_is_not_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_integrator(
        initial: np.ndarray,
        *_: object,
    ) -> tuple[bool, np.ndarray, int, int]:
        return True, initial.copy(), 1, 0

    energies = iter((100.0, 100.2, 100.3, 100.15))
    monkeypatch.setattr(ctmc, "integrate_relative_dop853", fake_integrator)
    monkeypatch.setattr(
        ctmc,
        "relative_three_body_energy",
        lambda *_: next(energies),
    )
    config = dataclasses.replace(_config(), backend="numba")
    outcome, success, drift = ctmc.simulate_one_trajectory(
        energy_keV_u=1.0,
        charge_state=0,
        impact_parameter_au=0.0,
        bound_to="target",
        binding_eV=ctmc.WATER_ORBITALS[0].binding_eV,
        start_separation_au=100.0,
        minimum_integration_time_au=1.0,
        config=config,
        rng=np.random.default_rng(31),
    )

    assert not success
    assert outcome == "energy_conservation_failure"
    assert drift == pytest.approx(1.5e-3)


def test_projectile_loss_uses_paper_switched_relative_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integrated: dict[str, object] = {}

    def fake_integrator(
        initial: np.ndarray,
        _minimum_time: float,
        reference_parameters: np.ndarray,
        other_parameters: np.ndarray,
        _rtol: float,
        _atol: float,
        _max_step: float,
        _minimum_radius: float,
        reference_mass: float,
        other_mass: float,
        _maximum_steps: int,
    ) -> tuple[bool, np.ndarray, int, int]:
        integrated.update(
            initial=initial.copy(),
            reference_parameters=reference_parameters.copy(),
            other_parameters=other_parameters.copy(),
            reference_mass=reference_mass,
            other_mass=other_mass,
        )
        return True, initial.copy(), 1, 0

    monkeypatch.setattr(ctmc, "integrate_relative_dop853", fake_integrator)
    endpoint_energies = iter((-1.0, 1.0))
    monkeypatch.setattr(
        ctmc,
        "_electron_core_energy_relative",
        lambda *_: next(endpoint_energies),
    )
    config = dataclasses.replace(_config(), backend="numba")
    outcome, success, _ = ctmc.simulate_one_trajectory(
        energy_keV_u=1.0,
        charge_state=2,
        impact_parameter_au=1.75,
        bound_to="projectile",
        binding_eV=ctmc.CARBON_OUTER_ORBITAL[2].binding_eV,
        start_separation_au=20_000.0,
        minimum_integration_time_au=10_000.0,
        config=config,
        rng=np.random.default_rng(19),
    )

    initial = np.asarray(integrated["initial"])
    _, projectile_core = ctmc._trajectory_cores(2, "projectile")
    target_core, _ = ctmc._trajectory_cores(2, "projectile")
    assert success
    assert outcome == "captured_projectile"
    assert np.linalg.norm(initial[:3]) == pytest.approx(20_000.0, rel=1.0e-6)
    assert np.linalg.norm(initial[3:6]) < 10.0
    np.testing.assert_array_equal(
        integrated["reference_parameters"],
        ctmc._cached_core_parameters(projectile_core),
    )
    np.testing.assert_array_equal(
        integrated["other_parameters"],
        ctmc._cached_core_parameters(target_core),
    )
    assert integrated["reference_mass"] == ctmc.CARBON_MASS_AU
    assert integrated["other_mass"] == ctmc.WATER_MASS_AU


def test_both_bound_trajectory_reuses_phase_at_extended_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integrated_states: list[np.ndarray] = []
    endpoint_energies = iter((-1.0, -1.0, -1.0, 1.0))

    def fake_integrator(
        initial: np.ndarray,
        *_: object,
    ) -> tuple[bool, np.ndarray, int, int]:
        integrated_states.append(initial.copy())
        return True, initial.copy(), 1, 0

    monkeypatch.setattr(ctmc, "integrate_relative_dop853", fake_integrator)
    monkeypatch.setattr(
        ctmc,
        "_electron_core_energy_relative",
        lambda *_: next(endpoint_energies),
    )
    config = dataclasses.replace(_config(), backend="numba")
    rng = np.random.default_rng(11)
    outcome, success, _ = ctmc.simulate_one_trajectory(
        energy_keV_u=1.0,
        charge_state=0,
        impact_parameter_au=0.0,
        bound_to="target",
        binding_eV=ctmc.WATER_ORBITALS[0].binding_eV,
        start_separation_au=100.0,
        minimum_integration_time_au=1.0,
        config=config,
        rng=rng,
    )

    assert success
    assert outcome == "retained_target"
    assert len(integrated_states) == 2
    assert np.linalg.norm(integrated_states[0][:3]) == pytest.approx(100.0)
    assert np.linalg.norm(integrated_states[1][:3]) == pytest.approx(200.0)
    np.testing.assert_array_equal(
        integrated_states[0][3:6],
        integrated_states[1][3:6],
    )
    expected_rng = np.random.default_rng(11)
    expected_rng.random(ctmc.RANDOM_DRAWS_PER_TRAJECTORY)
    assert rng.random() == expected_rng.random()


@pytest.mark.parametrize(
    "initial",
    (
        np.asarray(
            [
                10.0,
                0.0,
                0.0,
                2.0,
                1.0,
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,
                0.2,
                0.0,
            ]
        ),
        # The electron begins nearer the other core, exercising the exact
        # Appendix-A p/t coordinate interchange.
        np.asarray(
            [
                10.0,
                0.0,
                0.0,
                9.0,
                0.3,
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,
                0.2,
                0.0,
            ]
        ),
    ),
)
def test_numba_dop853_matches_scipy_in_paper_coordinates(
    initial: np.ndarray,
) -> None:
    target_core, projectile_core = ctmc._trajectory_cores(0, "target")

    def relative_reference_rhs(
        time_au: float,
        relative_state: np.ndarray,
    ) -> np.ndarray:
        full = ctmc._three_body_rhs(
            time_au,
            ctmc._relative_to_state(relative_state),
            target_core,
            projectile_core,
            1.0e-10,
        )
        derivative = np.empty(12)
        derivative[0:3] = full[3:6] - full[0:3]
        derivative[3:6] = full[6:9] - full[0:3]
        derivative[6:9] = full[12:15] - full[9:12]
        derivative[9:12] = full[15:18] - full[9:12]
        return derivative

    reference = solve_ivp(
        relative_reference_rhs,
        (0.0, 1.0),
        initial,
        method="DOP853",
        t_eval=(1.0,),
        rtol=1.0e-11,
        atol=1.0e-13,
    )
    success, final, _, _ = ctmc.integrate_relative_dop853(
        initial,
        1.0,
        ctmc._cached_core_parameters(target_core),
        ctmc._cached_core_parameters(projectile_core),
        1.0e-11,
        1.0e-13,
        float("inf"),
        1.0e-10,
        ctmc.WATER_MASS_AU,
        ctmc.CARBON_MASS_AU,
        0,
    )
    regularized_success, regularized_final, _, _ = (
        ctmc.integrate_relative_dop853_regularized(
            initial,
            1.0,
            ctmc._cached_core_parameters(target_core),
            ctmc._cached_core_parameters(projectile_core),
            1.0e-11,
            1.0e-13,
            float("inf"),
            1.0e-10,
            ctmc.WATER_MASS_AU,
            ctmc.CARBON_MASS_AU,
            # Disable the conditional invariant projection here to compare
            # the pure Sundman reparameterization with physical-time SciPy.
            1.0,
            0,
        )
    )

    assert success
    assert regularized_success
    assert reference.success
    np.testing.assert_allclose(
        final,
        reference.y[:, -1],
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        regularized_final,
        reference.y[:, -1],
        rtol=1.0e-7,
        atol=1.0e-9,
    )


def test_numba_dop853_matches_scipy_in_projectile_loss_frame() -> None:
    target_core, projectile_core = ctmc._trajectory_cores(2, "projectile")
    initial = np.asarray(
        [
            10.0,
            0.0,
            0.0,
            2.0,
            1.0,
            0.0,
            0.1,
            0.0,
            0.0,
            0.0,
            0.2,
            0.0,
        ]
    )

    def relative_reference_rhs(
        time_au: float,
        relative_state: np.ndarray,
    ) -> np.ndarray:
        full = ctmc._three_body_rhs(
            time_au,
            ctmc._relative_to_state(relative_state, "projectile"),
            target_core,
            projectile_core,
            1.0e-10,
        )
        derivative = np.empty(12)
        derivative[0:3] = full[0:3] - full[3:6]
        derivative[3:6] = full[6:9] - full[3:6]
        derivative[6:9] = full[9:12] - full[12:15]
        derivative[9:12] = full[15:18] - full[12:15]
        return derivative

    reference = solve_ivp(
        relative_reference_rhs,
        (0.0, 1.0),
        initial,
        method="DOP853",
        t_eval=(1.0,),
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    success, final, _, _ = ctmc.integrate_relative_dop853(
        initial,
        1.0,
        ctmc._cached_core_parameters(projectile_core),
        ctmc._cached_core_parameters(target_core),
        1.0e-9,
        1.0e-11,
        float("inf"),
        1.0e-10,
        ctmc.CARBON_MASS_AU,
        ctmc.WATER_MASS_AU,
        0,
    )

    assert success
    assert reference.success
    np.testing.assert_allclose(
        final,
        reference.y[:, -1],
        rtol=1.0e-9,
        atol=1.0e-11,
    )
