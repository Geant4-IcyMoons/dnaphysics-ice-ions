from pathlib import Path
import sys

import numpy as np


PHYSICS_ICE = Path(__file__).resolve().parents[1] / "python_scripts" / "physics_ice"
sys.path.insert(0, str(PHYSICS_ICE))

import ctmc_tail_supervisor as supervisor  # noqa: E402


def test_parse_qstat_states_ignores_array_parent() -> None:
    output = """Job id Name User Time Use S Queue
----- ----- ---- -------- - -----
110831[].pbs02 c_mc user 0 B idlex
110831[0].pbs02 c_mc user 01:02:03 R idlex
110831[1].pbs02 c_mc user 0 Q idlex
110831[2].pbs02 c_mc user 01:04:03 X idlex
110900.pbs02 c_fix user 00:01:00 R idlex
"""
    assert supervisor.parse_qstat_states(output) == ["R", "Q", "X", "R"]


def test_pbs_parent_job_id_accepts_scalar_and_array_identifiers() -> None:
    assert supervisor.pbs_parent_job_id("110831[]") == "110831"
    assert supervisor.pbs_parent_job_id("110900") == "110900"


def test_rebalance_waits_for_queued_or_held_writers() -> None:
    common = {
        "fraction": 0.68,
        "trigger_fraction": 0.01,
        "active": 0,
        "active_threshold": 84,
    }
    assert not supervisor.should_rebalance(**common, waiting=336)
    assert not supervisor.should_rebalance(**common, waiting=1)
    assert supervisor.should_rebalance(**common, waiting=0)


def test_failure_diagnostics_select_only_current_generation(tmp_path: Path) -> None:
    base = tmp_path / "carbon_charge_exchange_ctmc_failure_shard_0052.json"
    base.write_text('{"integrator_policy_version": 1}')
    rebalance = (
        tmp_path
        / "carbon_charge_exchange_ctmc_failure_shard_0064_rebalance_abc.json"
    )
    rebalance.write_text('{"integrator_policy_version": 2}')

    base_records = supervisor.failure_diagnostics(tmp_path, "carbon", None)
    assert [(shard, payload["integrator_policy_version"]) for shard, _, payload in base_records] == [(52, 1)]
    rebalance_records = supervisor.failure_diagnostics(
        tmp_path, "carbon", "abc"
    )
    assert [(shard, payload["integrator_policy_version"]) for shard, _, payload in rebalance_records] == [(64, 2)]


def test_recovery_policy_is_bounded_and_versioned() -> None:
    current = supervisor.CTMC_INTEGRATOR_POLICY_VERSION
    assert supervisor.recovery_action(current - 1, 0, 3) == "resubmit"
    assert (
        supervisor.recovery_action(current - 1, 3, 3)
        == "quarantine_attempts"
    )
    assert (
        supervisor.recovery_action(current, 0, 3)
        == "quarantine_current_policy"
    )


def test_failure_checkpoint_signature_must_match(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.npz"
    np.savez(checkpoint, signature=np.asarray("expected"))
    payload = {
        "checkpoint": str(checkpoint),
        "configuration_signature": "expected",
    }
    assert supervisor.validate_failure_checkpoint(payload) is None
    payload["configuration_signature"] = "wrong"
    assert "signatures differ" in (
        supervisor.validate_failure_checkpoint(payload) or ""
    )


def test_checkpoint_paths_encode_generation(tmp_path: Path) -> None:
    paths = supervisor.checkpoint_paths(tmp_path, "carbon", 2, "abc123")
    assert paths == [
        tmp_path
        / "carbon_charge_exchange_ctmc_checkpoint.rebalance-abc123.shard-00000-of-00002.npz",
        tmp_path
        / "carbon_charge_exchange_ctmc_checkpoint.rebalance-abc123.shard-00001-of-00002.npz",
    ]


def test_total_trajectories_counts_only_active_channels() -> None:
    total = supervisor._total_trajectories(
        np.asarray([1.0, 2.0]),
        np.asarray([0, 1]),
        np.asarray([0.0, 2.0]),
        100,
        [1.0, 3.0],
        [1.0],
    )
    # At b=0: two target channels plus loss for q=0; at b=2: one target.
    # Thus each energy has (3 + 1) for q=0 and (2 + 1) for q=1.
    assert total == 2 * 7 * 100
