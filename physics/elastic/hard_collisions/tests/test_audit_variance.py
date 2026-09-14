import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("audit_variance", Path(__file__).parents[1] / "audit_variance.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def stats(a, b):
    a, b = np.asarray(a, dtype=np.longdouble), np.asarray(b, dtype=np.longdouble)
    return [len(a), a.sum(), b.sum(), (a*a).sum(), (b*b).sum(), (a*b).sum()]


def test_ratio_variance_matches_individual_observations():
    a, b = np.arange(1, 17), np.arange(1, 17) % 3 + 1
    rows = np.array([stats(a[:8], b[:8]), stats(a[8:], b[8:])])
    result, order, fractions = audit.ratio_summary(rows)
    ratio = a.sum() / b.sum()
    expected = np.sqrt(len(a) * np.square(a-ratio*b).sum() / (len(a)-1)) / b.sum()
    assert result["estimate"] == pytest.approx(ratio)
    assert result["standard_error"] == pytest.approx(expected)
    assert fractions.sum() == pytest.approx(1)


def test_constant_ratio_has_zero_variance():
    rows = np.array([stats([3, 6], [1, 2]), stats([9, 12], [3, 4])])
    result, _, _ = audit.ratio_summary(rows)
    assert result["standard_error"] == 0


def test_outlier_dominates_without_being_removed():
    rows = np.array([stats([1, 1], [1, 1]) for _ in range(99)] + [stats([1, 1e6], [1, 1])])
    result, order, _ = audit.ratio_summary(rows)
    assert order[0] == 99
    assert result["largest_batch_variance_fraction"] > .99
    assert result["estimate"] == pytest.approx((199+1e6)/200)


def test_negative_variance_rejected_even_for_tiny_units():
    with pytest.raises(ValueError, match="negative"):
        audit.ratio_summary(np.array([[10, 1e-20, 10, 0, 10, 1e-20]], dtype=np.longdouble))


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        audit.ratio_summary(np.array([[10, float("nan"), 10, 1, 10, 1]]))


def test_snapshot_ignores_append_but_rejects_mutation(tmp_path):
    path = tmp_path / "production/seed/direction/1eV/.trajectory_checkpoints/signed"
    path.mkdir(parents=True)
    first = path / "batch_000000000_000000010.manifest.json"
    first.write_text("{}")
    frozen = audit.freeze_case(tmp_path, "seed/direction/1eV")
    (path / "batch_000000010_000000020.manifest.json").write_text("{}")
    audit.verify_snapshot(tmp_path, frozen)
    first.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        audit.verify_snapshot(tmp_path, frozen)


def test_noncontiguous_prefix_rejected(tmp_path):
    path = tmp_path / "production/seed/direction/1eV/.trajectory_checkpoints/signed"
    path.mkdir(parents=True)
    (path / "batch_000000010_000000020.manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Non-contiguous"):
        audit.freeze_case(tmp_path, "seed/direction/1eV")


def test_distribution_hash_and_range(tmp_path):
    sample = tmp_path / "sample.npz"
    np.savez(sample, trajectory=np.array([2, 4]), total_recoil_energy_ev=np.array([1., 9.]),
             final_deflection_rad=np.array([0., .1]))
    record = {"distribution_sample": sample.name, "distribution_sample_sha256": audit.digest(sample.read_bytes()),
              "summary": {"target_distribution_sample_count": 2}, "trajectory_start": 0, "trajectory_stop": 5}
    result = audit.inspect_distribution(tmp_path / "batch.manifest.json", record, 10)
    assert result["recoil_ev"]["top"][0] == {"trajectory": 4, "value": 9.}
    with pytest.raises(ValueError, match="Nonphysical"):
        audit.inspect_distribution(tmp_path / "batch.manifest.json", record, 5)
    record["distribution_sample_sha256"] = "wrong"
    with pytest.raises(ValueError, match="checksum"):
        audit.inspect_distribution(tmp_path / "batch.manifest.json", record, 10)


def test_running_case_configuration_does_not_need_final_manifest(tmp_path):
    cfg = {"initial_condition_sampling": "collision_tube_mixture", "tube_mixture_fraction": .5}
    signature = audit.digest(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode())
    (tmp_path / "configuration.json").write_text(json.dumps({"configuration_signature": signature, "configuration": cfg}))
    frozen = {"batches": [["batch_0_2.manifest.json", 0, 0]], "signature": signature}
    assert audit.signed_configuration(tmp_path, frozen)[0] == cfg
    frozen["signature"] = "wrong"
    with pytest.raises(ValueError, match="configuration mismatch"):
        audit.signed_configuration(tmp_path, frozen)
