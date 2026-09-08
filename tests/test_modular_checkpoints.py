"""Resume tests cover task loss, incompatible inputs and worker independence."""
import numpy as np
import pytest
from pathlib import Path
import shutil

from physics.inelastic_dielectric import checkpoints
from test_projectile_relativistic_dcs import MODULE as gen, _optical_and_dispersion


def test_source_digest_is_independent_of_snapshot_location(tmp_path, monkeypatch):
    root = Path(checkpoints.__file__).resolve().parents[1]
    expected = checkpoints.source_digest()
    snapshot = tmp_path / "output" / "runs" / "source" / "physics"
    shutil.copytree(root, snapshot, ignore=shutil.ignore_patterns(
        "output", "plots", "runs", "__pycache__", "*.pyc"))
    monkeypatch.setattr(checkpoints, "__file__", str(snapshot / "inelastic_dielectric" / "checkpoints.py"))
    assert checkpoints.source_digest() == expected


def test_atomic_failure_preserves_old_product(tmp_path, monkeypatch):
    path = tmp_path / "task.npz"
    checkpoints.atomic_savez(path, x=[1, 2])
    before = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError("simulated interrupted write")
    monkeypatch.setattr(checkpoints.os, "replace", fail)
    with pytest.raises(OSError):
        checkpoints.atomic_savez(path, x=[3, 4])
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_energy_and_manifest_reject_incompatible_resume(tmp_path):
    signature = checkpoints.require_manifest(tmp_path, {"source": "one", "dq": 24})
    checkpoints.save_energy(tmp_path, 0, 1e5, {"x": np.array([1., 2.]), "absent": None}, signature)
    assert checkpoints.load_energy(tmp_path, 0, 1e5, signature) == {"x": [1., 2.], "absent": None}
    assert checkpoints.load_energy(tmp_path, 1, 1e6, signature) is None
    with pytest.raises(ValueError, match="Incompatible"):
        checkpoints.load_energy(tmp_path, 0, 1e6, signature)
    with pytest.raises(ValueError, match="Incompatible"):
        checkpoints.require_manifest(tmp_path, {"source": "two", "dq": 24})


@pytest.mark.parametrize("relativistic", [False, True])
@pytest.mark.parametrize("barkas", [False, True])
def test_interrupted_dcs_resumes_identically(tmp_path, monkeypatch, relativistic, barkas):
    gen.set_projectile("proton")
    gen._set_projectile_relativistic_dcs(relativistic)
    s, C = _optical_and_dispersion()
    for name in ("excitation", "ionization", "kshell"):
        monkeypatch.setattr(gen, "_selected_dsigma_" + name, lambda *a: 1e-23)
    kwargs = dict(T_list=np.array([1e5, 1e7]), NE=12, Nq=24, return_data=True,
                  parallel_channels=False, include_barkas_dcs=barkas,
                  charge_mode="bare", merge_energy_patches=False)
    reference = gen.write_emfietzoglou_dcs_tables(s, C, out_dir=tmp_path / "reference", **kwargs)
    directory = tmp_path / "tasks"
    original = checkpoints.save_task
    completed = []
    def interrupt_after_two(*args):
        original(*args)
        completed.append(args[1])
        if len(completed) == 2:
            raise RuntimeError("simulated cancellation")
    monkeypatch.setattr(checkpoints, "save_task", interrupt_after_two)
    with pytest.raises(RuntimeError, match="simulated cancellation"):
        gen.write_emfietzoglou_dcs_tables(s, C, out_dir=tmp_path / "resumed",
                                         checkpoint_dir=directory, **kwargs)
    saved = {p: p.stat().st_mtime_ns for p in directory.glob("*.npz")}
    assert len(saved) == 2
    monkeypatch.setattr(checkpoints, "save_task", original)
    resumed = gen.write_emfietzoglou_dcs_tables(s, C, out_dir=tmp_path / "resumed",
                                              checkpoint_dir=directory, **kwargs)
    assert saved == {p: p.stat().st_mtime_ns for p in saved}
    for key in ("T_line", "E_line", "exc_vals", "ion_vals"):
        np.testing.assert_array_equal(resumed[key], reference[key])
    assert {p.name: p.read_bytes() for p in (tmp_path / "reference").glob("*.dat")} == {
        p.name: p.read_bytes() for p in (tmp_path / "resumed").glob("*.dat")}
    with pytest.raises(ValueError, match="Incompatible"):
        gen.write_emfietzoglou_dcs_tables(s, C, out_dir=tmp_path / "resumed",
                                         checkpoint_dir=directory, **dict(kwargs, Nq=25))
    gen._set_projectile_relativistic_dcs(False, False, use_density_effect=False)
