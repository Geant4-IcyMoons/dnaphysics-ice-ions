"""Bounded DO-GMF anchor for the carbon q=1 pz state at 12 angstrom."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .gpaw_mom_retention import _trace_observer
from .gpaw_prepared_q1 import (
    PreparedQ1Settings,
    _install_orbitals,
    sha256_file,
    signature,
)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".next")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def build_manifest(output_root: Path, retention_bundle_path: Path) -> Path:
    retention_bundle_path = retention_bundle_path.resolve()
    bundle = json.loads(retention_bundle_path.read_text())
    claimed = bundle.pop("bundle_signature")
    if signature(bundle) != claimed:
        raise RuntimeError("Retention bundle signature mismatch.")
    bundle["bundle_signature"] = claimed
    result_path = retention_bundle_path.with_name("result.json")
    result = json.loads(result_path.read_text())
    if not (result.get("tight_inner_converged") and result.get("retention_gate_passed")):
        raise RuntimeError("DO-GMF requires a passed retention bundle.")
    unsigned = {
        "schema_version": 1,
        "method": "do_gmf_fixed_occupations_full_hessian_order",
        "component": "p_parallel",
        "charge": 1,
        "separation_angstrom": 12.0,
        "orientation": "oxygen_back",
        "fixed_charge_multiplier_hartree": 0.1,
        "fixed_spin_multiplier_hartree": 0.1,
        "retention_bundle": bundle,
        "retention_result": {"path": str(result_path), "sha256": sha256_file(result_path)},
        "workflow_sha256": sha256_file(Path(__file__)),
        "production_eligible": False,
    }
    manifest = {**unsigned, "manifest_signature": signature(unsigned)}
    path = output_root.resolve() / "manifest.json"
    _atomic_json(path, manifest)
    return path


def _load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text())
    claimed = manifest.pop("manifest_signature")
    if signature(manifest) != claimed:
        raise RuntimeError("DO-GMF manifest signature mismatch.")
    manifest["manifest_signature"] = claimed
    for key in ("gpw", "arrays"):
        item = manifest["retention_bundle"][key]
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("DO-GMF retention artifact checksum mismatch.")
    return manifest


def execute(manifest_path: Path) -> dict[str, object]:
    from ase.units import Hartree
    from gpaw import GPAW
    from gpaw.cdft.cdft import CDFT
    from gpaw.directmin.derivatives import Davidson
    from gpaw.directmin.etdm_lcao import LCAOETDM
    from gpaw.mpi import world

    manifest = _load_manifest(manifest_path)
    root = manifest_path.parent / "anchor"
    root.mkdir(parents=True, exist_ok=True)
    bundle = manifest["retention_bundle"]
    combined = GPAW(bundle["gpw"]["path"], txt=str(root / "mom_rehydrate.log"))
    combined.set_positions(combined.atoms)
    atoms = combined.atoms
    with np.load(bundle["arrays"]["path"], allow_pickle=False) as data:
        coefficients = [data["alpha_coefficients"], data["beta_coefficients"]]
        occupations = [data["alpha_occupations"], data["beta_occupations"]]
        references = {"p_x": data["reference_px"], "p_y": data["reference_py"],
                      "p_z": data["reference_pz"], "s": data["reference_s"]}
    _install_orbitals(combined, atoms, coefficients, occupations,
                      use_projections=False)

    cdft = CDFT(
        calc=combined, atoms=atoms, charge_regions=[[0]], charges=[1.0],
        spin_regions=[[0]], spins=[1.0],
        charge_coefs=[0.1 * Hartree], spin_coefs=[0.1 * Hartree],
        method="L-BFGS-B", minimizer_options={"gtol": 1e-5, "maxiter": 1000,
                                               "ftol": 1e-12},
        tol=1e-10, bounds=[(-100, 100), (-100, 100)], maxstep=100,
        compute_forces=False, restart=True, txt=str(root / "fixed_cdft.log"))
    combined.initialize(atoms)
    combined.set_positions(atoms)
    _install_orbitals(combined, atoms, coefficients, occupations,
                      use_projections=False)
    atoms.calc = combined
    atoms.get_potential_energy()
    if not combined.scf.converged:
        raise RuntimeError("Preparatory fixed-multiplier MOM solve did not converge.")

    davidson = Davidson(combined.wfs.eigensolver, logfile=str(root / "saddle_order.log"),
                        eps=1e-2, seed=42)
    saddle_order = int(davidson.estimate_sp_order(
        combined, method="full-hess", target_more=3))
    if saddle_order < 1:
        raise RuntimeError("Full-Hessian stability analysis found no saddle direction.")

    coefficients = [np.asarray(k.C_nM).copy() for k in combined.wfs.kpt_u]
    occupations = [np.asarray(k.f_n).copy() for k in combined.wfs.kpt_u]
    combined.set(
        eigensolver=LCAOETDM(
            excited_state=True,
            searchdir_algo={"name": "l-bfgs-p_gmf"},
            linesearch_algo={"name": "max-step", "max_step": 0.20},
            partial_diagonalizer={"name": "Davidson", "logfile": str(root / "gmf_davidson.log"),
                                  "seed": 42, "sp_order": saddle_order,
                                  "remember_sp_order": True},
            update_ref_orbs_counter=1000,
            representation="u-invar",
            need_init_orbs=False,
        ),
        occupations={"name": "mom", "numbers": occupations,
                     "use_projections": False, "update_numbers": False,
                     "use_fixed_occupations": True},
        txt=str(root / "gmf.log"),
    )
    combined.initialize(atoms)
    combined.set_positions(atoms)
    _install_orbitals(combined, atoms, coefficients, occupations,
                      use_projections=False)
    trace = root / "gmf_trace.jsonl"
    combined.attach(_trace_observer, 1, combined, cdft, references, trace, {})
    atoms.calc = combined
    atoms.get_potential_energy()
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    final = rows[-1]
    passed = bool(
        combined.scf.converged
        and all(row["external_active"] for row in rows)
        and min(row["orbital_populations"]["p_z"] for row in rows) > 0.9999
        and abs(final["charge_residual_e"]) < 1e-3
        and abs(final["spin_residual_e"]) < 1e-3
    )
    final_gpw = root / "gmf_final.gpw"
    combined.write(str(final_gpw), mode="all")
    result = {
        "schema_version": 1,
        "method": manifest["method"],
        "saddle_order": saddle_order,
        "gmf_converged": bool(combined.scf.converged),
        "gmf_anchor_gate_passed": passed,
        "iterations": len(rows),
        "minimum_pz_population": min(row["orbital_populations"]["p_z"] for row in rows),
        "final": final,
        "final_gpw_sha256": sha256_file(final_gpw),
        "production_eligible": False,
    }
    if world.rank == 0:
        _atomic_json(root / ("result.json" if passed else "rejected_result.json"), result)
    if not passed:
        raise RuntimeError("DO-GMF anchor retention gate failed.")
    return result
