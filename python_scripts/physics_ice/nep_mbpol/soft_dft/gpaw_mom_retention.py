"""Fixed-multiplier controls for GPAW DO-MOM state retention."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from .geometry import ORIENTATIONS, build_scan_geometry
from .gpaw_prepared_q1 import (
    IMPLEMENTATION_VERSION,
    SCHEMA_VERSION,
    PreparedQ1Settings,
    _initialize_combined_calculator,
    _install_orbitals,
    _run_fragment,
    assemble_component_orbitals,
    sha256_file,
    signature,
)

CONTROL_MODES = ("same_process", "rehydrated_restart")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".next")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def build_manifest(output_root: Path) -> Path:
    settings = PreparedQ1Settings(
        constraint_tolerance_electrons=1.0e-5,
        maximum_multiplier_iterations=1000,
    )
    orientation = next(x for x in ORIENTATIONS if x.name == settings.orientation)
    geometry = build_scan_geometry(orientation, settings.separation_angstrom, "C")
    tasks = []
    for index, mode in enumerate(CONTROL_MODES):
        identity = {
            "schema_version": SCHEMA_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "control_mode": mode,
            "component": "p_parallel",
            "geometry": geometry.as_dict(),
            "settings": asdict(settings),
            "workflow_sha256": sha256_file(Path(__file__)),
            "runner_sha256": sha256_file(
                Path(__file__).resolve().parents[1] / "run_gpaw_mom_retention.py"
            ),
        }
        tasks.append({**identity, "task_index": index,
                      "task_signature": signature(identity)})
    unsigned = {"schema_version": 1, "task_count": 2, "tasks": tasks,
                "production_eligible": False}
    payload = {**unsigned, "manifest_signature": signature(unsigned)}
    path = output_root.resolve() / "manifest.json"
    _atomic_json(path, payload)
    return path


def _load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    claimed = payload.pop("manifest_signature")
    if signature(payload) != claimed:
        raise RuntimeError("Retention manifest signature mismatch.")
    payload["manifest_signature"] = claimed
    return payload


def _write_bundle(root: Path, combined: object, coefficients: list[np.ndarray],
                  occupations: list[np.ndarray], references: Mapping[str, np.ndarray],
                  task_signature: str) -> dict[str, object]:
    gpw = root / "prepared.gpw"
    arrays = root / "mom_state.npz"
    combined.write(str(gpw), mode="all")
    combined.world.barrier()
    bundle_path = root / "restart_bundle.json"
    if combined.world.rank == 0:
        np.savez_compressed(
            arrays,
            alpha_coefficients=coefficients[0], beta_coefficients=coefficients[1],
            alpha_occupations=occupations[0], beta_occupations=occupations[1],
            reference_pz=references["z"], reference_px=references["x"],
            reference_py=references["y"], reference_s=references["s"],
        )
        unsigned = {
            "schema_version": 1,
            "compatibility_signature": task_signature,
            "component": "p_parallel",
            "spin_resolved_occupations": True,
            "mom": {"use_projections": False, "update_numbers": False,
                    "use_fixed_occupations": True, "excited_state": True},
            "gpw": {"path": str(gpw.resolve()), "sha256": sha256_file(gpw)},
            "arrays": {"path": str(arrays.resolve()), "sha256": sha256_file(arrays)},
        }
        bundle = {**unsigned, "bundle_signature": signature(unsigned)}
        _atomic_json(bundle_path, bundle)
    combined.world.barrier()
    return json.loads(bundle_path.read_text())


def _trace_observer(calc: object, cdft: object, references: Mapping[str, np.ndarray],
                    trace: Path, baseline: dict[str, np.ndarray]) -> None:
    from ase.units import Hartree
    overlap = np.asarray(calc.wfs.S_qMM[0])
    pops = {}
    for name, ref in references.items():
        pops[name] = float(sum(np.sum(np.asarray(k.f_n) *
            np.abs(np.asarray(k.C_nM).conj() @ overlap @ ref) ** 2)
            for k in calc.wfs.kpt_u))
    cdft.get_atomic_density_correction()
    correction = np.asarray(cdft.get_energy_correction(return_density=True))
    density = np.asarray(calc.density.nt_sg)
    charge_e = float(cdft.gd.integrate(cdft.ext.w_ig[0] * density.sum(axis=0),
                                      global_integral=True) + correction[0])
    spin_e = float(cdft.gd.integrate(cdft.ext.w_ig[1] * (density[0] - density[1]),
                                    global_integral=True) + correction[1])
    total = density.sum(axis=0)
    if "density" not in baseline:
        baseline["density"] = total.copy()
    density_l1 = float(cdft.gd.integrate(abs(total - baseline["density"]),
                                        global_integral=True))
    record = {
        "scf_iteration": int(calc.scf.niter),
        "energy_ev": float(calc.hamiltonian.e_total_extrapolated * Hartree),
        "charge_residual_e": charge_e - 5.0,
        "spin_residual_e": spin_e - 1.0,
        "multiplier_ev": (np.asarray(cdft.v_i) * Hartree).tolist(),
        "orbital_populations": pops,
        "density_l1_from_first_iteration_e": density_l1,
        "external_active": calc.parameters.external is cdft.ext,
    }
    if calc.world.rank == 0:
        with trace.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def execute(manifest_path: Path, task_index: int) -> dict[str, object]:
    from ase import Atoms
    from ase.units import Hartree
    from gpaw import GPAW
    from gpaw.cdft.cdft import CDFT
    from gpaw.mpi import world

    manifest = _load_manifest(manifest_path)
    task = manifest["tasks"][task_index]
    root = manifest_path.parent / "tasks" / f"{task_index:04d}_{task['control_mode']}"
    root.mkdir(parents=True, exist_ok=True)
    settings = PreparedQ1Settings(**task["settings"])
    rows = task["geometry"]["coordinates_angstrom"]
    atoms = Atoms([r[0] for r in rows], positions=[[float(x) for x in r[1:]] for r in rows])
    atoms.center(vacuum=settings.vacuum_angstrom)
    atoms.set_initial_magnetic_moments([1.0, 0.0, 0.0, 0.0])
    carbon = _run_fragment(atoms.copy(), settings, fragment="carbon",
                           path=root / "carbon_fragment.gpw")
    water = _run_fragment(atoms.copy(), settings, fragment="water",
                          path=root / "water_fragment.gpw")
    combined = _initialize_combined_calculator(atoms, settings, root / "prepared.log")
    coefficients, occupations, references = assemble_component_orbitals(
        carbon, water, combined, "p_parallel")
    _install_orbitals(combined, atoms, coefficients, occupations,
                      use_projections=False)
    bundle = _write_bundle(root, combined, coefficients, occupations, references,
                           task["task_signature"])

    if task["control_mode"] == "rehydrated_restart":
        if sha256_file(Path(bundle["gpw"]["path"])) != bundle["gpw"]["sha256"]:
            raise RuntimeError("Bundle GPW checksum mismatch.")
        combined = GPAW(bundle["gpw"]["path"], txt=str(root / "rehydrated.log"))
        combined.set_positions(combined.atoms)
        if not bool(combined.wfs.eigensolver.excited_state):
            raise RuntimeError("Rehydrated DO eigensolver is not excited-state mode.")
        with np.load(bundle["arrays"]["path"], allow_pickle=False) as data:
            coefficients = [data["alpha_coefficients"], data["beta_coefficients"]]
            occupations = [data["alpha_occupations"], data["beta_occupations"]]
            references = {"x": data["reference_px"], "y": data["reference_py"],
                          "z": data["reference_pz"], "s": data["reference_s"]}
        _install_orbitals(combined, combined.atoms, coefficients, occupations,
                          use_projections=False)
        atoms = combined.atoms

    cdft = CDFT(calc=combined, atoms=atoms, charge_regions=[[0]], charges=[1.0],
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
    trace = root / "inner_trace.jsonl"
    combined.attach(_trace_observer, 1, combined, cdft,
                    {"p_x": references["x"], "p_y": references["y"],
                     "p_z": references["z"], "s": references["s"]}, trace, {})
    atoms.calc = combined
    atoms.get_potential_energy()  # fixed multiplier: never call CDFT.calculate()
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    final = records[-1]
    passed = bool(combined.scf.converged and final["orbital_populations"]["p_z"] > 0.9999
                  and abs(final["charge_residual_e"]) < 1e-3
                  and abs(final["spin_residual_e"]) < 1e-3
                  and all(row["external_active"] for row in records))
    result = {"schema_version": 1, "control_mode": task["control_mode"],
              "tight_inner_converged": bool(combined.scf.converged),
              "retention_gate_passed": passed, "iterations": len(records),
              "final": final, "bundle": bundle, "production_eligible": False}
    if world.rank == 0:
        _atomic_json(root / ("result.json" if passed else "rejected_result.json"), result)
    if not passed:
        raise RuntimeError("MOM retention gate failed.")
    return result
