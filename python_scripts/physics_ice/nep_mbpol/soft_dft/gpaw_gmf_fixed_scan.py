"""Fixed-multiplier DO-GMF response scan for the carbon q=1 pz state."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .gpaw_prepared_q1 import (
    PreparedQ1Settings,
    _install_orbitals,
    sha256_file,
    signature,
)

LAMBDA_HARTREE = (0.05, 0.075, 0.10, 0.125, 0.15)
REPEAT_SEEDS = (42, 314159)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".next")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_manifest(
    output_root: Path,
    retention_bundle_path: Path,
    carbon_fragment_path: Path,
    water_fragment_path: Path,
) -> Path:
    bundle_path = retention_bundle_path.resolve()
    bundle = json.loads(bundle_path.read_text())
    claimed = bundle.pop("bundle_signature")
    if signature(bundle) != claimed:
        raise RuntimeError("Retention bundle signature mismatch.")
    bundle["bundle_signature"] = claimed
    bundle["path"] = str(bundle_path)
    for key in ("gpw", "arrays"):
        if sha256_file(Path(bundle[key]["path"])) != bundle[key]["sha256"]:
            raise RuntimeError(f"Retention bundle {key} checksum mismatch.")
    fragments = {}
    for name, path in (("carbon", carbon_fragment_path), ("water", water_fragment_path)):
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        fragments[name] = {"path": str(resolved), "sha256": sha256_file(resolved)}
    settings = PreparedQ1Settings()
    tasks = []
    index = 0
    for lambda_hartree in LAMBDA_HARTREE:
        for repeat, seed in enumerate(REPEAT_SEEDS):
            identity = {
                "schema_version": 1,
                "method": "fixed_lambda_charge_only_do_gmf_full_hessian",
                "lambda_hartree": lambda_hartree,
                "repeat": repeat,
                "davidson_seed": seed,
                "settings": asdict(settings),
                "retention_bundle": bundle,
                "fragments": fragments,
                "workflow_sha256": sha256_file(Path(__file__)),
                "runner_sha256": sha256_file(
                    Path(__file__).resolve().parents[1] / "run_gpaw_gmf_fixed_scan.py"
                ),
                "pbs_sha256": sha256_file(
                    Path(__file__).resolve().parents[4] / "pbs" /
                    "run_gpaw_gmf_fixed_scan.pbs"
                ),
                "production_eligible": False,
            }
            tasks.append({
                **identity,
                "task_index": index,
                "task_signature": signature(identity),
            })
            index += 1
    unsigned = {
        "schema_version": 1,
        "task_count": len(tasks),
        "lambdas_hartree": list(LAMBDA_HARTREE),
        "repeat_seeds": list(REPEAT_SEEDS),
        "tasks": tasks,
        "production_eligible": False,
    }
    manifest = {**unsigned, "manifest_signature": signature(unsigned)}
    path = output_root.resolve() / "manifest.json"
    _atomic_json(path, manifest)
    return path


def _load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    claimed = payload.pop("manifest_signature")
    if signature(payload) != claimed:
        raise RuntimeError("Fixed-lambda scan manifest signature mismatch.")
    payload["manifest_signature"] = claimed
    return payload


def _fragment_target(task: Mapping[str, object], combined: object) -> tuple[float, float]:
    from gpaw import GPAW
    from gpaw.cdft.cdft import WeightFunc

    for artifact in task["fragments"].values():
        if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
            raise RuntimeError("Fragment checkpoint checksum mismatch.")
    carbon = GPAW(task["fragments"]["carbon"]["path"], txt=None)
    water = GPAW(task["fragments"]["water"]["path"], txt=None)
    carbon.set_positions(carbon.atoms)
    water.set_positions(water.atoms)
    carbon.atoms.calc = carbon
    water.atoms.calc = water
    expected = carbon.atoms + water.atoms
    if expected.get_chemical_symbols() != combined.atoms.get_chemical_symbols():
        raise RuntimeError("Fragment and combined atom order differ.")
    if not np.allclose(expected.cell.array, combined.atoms.cell.array, atol=1.0e-12,
                       rtol=0.0):
        raise RuntimeError("Fragment and combined cells differ.")
    if not np.allclose(expected.positions, combined.atoms.positions, atol=1.0e-12,
                       rtol=0.0):
        raise RuntimeError("Fragment and combined positions differ.")
    grid = carbon.density.finegd
    for other in (water.density.finegd, combined.density.finegd):
        for attribute in ("N_c", "n_c", "beg_c"):
            if not np.array_equal(getattr(grid, attribute), getattr(other, attribute)):
                raise RuntimeError("Fragment and combined distributed fine grids differ.")
    if carbon.density.nt_sg is None:
        carbon.density.interpolate_pseudo_density()
    if water.density.nt_sg is None:
        water.density.interpolate_pseudo_density()
    weight = WeightFunc(grid, expected, [0]).construct_weight_function()
    pseudo_density = (np.asarray(carbon.density.nt_sg).sum(axis=0)
                      + np.asarray(water.density.nt_sg).sum(axis=0))
    population = float(grid.integrate(weight * pseudo_density,
                                      global_integral=True))
    # This is GPAW 25.7.0 CDFT.get_atomic_density_correction() followed by
    # get_energy_correction(return_density=True), restricted to the carbon
    # charge region.  It is the same population expression used by CDFT.f,
    # evaluated on the sum of the independently prepared fragment densities.
    correction_s = np.zeros(2)
    for atom_index, density_matrix in carbon.density.D_asp.items():
        if atom_index != 0:
            raise RuntimeError("Carbon fragment contains an unexpected atom index.")
        setup = carbon.wfs.setups[atom_index]
        for spin in (0, 1):
            correction_s[spin] += np.sqrt(4.0 * np.pi) * (
                np.dot(density_matrix[spin], setup.Delta_pL)[0]
                + setup.Delta0 / 2.0
            )
    grid.comm.sum(correction_s)
    correction_s += carbon.atoms[0].number / 2.0
    population += float(correction_s.sum())
    return 6.0 - population, population


def _population(calc: object, cdft: object) -> tuple[float, float]:
    cdft.get_atomic_density_correction()
    correction = np.asarray(cdft.get_energy_correction(return_density=True))
    density = np.asarray(calc.density.nt_sg)
    weight = cdft.ext.w_ig[0]
    charge_population = float(
        cdft.gd.integrate(weight * (density[0] + density[1]), global_integral=True)
        + correction[0]
    )
    atomic = np.asarray(cdft.get_atomic_density_correction(return_els=True))
    spin_population = float(
        cdft.gd.integrate(weight * (density[0] - density[1]), global_integral=True)
        + atomic[0, 0] - atomic[1, 0]
    )
    return charge_population, spin_population


def _orbital_populations(calc: object, references: Mapping[str, np.ndarray]) -> dict[str, float]:
    overlap = np.asarray(calc.wfs.S_qMM[0])
    return {
        name: float(sum(
            np.sum(np.asarray(kpoint.f_n)
                   * np.abs(np.asarray(kpoint.C_nM).conj() @ overlap @ reference) ** 2)
            for kpoint in calc.wfs.kpt_u
        ))
        for name, reference in references.items()
    }


def _observer(calc: object, cdft: object, references: Mapping[str, np.ndarray],
              target: float, trace_path: Path) -> None:
    from ase.units import Hartree
    from gpaw.cdft.cdft import get_ks_energy_wo_external

    population, spin = _population(calc, cdft)
    row = {
        "scf_iteration": int(calc.scf.niter),
        "lambda_hartree": float(cdft.v_i[0]),
        "ks_energy_ev": float(get_ks_energy_wo_external(calc)),
        "external_energy_ev": float(calc.hamiltonian.e_total_extrapolated * Hartree),
        "carbon_population_e": population,
        "target_population_e": target,
        "target_residual_e": population - target,
        "carbon_spin_e": spin,
        "orbital_populations": _orbital_populations(calc, references),
        "external_active": calc.parameters.external is cdft.ext,
    }
    if calc.world.rank == 0:
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def execute(manifest_path: Path, task_index: int) -> dict[str, object]:
    from ase.units import Hartree
    from gpaw import GPAW
    from gpaw.cdft.cdft import CDFT, get_ks_energy_wo_external
    from gpaw.directmin.derivatives import Davidson
    from gpaw.directmin.etdm_lcao import LCAOETDM
    from gpaw.mpi import world

    manifest = _load_manifest(manifest_path)
    if task_index < 0 or task_index >= len(manifest["tasks"]):
        raise IndexError(task_index)
    task = manifest["tasks"][task_index]
    root = (manifest_path.parent / "tasks" /
            f"{task_index:04d}_lambda_{task['lambda_hartree']:.3f}_r{task['repeat']}")
    if world.rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    world.barrier()
    result_path = root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result.get("task_signature") == task["task_signature"]:
            return result
        raise RuntimeError("Existing fixed-lambda result has incompatible identity.")
    for key in ("gpw", "arrays"):
        artifact = task["retention_bundle"][key]
        if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
            raise RuntimeError("Retention artifact checksum mismatch.")
    combined = GPAW(task["retention_bundle"]["gpw"]["path"],
                    txt=str(root / "rehydrate.log"))
    combined.set_positions(combined.atoms)
    atoms = combined.atoms
    with np.load(task["retention_bundle"]["arrays"]["path"],
                 allow_pickle=False) as data:
        coefficients = [data["alpha_coefficients"], data["beta_coefficients"]]
        occupations = [data["alpha_occupations"], data["beta_occupations"]]
        references = {
            "p_x": data["reference_px"], "p_y": data["reference_py"],
            "p_z": data["reference_pz"], "s": data["reference_s"],
        }
    _install_orbitals(combined, atoms, coefficients, occupations,
                      use_projections=False)
    target_charge, target_population = _fragment_target(task, combined)
    lambda_hartree = float(task["lambda_hartree"])
    cdft = CDFT(
        calc=combined, atoms=atoms, charge_regions=[[0]], charges=[target_charge],
        spin_regions=[], spins=None, charge_coefs=[lambda_hartree * Hartree],
        spin_coefs=None, method="L-BFGS-B",
        minimizer_options={"gtol": 1.0e-5, "maxiter": 1, "ftol": 1.0e-12},
        tol=1.0e-10, bounds=[(-100.0, 100.0)], maxstep=100.0,
        compute_forces=False, restart=True, txt=str(root / "fixed_cdft.log"),
    )
    combined.initialize(atoms)
    combined.set_positions(atoms)
    _install_orbitals(combined, atoms, coefficients, occupations,
                      use_projections=False)
    # Fixed lambda: CDFT supplies only the external potential.  Never call
    # CDFT.calculate(), which would invoke scipy's outer optimizer.
    atoms.calc = combined
    atoms.get_potential_energy()
    preparatory_population, preparatory_spin = _population(combined, cdft)
    preparatory_pops = _orbital_populations(combined, references)
    if not combined.scf.converged or preparatory_pops["p_z"] <= 0.9999:
        raise RuntimeError("Preparatory fixed-lambda state did not retain pz.")

    seed = int(task["davidson_seed"])
    hessian = Davidson(
        combined.wfs.eigensolver, logfile=str(root / "hessian.log"),
        eps=1.0e-2, seed=seed,
    )
    saddle_order = int(hessian.estimate_sp_order(
        combined, method="full-hess", target_more=3,
    ))
    lowest_eigenvalues = [float(value) for value in hessian.eigenvalues]
    near_zero = sum(abs(value) <= 1.0e-8 for value in lowest_eigenvalues)
    coefficients = [np.asarray(kpoint.C_nM).copy()
                    for kpoint in combined.wfs.kpt_u]
    occupations = [np.asarray(kpoint.f_n).copy()
                   for kpoint in combined.wfs.kpt_u]
    combined.set(
        eigensolver=LCAOETDM(
            excited_state=True,
            searchdir_algo={"name": "l-bfgs-p_gmf"},
            linesearch_algo={"name": "max-step", "max_step": 0.20},
            partial_diagonalizer={
                "name": "Davidson", "logfile": str(root / "gmf_davidson.log"),
                "seed": seed, "sp_order": saddle_order,
                "remember_sp_order": True,
            },
            update_ref_orbs_counter=1000,
            representation="u-invar",
            need_init_orbs=False,
        ),
        occupations={
            "name": "mom", "numbers": occupations, "use_projections": False,
            "update_numbers": False, "use_fixed_occupations": True,
        },
        txt=str(root / "gmf.log"),
    )
    combined.initialize(atoms)
    combined.set_positions(atoms)
    _install_orbitals(combined, atoms, coefficients, occupations,
                      use_projections=False)
    trace_path = root / "trace.jsonl"
    combined.attach(_observer, 1, combined, cdft, references,
                    target_population, trace_path)
    atoms.calc = combined
    atoms.get_potential_energy()
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    final = rows[-1]
    minimum_pz = min(row["orbital_populations"]["p_z"] for row in rows)
    passed = bool(
        combined.scf.converged
        and all(row["external_active"] for row in rows)
        and minimum_pz > 0.9999
        and abs(final["carbon_population_e"] - target_population) < 0.1
    )
    density_path = None
    density_hash = None
    if passed:
        total = np.asarray(combined.density.nt_sg).sum(axis=0)
        collected = combined.density.finegd.collect(total, broadcast=False)
        density_path = root / "total_pseudo_density.npz"
        if world.rank == 0:
            np.savez_compressed(
                density_path,
                density=collected,
                dv_bohr3=float(combined.density.finegd.dv),
            )
            density_hash = sha256_file(density_path)
    world.barrier()
    result = {
        "schema_version": 1,
        "task_signature": task["task_signature"],
        "lambda_hartree": lambda_hartree,
        "repeat": int(task["repeat"]),
        "davidson_seed": seed,
        "fragment_defined_target_charge": target_charge,
        "fragment_defined_target_population_e": target_population,
        "preparatory_population_e": preparatory_population,
        "preparatory_spin_e": preparatory_spin,
        "preparatory_orbital_populations": preparatory_pops,
        "hessian_inertia": {
            "dimension": int(hessian.dimtot),
            "negative": saddle_order,
            "near_zero_in_computed_lowest": near_zero,
            "positive_inferred": int(hessian.dimtot - saddle_order - near_zero),
            "threshold": 1.0e-8,
        },
        "lowest_hessian_eigenvalues": lowest_eigenvalues,
        "gmf_converged": bool(combined.scf.converged),
        "gmf_iterations": len(rows),
        "minimum_pz_population": minimum_pz,
        "final": final,
        "ks_energy_ev": float(get_ks_energy_wo_external(combined)),
        "density_path": None if density_path is None else str(density_path),
        "density_sha256": density_hash,
        "fixed_lambda_gate_passed": passed,
        "accepted_soft_potential_state": False,
        "production_eligible": False,
    }
    if world.rank == 0:
        _atomic_json(root / ("result.json" if passed else "rejected_result.json"), result)
    world.barrier()
    if not passed:
        raise RuntimeError("Fixed-lambda DO-GMF response probe failed.")
    return result


def collect(manifest_path: Path) -> dict[str, object]:
    manifest = _load_manifest(manifest_path)
    pairs = []
    for lambda_hartree in manifest["lambdas_hartree"]:
        selected = [task for task in manifest["tasks"]
                    if task["lambda_hartree"] == lambda_hartree]
        results = []
        densities = []
        for task in sorted(selected, key=lambda item: item["repeat"]):
            root = (manifest_path.parent / "tasks" /
                    f"{task['task_index']:04d}_lambda_{task['lambda_hartree']:.3f}_r{task['repeat']}")
            path = root / "result.json"
            if not path.is_file():
                raise RuntimeError(f"Missing accepted fixed-lambda result: {path}")
            result = json.loads(path.read_text())
            if result["task_signature"] != task["task_signature"]:
                raise RuntimeError("Fixed-lambda result signature mismatch.")
            results.append(result)
            with np.load(result["density_path"], allow_pickle=False) as data:
                densities.append((np.asarray(data["density"]), float(data["dv_bohr3"])))
        if densities[0][0].shape != densities[1][0].shape:
            raise RuntimeError("Repeat density shapes differ.")
        density_l1 = float(np.sum(np.abs(densities[0][0] - densities[1][0]))
                           * densities[0][1])
        pairs.append({
            "lambda_hartree": lambda_hartree,
            "target_population_difference_e": abs(
                results[0]["fragment_defined_target_population_e"]
                - results[1]["fragment_defined_target_population_e"]
            ),
            "population_difference_e": abs(
                results[0]["final"]["carbon_population_e"]
                - results[1]["final"]["carbon_population_e"]
            ),
            "energy_difference_ev": abs(
                results[0]["ks_energy_ev"] - results[1]["ks_energy_ev"]
            ),
            "density_l1_e": density_l1,
            "minimum_pz": min(result["minimum_pz_population"] for result in results),
            "saddle_orders": [result["hessian_inertia"]["negative"]
                              for result in results],
            "populations_e": [result["final"]["carbon_population_e"]
                              for result in results],
            "energies_ev": [result["ks_energy_ev"] for result in results],
        })
    populations = [sum(pair["populations_e"]) / 2.0 for pair in pairs]
    monotonic = bool(all(
        (right - left) <= max(pair["population_difference_e"] for pair in pairs)
        for left, right in zip(populations, populations[1:])
    ))
    summary = {
        "schema_version": 1,
        "pairs": pairs,
        "mean_populations_e": populations,
        "response_nonincreasing_with_lambda": monotonic,
        "production_eligible": False,
    }
    _atomic_json(manifest_path.parent / "summary.json", summary)
    return summary
