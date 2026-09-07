#!/usr/bin/env python3
"""Reproduce the charge-state density library; PySCF is a build-only dependency.

No empirical effective charge, target property, or stopping curve enters an
atomic calculation. The orbital space contains the occupied angular momenta
(s,p for the requested elements); the radial basis is fully uncontracted.
Open-shell states use high-spin ROHF, followed by a spherical density average.
See PROJECTILE_FORM_FACTORS.md for the approximation and benchmark protocol.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
from scipy.special import gamma

sys.path.insert(0, str(Path(__file__).resolve().parent))
from projectile_form_factors import DATA_PATH, DEFAULT_WORKERS, ELEMENTS, VERSION, ProjectileDensity


def configuration(electrons):
    remaining = electrons
    occupied = []
    spin = 0
    for label, capacity, l in (("1s", 2, 0), ("2s", 2, 0), ("2p", 6, 1), ("3s", 2, 0), ("3p", 6, 1)):
        count = min(remaining, capacity)
        if count:
            occupied.append((label, count, l))
            spin += min(count, capacity-count)
        remaining -= count
    if remaining:
        raise ValueError("Ground-state configuration outside H--S model scope.")
    return occupied, spin


def gaussian_density_terms(mol, density_matrix):
    """Expand the angular average of the AO density into radial Gaussians.

    For normalized real Y_lm, angular integration leaves only equal (l,m).
    libcint coefficients normalize the radial part with integral R^2*r^2=1.
    Each stored weight is the integral of one signed density term, so its
    Fourier transform can be evaluated analytically without spatial grids.
    """
    dm = np.asarray(density_matrix)
    if dm.ndim == 3:
        dm = dm.sum(axis=0)
    loc = mol.ao_loc_nr()
    terms = {}
    for i in range(mol.nbas):
        l = mol.bas_angular(i)
        for j in range(i, mol.nbas):
            if mol.bas_angular(j) != l:
                continue
            block = dm[loc[i]:loc[i+1], loc[j]:loc[j+1]].reshape(mol.bas_nctr(i), 2*l+1, mol.bas_nctr(j), 2*l+1)
            radial = mol._libcint_ctr_coeff(i) @ np.einsum("ambm->ab", block) @ mol._libcint_ctr_coeff(j).T
            if i != j:
                radial *= 2
            exponents = mol.bas_exp(i)[:, None] + mol.bas_exp(j)[None, :]
            weights = radial * gamma(l+1.5)/(2*exponents**(l+1.5))
            for a, weight in zip(exponents.flat, weights.flat):
                key = (l, float(a))
                terms[key] = terms.get(key, 0.0) + float(weight)
    return [[l, a, weight] for (l, a), weight in sorted(terms.items()) if weight != 0.0]


def calculate_state(task):
    from pyscf import __version__ as pyscf_version, gto, lib, scf
    element, charge, level = task
    z = ELEMENTS[element]
    n = z-charge
    occupied, spin = configuration(n)
    core_basis = f"cc-pV{level}Z" if element == "He" else f"cc-pCV{level}Z"
    diffuse_basis = f"aug-cc-pV{level}Z"
    basis_name = f"{core_basis}+{diffuse_basis}"
    max_l = max(l for _, _, l in occupied)
    # Retain only the occupied angular momenta, as in an atomic orbital HF
    # expansion. No neutral-atom contraction or effective core potential.
    primitives = gto.basis.load(core_basis, element) + gto.basis.load(diffuse_basis, element)
    # uncontract also removes duplicate primitive exponents in this union.
    basis = [shell for shell in gto.uncontract(primitives) if shell[0] <= max_l]
    lib.num_threads(1)
    mol = gto.M(atom=f"{element} 0 0 0", basis={element: basis}, charge=charge, spin=spin, symmetry=True, verbose=0)
    mf = scf.ROHF(mol) if spin else scf.RHF(mol)
    mf.chkfile = None
    mf.max_memory = 800
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-7
    mf.max_cycle = 150
    mf.init_guess = "1e"
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"Atomic SCF did not converge: {element}:{charge} {basis_name}.")
    dm = mf.make_rdm1()
    dm_total = np.asarray(dm).sum(axis=0) if np.asarray(dm).ndim == 3 else dm
    count = float(np.einsum("ij,ji", dm_total, mol.intor("int1e_ovlp")))
    if abs(count-n) > 1e-8:
        raise RuntimeError(f"Atomic electron count failed for {element}:{charge}: {count}.")
    record = {
        "method": "all-electron-ROHF-spherical-monopole",
        "configuration": " ".join(f"{label}{count}" for label, count, _ in occupied),
        "spin_2S": spin,
        "basis": f"uncontracted-{basis_name}-occupied-angular-momenta",
        "basis_sha256": hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest(),
        "pyscf_version": pyscf_version,
        "scf_settings": {"energy_tolerance_hartree": mf.conv_tol, "gradient_tolerance": mf.conv_tol_grad,
                         "max_cycle": mf.max_cycle, "initial_guess": mf.init_guess, "symmetry": "SO3"},
        "energy_hartree": float(energy),
        "ao_electron_count": count,
        "gaussian_terms": gaussian_density_terms(mol, dm),
    }
    density = ProjectileDensity(element, charge, record)
    radial = np.geomspace(1e-7, 80, 2001)
    if abs(density.moment(0)-n) > 1e-8 or np.min(density.density(radial)) < -1e-11:
        raise RuntimeError(f"Invalid exported radial density for {element}:{charge}.")
    # Independently evaluate AO densities on a Lebedev sphere to check the
    # radial export, including libcint normalization and open-shell averaging.
    from pyscf.dft import gen_grid, numint
    angular = gen_grid.MakeAngularGrid(50)
    radii = np.geomspace(1e-4, 20, 80)
    xyz = (radii[:, None, None]*angular[None, :, :3]).reshape(-1, 3)
    ao = numint.eval_ao(mol, xyz)
    values = numint.eval_rho(mol, ao, dm_total).reshape(len(radii), -1) @ angular[:, 3]
    error = float(np.max(np.abs(values-density.density(radii)))/np.max(values))
    if error > 1e-10:
        raise RuntimeError(f"AO-to-radial density export failed for {element}:{charge}: {error}.")
    record["ao_radial_export_max_scaled_error"] = error
    return f"{element}:{charge}", record


def build_library(workers, fine="5", coarse="Q"):
    states = [(symbol, q) for symbol, z in ELEMENTS.items() for q in range(z-1)]
    tasks = [(symbol, q, level) for symbol, q in states for level in (coarse, fine)]
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for task, (key, record) in zip(tasks, pool.map(calculate_state, tasks)):
            results[(key, task[2])] = record
            print(f"{key:>5} {task[2]}Z E={record['energy_hartree']:.10f} Ha", flush=True)
    records = {}
    k = np.geomspace(1e-4, 1000, 1001)
    for symbol, q in states:
        key = f"{symbol}:{q}"
        low = ProjectileDensity(symbol, q, results[(key, coarse)])
        high = ProjectileDensity(symbol, q, results[(key, fine)])
        f_error = float(np.max(np.abs(high.form_factor(k, direct=True)-low.form_factor(k, direct=True)))/high.electrons)
        amp_error = float(np.max(np.abs(high.charge_amplitude(k)/low.charge_amplitude(k)-1)))
        record = dict(high.record)
        square_error = float(np.max(np.abs(high.squared_charge(k)/low.squared_charge(k)-1)))
        if square_error > 0.005:
            raise RuntimeError(f"Basis convergence exceeds 0.5% in squared charge for {key}: {square_error}.")
        record["basis_comparison"] = {
            "coarse_basis": low.record["basis"],
            "coarse_energy_hartree": low.record["energy_hartree"],
            "max_form_factor_difference_per_electron": f_error,
            "max_charge_amplitude_relative_difference": amp_error,
            "max_squared_charge_relative_difference": square_error,
            "wave_number_range_bohr_inverse": [float(k[0]), float(k[-1])],
            "wave_number_points": len(k),
            "r2_relative_difference": high.moment(2)/low.moment(2)-1,
        }
        records[key] = record
    return {
        "version": VERSION,
        "description": "Computed nonrelativistic fixed-ground-configuration ROHF spherical densities; not digitized Clementi-Roetti coefficients.",
        "references": ["https://doi.org/10.1063/5.0006074", "https://doi.org/10.1016/S0092-640X(74)80016-1"],
        "units": {"energy": "hartree", "radius": "bohr", "gaussian_exponent": "bohr^-2", "gaussian_weight": "electrons"},
        "states": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    data = build_library(args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(f"Wrote {len(data['states'])} multielectron states to {args.output}; one-electron and bare states are analytic.")


if __name__ == "__main__":
    main()
