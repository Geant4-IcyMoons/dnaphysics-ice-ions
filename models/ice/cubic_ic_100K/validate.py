"""Audit Ic dynamics and publish only snapshots passing the structural gates."""
from __future__ import annotations
import argparse
import fcntl
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from models.ice.cubic_ic_100K.run import (
    CONFIG, RUNS, atomic_bytes, cell_length, completed_block, file_sha256,
    iter_xyz_frames, prepare, timestamp, write_json,
)
from models.ice.cubic_ic_100K.statistics import hac_linear_trend, hac_mean_interval, holm_adjust
from models.ice.structure_checks import chill_plus, bernal_fowler_topology
from physics.constants import AVOGADRO, H2O_MOLAR_MASS_G_MOL

REFLECTIONS = ((1, 1, 1), (2, 2, 0), (3, 1, 1), (4, 0, 0), (3, 3, 1), (4, 2, 2), (4, 4, 0))
THERMO_NAMES = ("temperature_k", "kinetic_energy_ev", "potential_energy_ev",
               "pxx_gpa", "pyy_gpa", "pzz_gpa", "pyz_gpa", "pxz_gpa", "pxy_gpa")


def bragg(oxygen: np.ndarray, length: float) -> dict:
    fractional = oxygen / length
    replication = CONFIG["replication"][0]
    def intensity(h):
        return float(abs(np.exp(2j * np.pi * (fractional @ h)).mean()) ** 2)
    rows = []
    for reflection in REFLECTIONS:
        h = np.asarray(reflection) * replication
        center = intensity(h)
        # The nearest non-Bragg reciprocal points of the periodic supercell.
        neighbors = [intensity(h + sign * np.eye(3)[axis])
                     for axis in range(3) for sign in (-1, 1)]
        rows.append({"hkl": reflection, "normalized_peak": center,
                     "largest_neighbor": max(neighbors), "local_maximum": center > max(neighbors)})
    return {"reflections": rows, "all_local_maxima": all(r["local_maximum"] for r in rows),
            "forbidden_200_intensity": intensity(np.array([2, 0, 0]) * replication),
            "forbidden_222_intensity": intensity(np.array([2, 2, 2]) * replication)}


def structural_checks(path: Path) -> dict:
    frame, = list(iter_xyz_frames(path))
    expected_count = 24 * int(np.prod(CONFIG["replication"]))
    length = cell_length()
    if frame.species.size != expected_count or not np.all(frame.species.reshape(-1, 3) == ["O", "H", "H"]):
        raise ValueError("Wrong water count or molecular order")
    if not all(frame.pbc) or not np.allclose(frame.lattice_angstrom, np.eye(3) * length, rtol=0, atol=1e-7):
        raise ValueError("Wrong periodic cell")
    positions = frame.positions_angstrom % length
    oxygen = positions[frame.species == "O"]
    hydrogen = positions[frame.species == "H"]
    chill, neighbors, _, tree = chill_plus(oxygen, np.full(3, length))
    topology, _ = bernal_fowler_topology(oxygen, hydrogen, neighbors, tree)
    diffraction = bragg(oxygen, length)
    passed = (chill["cubic_fraction"] >= CONFIG["minimum_cubic_fraction"]
              and chill["hexagonal_count"] == 0
              and chill["four_neighbor_count"] == len(oxygen)
              and topology["passes_bernal_fowler_rules"] and diffraction["all_local_maxima"])
    return {"sha256": file_sha256(path), "passed": bool(passed), "chill": chill,
            "bernal_fowler": topology, "bragg": diffraction}


def geometric_density_sensitivity(path: Path) -> dict:
    """Test figure geometry over an illustrative 0.1% density perturbation."""
    frame, = list(iter_xyz_frames(path))
    length = cell_length()
    molecules = frame.positions_angstrom.reshape(-1, 3, 3)
    offsets = molecules - molecules[:, :1]
    offsets -= np.rint(offsets / length) * length
    rows = []
    for relative_density in (0.999, 1.0, 1.001):
        factor = relative_density ** (-1 / 3)
        side = length * factor
        scaled = (molecules[:, :1] * factor + offsets).reshape(-1, 3) % side
        oxygen, hydrogen = scaled[0::3], scaled.reshape(-1, 3, 3)[:, 1:].reshape(-1, 3)
        chill, neighbors, _, tree = chill_plus(oxygen, np.full(3, side))
        topology, _ = bernal_fowler_topology(oxygen, hydrogen, neighbors, tree)
        crop_count = int(np.all(np.abs(oxygen - side / 2) < 7.0, axis=1).sum())
        rows.append({"relative_density": relative_density, "relative_length": factor,
                     "cubic_fraction": chill["cubic_fraction"],
                     "ice_rules_pass": topology["passes_bernal_fowler_rules"],
                     "nearest_oo_distances_angstrom": chill["four_nearest_oo_distance_a"],
                     "waters_in_14_angstrom_crop": crop_count})
    return {"scenarios": rows,
            "basis": "Illustrative +/-0.1% density sensitivity, motivated by sub-0.1% D2O polytype differences reported in arXiv:2602.13053v1; not an uncertainty bound for H2O.",
            "scope": "Geometric perturbations only. No relaxation, force, stress, or equilibrium-density prediction is made."}


def validate(seed: int) -> dict:
    run = prepare(seed)
    preparation = json.loads((run / "preparation.json").read_text())
    completion = json.loads((run / "dynamics_completed.json").read_text())
    if preparation["signature"] != completion["signature"]:
        raise ValueError("Dynamics signature mismatch")
    count = CONFIG["equilibration_blocks"] + CONFIG["sampling_blocks"]
    source = run / "initial.xyz"
    sampling, frames, records = [], [], []
    for index in tqdm(range(count), desc=f"Validate seed {seed}", unit="block"):
        block = run / f"block{index:03d}"
        receipt = completed_block(block, preparation["signature"], file_sha256(source))
        if receipt is None:
            raise ValueError(f"Missing committed block {index}")
        records.append({"block": index, "receipt_sha256": file_sha256(block / "completed.json")})
        source = block / "dump.xyz"
        if index >= CONFIG["equilibration_blocks"]:
            sampling.append(np.loadtxt(block / "thermo.out", ndmin=2))
            frames.append(structural_checks(source))
    if file_sha256(source) != completion["final_sha256"]:
        raise ValueError("Final dynamics checksum mismatch")
    values = np.concatenate(sampling)
    interval_ns = CONFIG["thermo_interval_steps"] * CONFIG["timestep_fs"] / 1e6
    trends = {THERMO_NAMES[column]: hac_linear_trend(values[:, column], interval_ns)
              for column in (0, 2, 3, 4, 5, 6, 7, 8)}
    adjusted = holm_adjust(t["raw_two_sided_p_value"] for t in trends.values())
    alpha = CONFIG["familywise_alpha"] / len(CONFIG["seeds"])
    for trend, pvalue in zip(trends.values(), adjusted):
        trend["holm_adjusted_p_value"] = pvalue
    temperature = hac_mean_interval(values[:, 0], alpha)
    lo, hi = temperature["confidence_interval"]
    gates = {"all_sampling_structures_pass": all(frame["passed"] for frame in frames),
             "target_temperature_in_mean_interval": lo <= CONFIG["temperature_k"] <= hi,
             "no_resolved_drift": all(p > alpha for p in adjusted)}
    report = {"seed": seed, "created_utc": timestamp(), "config": CONFIG,
              "validator_sha256": file_sha256(Path(__file__)),
              "statistics_sha256": file_sha256(HERE / "statistics.py"),
              "signature": preparation["signature"], "passed": all(gates.values()), "gates": gates,
              "temperature": temperature, "trends": trends, "frames": frames,
              "blocks": records, "sampling_records": len(values),
              "sampling_column_means": dict(zip(THERMO_NAMES, map(float, values[:, :9].mean(axis=0)))),
              "block_means": [dict(zip(THERMO_NAMES, map(float, v[:, :9].mean(axis=0)))) for v in sampling],
              "normal_stress_gpa": float(values[:, 3:6].mean()),
              "geometric_density_sensitivity": geometric_density_sensitivity(source),
              "limits": ["Density is the accepted Ih-volume approximation, not an Ic measurement.",
                         "Classical nuclear dynamics; nuclear quantum effects are omitted.",
                         "No significant drift is not proof of full equilibrium.",
                         "Structural acceptance does not validate NEP forces or ion interactions."]}
    write_json(HERE / "validation" / f"seed{seed}.json", report)
    if not report["passed"]:
        raise RuntimeError(f"Seed {seed} failed validation: {gates}")
    snapshot = HERE / f"seed{seed}_final.xyz.gz"
    atomic_bytes(snapshot, gzip.compress(source.read_bytes(), mtime=0))
    write_json(snapshot.with_suffix(".json"), {
        "phase": CONFIG["phase"], "state": "structurally validated classical 100 K snapshot; density proxy",
        "collision_ready": False, "sha256": file_sha256(snapshot),
        "density_role": CONFIG["density_role"],
        "validation_report": f"validation/seed{seed}.json",
        "validation_sha256": file_sha256(HERE / "validation" / f"seed{seed}.json")})
    return report


def collect() -> None:
    reports = []
    preparations = []
    structures = []
    orientations = []
    for seed in CONFIG["seeds"]:
        path = HERE / "validation" / f"seed{seed}.json"
        report = json.loads(path.read_text())
        if not report["passed"] or report["config"] != CONFIG:
            raise ValueError(f"Unaccepted or incompatible seed {seed}")
        preparation = json.loads((RUNS / f"seed{seed}" / "preparation.json").read_text())
        if report["signature"] != preparation["signature"]:
            raise ValueError("Validation signature mismatch")
        orientations.append(preparation["bernal_fowler"]["orientation_sha256"])
        retained = HERE / "preparation" / f"seed{seed}.json"
        write_json(retained, preparation)
        preparations.append({"path": str(retained.relative_to(HERE)), "sha256": file_sha256(retained)})
        snapshot = HERE / f"seed{seed}_final.xyz.gz"
        metadata = json.loads(snapshot.with_suffix(".json").read_text())
        if file_sha256(snapshot) != metadata["sha256"] or metadata["validation_sha256"] != file_sha256(path):
            raise ValueError("Published snapshot or report changed")
        frame, = list(iter_xyz_frames(snapshot))
        structures.append({"path": snapshot.name, "sha256": file_sha256(snapshot),
                           "phase": CONFIG["phase"], "temperature_k": CONFIG["temperature_k"],
                           "water_molecules": int(len(frame.species) // 3),
                           "lattice_angstrom": frame.lattice_angstrom.tolist(),
                           "density_g_cm3": len(frame.species) / 3 * H2O_MOLAR_MASS_G_MOL / AVOGADRO / (abs(np.linalg.det(frame.lattice_angstrom)) * 1e-24),
                           "collision_ready": False})
        reports.append({"path": str(path.relative_to(HERE)), "sha256": file_sha256(path)})
    if len(set(orientations)) != len(CONFIG["seeds"]):
        raise ValueError("Proton configurations are not distinct")
    write_json(HERE / "manifest.json", {
        "status": "structurally_validated_density_proxy", "created_utc": timestamp(),
        "config": CONFIG, "structures": structures, "reports": reports, "preparations": preparations,
        "distinct_proton_configurations": True,
        "collision_ready": False,
        "integration": "Available for visualization; cubic ion-collision phase is not configured."})
    from models.ice.plot_structure import plot_ice_structure_comparison
    plot_ice_structure_comparison(
        HERE.parent / "hexagonal_ih_100K_experimental/seed1000_final.xyz.gz",
        HERE.parent / "epsr_lda80k/artifacts/lda80k_epsr.xyz.gz",
        HERE.parent / "ice_structure_comparison.png",
        cubic_path=HERE / "seed1000_final.xyz.gz")
    registry_path = HERE.parent / "registry.json"
    with (RUNS / "collection.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = json.loads(registry_path.read_text())
        registry["models"]["cubic_ic_100k"] = {
            "status": "structurally_validated_density_proxy",
            "temperature_k": CONFIG["temperature_k"], "configuration_count": len(structures),
            "manifest": "cubic_ic_100K/manifest.json",
            "density_g_cm3": structures[0]["density_g_cm3"],
            "density_role": CONFIG["density_role"], "collision_ready": False}
        write_json(registry_path, registry)
    campaign_path = RUNS / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    previous = campaign["previous_comparison"]
    legacy = HERE.parents[2] / previous["path"]
    canonical = HERE.parent / "ice_structure_comparison.png"
    if legacy.is_symlink() and legacy.resolve() == canonical.resolve():
        pass
    elif file_sha256(legacy) == previous["sha256"]:
        # Replace only the inspected old figure, preserving a concurrently edited one.
        temporary = legacy.with_name(legacy.name + ".link-tmp")
        temporary.symlink_to(canonical)
        temporary.replace(legacy)
    else:
        campaign["legacy_plot_note"] = "Old figure changed concurrently; preserved it. New comparison is under models/ice."
    campaign["state_at_record"] = "All three configurations validated; comparison generated"
    campaign["completed_utc"] = timestamp()
    write_json(campaign_path, campaign)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect()
    elif args.seed is not None:
        validate(args.seed)
    else:
        parser.error("Specify --seed or --collect")
