#!/usr/bin/env python3
"""One restartable NLH production protocol: pair DCS, then ice response.

Deterministic binary cross sections and sampled finite-path ice responses have
separate acceptance and consumption contracts. Neither is silently relabelled
as a homogeneous, history-independent collision law for structured ice.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import importlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tarfile
import time

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy import sparse
from scipy.special import logsumexp
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PREFIX = "python_scripts/physics_ice/nep_mbpol"
SCHEMA = 7
OBSERVABLES = ("hard_collision_rate_per_angstrom",
               "hard_nuclear_stopping_ev_per_angstrom",
               "hard_transport_rate_per_angstrom",
               "mean_recoil_energy_ev_per_collision",
               "mean_one_minus_cosine_per_collision")
RATIOS = ((1, 0), (2, 0), (3, 0), (2, 1), (3, 1))
_CACHE = {}
_STOP = False


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        f.write(canonical(value) + "\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class Policy:
    # Computational limits are not physical accuracy requirements.
    max_histories: int = 10_000_000
    max_case_histories: int = 524_288
    max_cases: int = 512
    batch_size: int = 128
    training_histories: int = 2048
    initial_validation_histories: int = 2048
    max_proposal_rounds: int = 3
    max_energy_depth: int = 4
    max_bin_level: int = 5
    base_bins: int = 16
    max_collisions: int = 10_000
    max_seconds: int = 68_400
    confidence: float = 0.95
    statistical_relative_tolerance: float = 0.05
    scalar_precision_mode: str = "relative_standard_error"
    output_scope: str = "joint_distribution"
    scalar_interpolation_tolerance: float = 0.05
    sampling_growth: float = 0.2
    distribution_tv_tolerance: float = 0.005
    pair_distribution_tv_tolerance: float = 0.005
    betting_grid_size: int = 20
    tail_score_fractions: tuple = (0., 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.)
    grid_cdf_tolerance: float = 0.005
    interpolation_tv_tolerance: float = 0.005
    interpolation_relative_tolerance: float = 0.005
    kernel_relative_tolerance: float = 0.005
    defensive_fraction: float = 0.2
    local_centres: int = 8
    proposal_scale_powers: tuple = (2, 5, 8, 11, 14, 17, 20)
    pilot_first: bool = True
    kernel_quadrature_orders: tuple = (192, 384, 768)
    pair_max_points: int = 65_536
    pair_max_refinements: int = 16
    pair_bin_count: int = 32
    pair_max_products: int = 4096
    pair_energy_depth: int = 4

    def validate(self):
        if self.output_scope not in ("scalar", "joint_distribution"):
            raise ValueError("Unknown output scope")
        if self.output_scope == "scalar" and self.scalar_precision_mode != "relative_standard_error":
            raise ValueError("Scalar production requires estimated standard errors")
        if not 0 < self.scalar_interpolation_tolerance < 1 or not 0 < self.sampling_growth <= 1:
            raise ValueError("Invalid scalar interpolation tolerance or sampling growth")
        if self.scalar_precision_mode not in ("relative_standard_error", "simultaneous_bound"):
            raise ValueError("Unknown scalar precision mode")
        integers = ("max_histories", "max_case_histories", "max_cases", "batch_size",
                    "training_histories", "initial_validation_histories", "max_proposal_rounds",
                    "max_energy_depth", "max_bin_level", "base_bins", "max_collisions",
                    "max_seconds", "local_centres", "pair_max_points", "pair_max_refinements",
                    "pair_bin_count", "pair_max_products", "pair_energy_depth", "betting_grid_size")
        for name in integers:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("confidence", "statistical_relative_tolerance", "distribution_tv_tolerance",
                     "pair_distribution_tv_tolerance",
                     "grid_cdf_tolerance", "interpolation_tv_tolerance", "interpolation_relative_tolerance",
                     "kernel_relative_tolerance", "defensive_fraction"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie in (0, 1)")
        if min(self.training_histories, self.initial_validation_histories) < 2 * self.batch_size:
            raise ValueError("Training and validation each need at least two batches")
        orders = self.kernel_quadrature_orders
        if len(orders) < 2 or any(type(n) is not int or n < 32 for n in orders) or sorted(set(orders)) != list(orders):
            raise ValueError("Quadrature orders must be increasing integers >= 32")
        if type(self.pilot_first) is not bool:
            raise ValueError("pilot_first must be boolean")
        if not self.proposal_scale_powers or any(type(n) is not int or not 1 <= n <= 24 for n in self.proposal_scale_powers):
            raise ValueError("Proposal scale powers must be integers between 1 and 24")
        if self.initial_validation_histories > self.max_case_histories:
            raise ValueError("Initial validation exceeds the case budget")
        edges = np.asarray(self.tail_score_fractions, float)
        if (edges.ndim != 1 or not 2 <= len(edges) <= 17 or not np.isfinite(edges).all()
                or edges[0] != 0 or edges[-1] != 1 or np.any(np.diff(edges) <= 0)):
            raise ValueError("Tail score fractions must partition [0, 1] with at most 16 bands")


def runtime_path(root):
    return Path(root) / "runtime" / PREFIX


def load_runtime(root):
    directory = runtime_path(root).resolve()
    for name in ("bca", "nlh", "ion_ice"):
        if name in sys.modules:
            source = Path(sys.modules[name].__file__).resolve()
            if directory not in source.parents:
                raise RuntimeError("A different NLH runtime is already imported; use a fresh process")
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    from bca.runtime import AdaptiveKernelTable
    from bca.structure import load_ice_structure
    from bca.trajectory import PeriodicHardCollisionTransport
    return AdaptiveKernelTable, load_ice_structure, PeriodicHardCollisionTransport


def verify(root):
    root = Path(root)
    manifest = read(root / "manifest.json")
    signature = manifest.pop("signature")
    if digest(manifest) != signature or manifest["schema"] != SCHEMA:
        raise ValueError("Campaign signature/schema mismatch")
    manifest["signature"] = signature
    if file_hash(Path(__file__)) != manifest["controller_sha256"]:
        raise ValueError("Controller changed: do not mix evidence from different implementations")
    for name, expected in manifest.get("implementation_modules", {}).items():
        if Path(name).name != name or file_hash(HERE / name) != expected:
            raise ValueError("Signed private implementation changed")
    if manifest["environment"] != {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}:
        raise ValueError("Python/NumPy/SciPy environment changed")
    for name, expected in manifest["files"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if file_hash(path) != expected:
            raise ValueError(f"Changed signed input: {path}")
    if "tail_support" not in manifest:
        raise ValueError("Missing signed physical tail support")
    return manifest


def prepare(root, spec_path, projectile=None, material=None, reuse_pairs=None):
    root, spec_path = Path(root).resolve(), Path(spec_path).resolve()
    if root.exists():
        raise ValueError("Prepare requires a new campaign directory; run resumes existing campaigns")
    spec = read(spec_path)
    if projectile is not None and projectile != spec.get("projectile"):
        spec["projectile"] = projectile
        spec.pop("historical_diagnostic", None)
        spec.pop("saved_replays", None)
    if material is not None:
        spec["structures"] = [s for s in spec["structures"] if s.get("material") == material]
        if not spec["structures"]:
            raise ValueError("The requested material has no declared, attested input")
    policy = Policy(**spec.get("policy", {}))
    policy.validate()
    required = {"runtime_commit", "kernel_manifest", "structures", "energies_ev", "directions",
                "projectile", "path_length_angstrom", "seed"}
    if not required.issubset(spec):
        raise ValueError(f"Missing specification fields: {sorted(required - spec.keys())}")
    energies = sorted(set(float(e) for e in spec["energies_ev"]))
    if len(energies) < 2 or not all(math.isfinite(e) and e > 0 for e in energies):
        raise ValueError("At least two finite positive total energies are required")
    if not math.isfinite(spec["path_length_angstrom"]) or spec["path_length_angstrom"] <= 0:
        raise ValueError("The declared path length must be positive")
    if type(spec["seed"]) is not int or spec["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    names = [d["name"] for d in spec["directions"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Directions require unique names")
    for direction in spec["directions"]:
        if direction["vector"] is not None:
            v = np.asarray(direction["vector"], float)
            if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
                raise ValueError("Invalid direction vector")
            direction["vector"] = (v / np.linalg.norm(v)).tolist()
    if len(spec["structures"]) * len(names) * len(energies) > policy.max_cases:
        raise ValueError("Base matrix exceeds max_cases")
    def resolve(name):
        p = Path(name).expanduser()
        return (spec_path.parent / p).resolve() if not p.is_absolute() else p.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "--verify", spec["runtime_commit"] + "^{commit}"],
                                     cwd=REPO, text=True).strip()
    root.mkdir(parents=True)
    try:
        archive = subprocess.check_output(["git", "archive", commit, PREFIX + "/bca", PREFIX + "/nlh",
                                           PREFIX + "/ion_ice", PREFIX + "/simulate_nlh_hard_collisions.py"], cwd=REPO)
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(root / "runtime", filter="data")
        Table, load_structure, _ = load_runtime(root)
        table = Table(resolve(spec["kernel_manifest"]))
        kernel_metadata = read(table.manifest_path)
        supported = kernel_metadata["configuration"]["projectiles"]
        if spec["projectile"] not in supported:
            raise ValueError(f"Missing retained NLH kernels for {spec['projectile']}; available projectiles: {', '.join(supported)}")
        files = {str(p.relative_to(root)): file_hash(p) for p in (root / "runtime").rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts}
        for p in (table.manifest_path, table.csv_path):
            files[str(p)] = file_hash(p)
        records = []
        for entry in spec["structures"]:
            path = resolve(entry["path"])
            metadata = resolve(entry["metadata"]) if entry.get("metadata") else path.with_suffix(".json")
            structure = load_structure(path, metadata_path=metadata, frame_index=entry.get("frame_index", -1))
            record = structure.manifest_record()
            record["metadata"] = str(metadata)
            record["name"] = entry["name"]
            record["material"] = entry.get("material", "specified_snapshot")
            if record["material"] in ("hexagonal", "amorphous") and record["material"] not in record["phase"].lower():
                raise ValueError("Requested material label disagrees with the structure attestation")
            record["directions"] = entry.get("directions", names)
            if not record["directions"] or len(set(record["directions"])) != len(record["directions"]) or not set(record["directions"]).issubset(names):
                raise ValueError("Each structure requires a nonempty, unique subset of the declared directions")
            records.append(record)
            for p in (path, metadata):
                files[str(p)] = file_hash(p)
            for evidence in read(metadata)["validation_reports"]:
                p = Path(evidence["path"])
                p = p if p.is_absolute() else metadata.parent / p
                files[str(p.resolve())] = file_hash(p)
        if len({r["name"] for r in records}) != len(records) or not records:
            raise ValueError("Structure names must be nonempty and unique")
        for target in ("H", "O"):
            low, high = table.energy_bounds_ev(spec["projectile"], target)
            if energies[0] < low or energies[-1] > high:
                raise ValueError("Requested energy range is outside the retained kernel")
        seeds = []
        replay_request = None
        planning = []
        replay_paths = [resolve(filename) for filename in spec.get("saved_replays", [])]
        if spec.get("historical_diagnostic"):
            diagnostic = resolve(spec["historical_diagnostic"])
            replay_paths.extend(sorted((diagnostic / "replays").glob("*.json")))
            snapshot_path = diagnostic / "snapshot.json"
            old_root = Path(read(snapshot_path)["request"]["root"])
            ranked = []
            for p in (diagnostic / "cases").glob("*.json"):
                report = read(p)
                entry = report["observables"]["hard_transport_rate_per_angstrom"]["largest_variance_batches"][0]
                ranked.append((entry["variance_fraction"], p, report["case"], entry))
                cfg_path = (old_root / entry["manifest"]).parent / "configuration.json"
                if file_hash(cfg_path) != report["checkpoint_configuration_sha256"]:
                    raise ValueError("Historical planning configuration changed")
                cfg = read(cfg_path)["configuration"]
                if (digest(cfg) != report["configuration_signature"] or cfg["kernel_csv_sha256"] != table.csv_sha256
                        or cfg["projectile"] != spec["projectile"] or cfg["path_length_angstrom"] != spec["path_length_angstrom"]):
                    continue
                # Reconstruct weighted means only for full-path histories. A
                # stopped trajectory has a different denominator; do not guess it.
                if report["termination_counts"] != {"path_complete": report["trajectories"]}:
                    continue
                path_mean = spec["path_length_angstrom"] * report["weight_checks"]["mean"]
                means = [path_mean] + [path_mean * report["observables"][name]["raw_estimate"] for name in OBSERVABLES[:3]]
                planning.append({"case": report["case"], "energy_ev": cfg["projectile_energy_ev"],
                                 "structure_sha256": cfg["structure_sha256"], "direction": cfg["fixed_direction"],
                                 "means": means, "trajectories": report["trajectories"], "report_sha256": file_hash(p)})
                files[str(p)] = file_hash(p); files[str(cfg_path)] = file_hash(cfg_path)
            if ranked:
                _, report_path, case_name, entry = max(ranked, key=lambda row: row[0])
                batch_path = old_root / entry["manifest"]
                if file_hash(batch_path) != entry["sha256"]:
                    raise ValueError("Historical dominant batch changed")
                cfg_path = batch_path.parent / "configuration.json"
                batch, cfg = read(batch_path), read(cfg_path)["configuration"]
                if batch["configuration_signature"] != digest(cfg):
                    raise ValueError("Historical configuration signature mismatch")
                replay_request = {"case_name": case_name, "configuration": cfg, "batch": batch,
                                  "batch_sha256": entry["sha256"]}
                for p in (snapshot_path, report_path, batch_path, cfg_path):
                    files[str(p)] = file_hash(p)
        for p in replay_paths:
            replay = read(p)
            if replay["identity"]["legacy_commit"] != commit:
                raise ValueError("Historical seed has a different runtime")
            files[str(p)] = file_hash(p)
            # Seeds guide proposals only; replayed histories never add production exposure.
            seeds.append(replay)
        pair_energies = {}
        for target in ("H", "O"):
            # Later collisions reach energies below the entrance matrix. Qualify
            # the full retained reachable domain, not only entrance energies.
            low, _ = table.energy_bounds_ev(spec["projectile"], target)
            knots = table._pair(spec["projectile"], target).energies_ev
            pair_energies[target] = sorted(set([float(low), energies[-1]] +
                [float(e) for e in knots if low <= e <= energies[-1]]))
        manifest = {"schema": SCHEMA, "controller_sha256": file_hash(__file__), "files": files,
                    "implementation_modules": {"_differential.py": file_hash(HERE / "_differential.py")},
                    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
                    "runtime_commit": commit, "kernel_manifest": str(table.manifest_path),
                    "kernel_csv_sha256": table.csv_sha256, "projectile": spec["projectile"],
                    "minimum_turning_potential_ev": table.minimum_turning_potential_ev,
                    "structures": records, "directions": spec["directions"], "energies_ev": energies,
                    "path_length_angstrom": spec["path_length_angstrom"], "seed": spec["seed"],
                    "policy": asdict(policy), "historical_seeds": seeds,
                    "historical_batch_replay": replay_request,
                    "historical_planning": planning,
                    "estimand": "finite_path_entrance_conditioned_frozen_cell_response",
                    "orientation_domain": "declared discrete directions and explicitly isotropic averages only",
                    "old_trajectory_statistics_pooled": False}
        manifest["pair_energies_ev"] = pair_energies
        kin = [table.pair_kinematics(spec["projectile"], target, energies[0]) for target in ("H", "O")]
        manifest["tail_support"] = {
            "projectile_mass_c2_ev": kin[0].projectile_mass_c2_ev,
            "target_mass_c2_ev": [k.target_mass_c2_ev for k in kin],
            "energy_floor_ev": max(pair_energies[t][0] for t in ("H", "O")),
            "model": "stationary_targets_nonenergizing_primary_sequential_binary"}
        manifest["production_contracts"] = {
            "microscopic": "Stationary-target NLH binary DCS, correlated angle/recoil area map",
            "structured_ice": "Frozen geometry with sequential binary transport; correlations retained explicitly",
            "sampled_response": "Finite-path entrance-conditioned statistics, not a local Markov cross section",
            "homogeneous_phase_closure_qualified": False}
        source = pair_reuse_source(reuse_pairs, manifest) if reuse_pairs is not None else None
        if source is not None:
            manifest["pair_reuse"] = source[0]
        manifest["signature"] = digest(manifest)
        atomic_json(root / "manifest.json", manifest)
        state = {"output_scope": policy.output_scope, "cases": {}, "intervals": [], "reserved_histories": 0, "status": "prepared",
                 "differential": initialize_differential(manifest)}
        if source is not None:
            state["differential"] = import_pair_products(root, manifest, source[1])
        for si in range(len(records)):
            for di in case_directions(manifest, si):
                keys = [add_case(state, manifest, si, di, energy) for energy in energies]
                for a, b in zip(keys[:-1], keys[1:]):
                    state["intervals"].append({"a": a, "b": b, "depth": 0, "status": "pending"})
        if replay_request:
            cfg = replay_request["configuration"]
            for case in state["cases"].values():
                structure = records[case["structure"]]
                direction = spec["directions"][case["direction"]]["vector"]
                if (structure["sha256"] == cfg["structure_sha256"] and case["energy_ev"] == cfg["projectile_energy_ev"]
                        and direction == cfg["fixed_direction"] and cfg["kernel_csv_sha256"] == table.csv_sha256
                        and cfg["projectile"] == spec["projectile"] and cfg["path_length_angstrom"] == spec["path_length_angstrom"]):
                    case.update(phase="replay", target=replay_request["batch"]["trajectory_count"])
        state["pilot_cases"] = select_pilots(manifest, state) if policy.pilot_first else []
        atomic_json(root / "state.json", state)
        atomic_json(root / "readiness.json", readiness(manifest, state))
    except BaseException:
        # Preserve a failed preparation for diagnosis; never delete user inputs.
        atomic_json(root / "preparation_failed.json", {"status": "incomplete; not runnable"})
        raise
    return state


def pair_reuse_source(source, manifest):
    """Accept unchanged pair physics/numerics, never old phase acceptance."""
    source = Path(source).resolve()
    old = read(source / "manifest.json")
    if digest({k: v for k, v in old.items() if k != "signature"}) != old["signature"]:
        raise ValueError("Pair source manifest signature changed")
    for key in ("projectile", "kernel_csv_sha256", "runtime_commit", "minimum_turning_potential_ev",
                "pair_energies_ev", "implementation_modules", "environment"):
        if old.get(key) != manifest.get(key):
            raise ValueError(f"Incompatible pair reuse: {key}")
    previous, current = Policy(**old["policy"]), Policy(**manifest["policy"])
    for key in ("kernel_relative_tolerance", "pair_distribution_tv_tolerance", "kernel_quadrature_orders",
                "pair_bin_count", "pair_max_points", "pair_max_refinements", "pair_energy_depth"):
        a, b = getattr(previous, key), getattr(current, key)
        if key == "pair_distribution_tv_tolerance" and "pair_distribution_tv_tolerance" not in old["policy"]:
            a = old["policy"]["distribution_tv_tolerance"]
        if canonical(a) != canonical(b):
            raise ValueError(f"Incompatible pair numerical policy: {key}")
    index = read(source / "differential/index.json")
    if (index["campaign_signature"] != old["signature"] or
            index["status"] != "numerically_qualified_binary_differential_cross_sections"):
        raise ValueError("Pair source has no qualified index")
    return {"root": str(source), "source_signature": old["signature"],
            "manifest_sha256": file_hash(source / "manifest.json"),
            "index_sha256": file_hash(source / "differential/index.json"),
            "scope": "Unchanged binary products only; phase histories/acceptance not imported"}, index


def import_pair_products(root, manifest, index):
    """Reuse numerical payloads with provenance; revalidate before transport.

    An interrupted preparation is explicitly incomplete. No trajectory or
    source signature is rewritten. Imported acceptance is rechecked on export.
    """
    source = Path(manifest["pair_reuse"]["root"])
    old_signature = manifest["pair_reuse"]["source_signature"]
    stage = {"nodes": {}, "intervals": index["energy_intervals"], "status": "pending", "tables_verified": False}
    for entry in tqdm(index["products"], desc="Reusing binary DCS", unit="map"):
        key = entry["key"]
        if Path(entry["path"]).name != entry["path"] or entry["path"] != key + ".json":
            raise ValueError("Unsafe pair source path")
        path = source / "differential" / entry["path"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError("Pair source product checksum changed")
        envelope = read(path); checksum = envelope.pop("sha256")
        product = envelope["product"]
        if (digest(envelope) != checksum or envelope["campaign"] != old_signature or
                not product["passes"] or pair_key(product["target"], product["energy_ev"]) != key
                or product["target"] != entry["target"] or product["energy_ev"] != entry["energy_ev"]
                or product["projectile"] != manifest["projectile"] or key in stage["nodes"]):
            raise ValueError("Invalid source pair identity or qualification")
        updated = {"campaign": manifest["signature"], "product": product,
                   "reused_from": {"campaign": old_signature, "envelope_sha256": entry["sha256"]}}
        updated["sha256"] = digest(updated)
        atomic_json(Path(root) / "differential" / entry["path"], updated)
        stage["nodes"][key] = {"target": entry["target"], "energy_ev": entry["energy_ev"], "status": "qualified"}
    verify_pair_coverage(manifest, stage)
    return stage


def pair_key(target, energy):
    return digest([target, float(energy)])[:24]


def initialize_differential(manifest):
    stage = {"nodes": {}, "intervals": [], "status": "pending"}
    for target, energies in manifest["pair_energies_ev"].items():
        for energy in energies:
            stage["nodes"][pair_key(target, energy)] = {"target": target, "energy_ev": energy, "status": "pending"}
        for left, right in zip(energies[:-1], energies[1:]):
            stage["intervals"].append({"target": target, "a": pair_key(target, left),
                                       "b": pair_key(target, right), "depth": 0, "status": "pending"})
    if len(stage["nodes"]) > manifest["policy"]["pair_max_products"]:
        raise ValueError("Base pair energy domain exceeds the pair-product budget")
    return stage


def differential_backend():
    # Also works when the public entry point is loaded by importlib in tests.
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from _differential import build_pair_differential, compare_pair_energy
    return build_pair_differential, compare_pair_energy


def pair_product(root, manifest, key):
    path = Path(root) / "differential" / (key + ".json")
    envelope = read(path)
    checksum = envelope.pop("sha256")
    if digest(envelope) != checksum or envelope["campaign"] != manifest["signature"]:
        raise ValueError("Changed or incompatible pair product")
    product = envelope["product"]
    if pair_key(product["target"], product["energy_ev"]) != key or product["projectile"] != manifest["projectile"]:
        raise ValueError("Pair product physical identity mismatch")
    return product


def execute_pair(task):
    root, manifest, key, node = task
    root = Path(root); policy = Policy(**manifest["policy"])
    Table, _, _ = load_runtime(root)
    cache_key = (str(root), "pair_table")
    if cache_key not in _CACHE:
        _CACHE[cache_key] = Table(manifest["kernel_manifest"])
    table = _CACHE[cache_key]
    checkpoint = root / "differential" / "work" / (key + ".json")
    identity = {"campaign": manifest["signature"], "pair": key}
    cache = {}
    if checkpoint.exists():
        saved = read(checkpoint); signature = saved.pop("sha256")
        if digest(saved) != signature or saved["identity"] != identity:
            raise ValueError("Pair integration checkpoint changed")
        cache = saved["reference_cache"]
    last_saved = time.monotonic()

    def save_cache():
        saved = {"identity": identity, "reference_cache": cache}
        saved["sha256"] = digest(saved)
        atomic_json(checkpoint, saved)

    def progress(completed, maximum):
        nonlocal last_saved
        if time.monotonic() - last_saved >= 5:
            save_cache(); last_saved = time.monotonic()

    build, _ = differential_backend()
    started = time.monotonic()
    try:
        result = build(table, manifest["projectile"], node["target"], node["energy_ev"],
                       relative_tolerance=policy.kernel_relative_tolerance,
                       tv_tolerance=policy.pair_distribution_tv_tolerance,
                       quadrature_orders=policy.kernel_quadrature_orders,
                       max_points=policy.pair_max_points, max_refinements=policy.pair_max_refinements,
                       bin_count=policy.pair_bin_count, progress=progress, reference_cache=cache)
    finally:
        save_cache()
    result["wall_seconds_this_attempt"] = time.monotonic() - started
    envelope = {"campaign": manifest["signature"], "product": result}
    envelope["sha256"] = digest(envelope)
    atomic_json(root / "differential" / (key + ".json"), envelope)
    return {"status": "qualified" if result["passes"] else "blocked", "reason": result.get("reason"),
            "evaluations": result["evaluations"], "wall_seconds": result["wall_seconds_this_attempt"]}


def advance_differential(root, manifest, state):
    """Adapt shared energy intervals; no ice Monte Carlo is used for pair DCS."""
    stage = state["differential"]; policy = Policy(**manifest["policy"])
    _, compare = differential_backend()
    # Adopt products committed before a controller interruption.
    for key, node in stage["nodes"].items():
        if node["status"] == "pending" and (Path(root) / "differential" / (key + ".json")).exists():
            result = pair_product(root, manifest, key)
            node.update(status="qualified" if result["passes"] else "blocked", reason=result.get("reason"),
                        evaluations=result["evaluations"], wall_seconds=result.get("wall_seconds_this_attempt"))
    for interval in list(stage["intervals"]):
        if interval["status"] in ("qualified", "split", "blocked"):
            continue
        a, b = [stage["nodes"][interval[k]] for k in ("a", "b")]
        if any(n["status"] == "blocked" for n in (a, b)):
            interval.update(status="blocked", reason="pair_node_not_qualified")
            continue
        if any(n["status"] != "qualified" for n in (a, b)):
            continue
        if "checks" not in interval:
            energies = [math.exp((1 - w) * math.log(a["energy_ev"]) + w * math.log(b["energy_ev"]))
                        for w in (.25, .5, .75)]
            keys = [pair_key(a["target"], e) for e in energies]
            if len(stage["nodes"]) + sum(k not in stage["nodes"] for k in keys) > policy.pair_max_products:
                interval.update(status="blocked", reason="pair_product_budget")
                continue
            for key, energy in zip(keys, energies):
                stage["nodes"].setdefault(key, {"target": a["target"], "energy_ev": energy, "status": "pending"})
            interval["checks"] = keys
        tests = [stage["nodes"][k] for k in interval["checks"]]
        if any(n["status"] == "blocked" for n in tests):
            interval.update(status="blocked", reason="interior_pair_node_not_qualified")
            continue
        if any(n["status"] != "qualified" for n in tests):
            continue
        left, right = [pair_product(root, manifest, interval[k]) for k in ("a", "b")]
        checks = [compare(left, pair_product(root, manifest, key), right,
                          relative_tolerance=policy.interpolation_relative_tolerance,
                          tv_tolerance=policy.interpolation_tv_tolerance) for key in interval["checks"]]
        interval["assessment"] = checks
        if all(check["passes"] for check in checks):
            interval["status"] = "qualified"
        elif interval["depth"] >= policy.pair_energy_depth:
            interval.update(status="blocked", reason="pair_energy_refinement_limit")
        else:
            interval["status"] = "split"
            ordered = [interval["a"], *interval["checks"], interval["b"]]
            for low, high in zip(ordered[:-1], ordered[1:]):
                stage["intervals"].append({"target": interval["target"], "a": low, "b": high,
                                           "depth": interval["depth"] + 1, "status": "pending"})
    complete = all(n["status"] == "qualified" for n in stage["nodes"].values()) and all(
        i["status"] in ("qualified", "split") for i in stage["intervals"])
    stage["status"] = "qualified" if complete else ("not_qualified" if any(
        n["status"] == "blocked" for n in stage["nodes"].values()) or any(
        i["status"] == "blocked" for i in stage["intervals"]) else "pending")
    return complete


def verify_pair_coverage(manifest, stage):
    """Reject deletion of a base interval or one required refinement child."""
    original = initialize_differential(manifest)
    if not set(original["nodes"]).issubset(stage["nodes"]):
        raise ValueError("Missing requested pair energies")
    pairs = {(i["a"], i["b"]): i for i in stage["intervals"]}
    if len(pairs) != len(stage["intervals"]):
        raise ValueError("Duplicated pair energy interval")
    for interval in original["intervals"]:
        if (interval["a"], interval["b"]) not in pairs:
            raise ValueError("Missing requested pair energy interval")
    for key, node in stage["nodes"].items():
        if pair_key(node["target"], node["energy_ev"]) != key:
            raise ValueError("Changed pair energy identity")
    for interval in stage["intervals"]:
        a, b = [stage["nodes"][interval[k]] for k in ("a", "b")]
        if a["target"] != b["target"] or a["target"] != interval["target"] or a["energy_ev"] >= b["energy_ev"]:
            raise ValueError("Invalid pair energy interval")
        if interval["status"] in ("qualified", "split"):
            if len(interval.get("checks", [])) != 3:
                raise ValueError("Missing independent pair interpolation checks")
            for key, w in zip(interval["checks"], (.25, .5, .75)):
                node = stage["nodes"][key]
                expected = math.exp((1 - w) * math.log(a["energy_ev"]) + w * math.log(b["energy_ev"]))
                if node["target"] != a["target"] or not math.isclose(node["energy_ev"], expected, rel_tol=1e-14):
                    raise ValueError("Changed independent pair interpolation energy")
        if interval["status"] == "split":
            keys = [interval["a"], *interval["checks"], interval["b"]]
            if any((a, b) not in pairs for a, b in zip(keys[:-1], keys[1:])):
                raise ValueError("Missing pair refinement child")


def validate_pair_product(product, manifest, table=None):
    from types import SimpleNamespace
    differential_backend()
    from _differential import joint_bin_masses, two_body_observables
    policy = Policy(**manifest["policy"])
    if (not product["passes"] or product["projectile"] != manifest["projectile"]
            or product["minimum_turning_potential_ev"] != manifest["minimum_turning_potential_ev"]):
        raise ValueError("Unqualified or mismatched pair product")
    sigma = math.pi * product["maximum_impact_parameter_angstrom"] ** 2 * 1e-16
    if not math.isclose(product["hard_cross_section_cm2"], sigma, rel_tol=1e-12, abs_tol=0):
        raise ValueError("Pair total area differs from pi*bmax squared")
    if table is not None:
        expected = table.maximum_impact_parameter_angstrom(product["projectile"], product["target"], product["energy_ev"])
        if not math.isclose(product["maximum_impact_parameter_angstrom"], expected, rel_tol=1e-12, abs_tol=0):
            raise ValueError("Pair hard boundary differs from the signed physical model")
        expected_kinematics = table.pair_kinematics(product["projectile"], product["target"], product["energy_ev"])
        if any(not math.isclose(value, getattr(expected_kinematics, name), rel_tol=1e-13, abs_tol=0)
               for name, value in product["kinematics"].items()):
            raise ValueError("Pair masses or kinematics differ from the signed physical model")
    if sigma == 0:
        if product["status"] != "analytically_closed":
            raise ValueError("Zero hard area requires analytic closure")
        return
    if (product.get("angular_integral") != "positive_turning_difference_v1"
            or not 0 <= product["maximum_turning_identity_residual_fraction_of_roundoff_guard"] <= 1):
        raise ValueError("Pair reference lacks the stable integral or resolved turning root")
    checks = product["numerical_checks"]
    for name in ("recoil_relative_L1", "transport_relative_L1", "maximum_cm_angle_error_over_pi"):
        if not 0 <= checks[name] <= policy.kernel_relative_tolerance / 2:
            raise ValueError("Pair numerical interpolation check failed")
    if (not all(0 <= v <= policy.kernel_relative_tolerance / 4 for v in checks["moment_quadrature_relative_change"])
            or not 0 <= product["maximum_direct_quadrature_relative_change"] <= policy.kernel_relative_tolerance / 4
            or not 0 <= checks["bin_TV_with_root_allowance"] <= policy.pair_distribution_tv_tolerance):
        raise ValueError("Pair reference or joint-TV check failed")
    kin = SimpleNamespace(**product["kinematics"])
    bins = product["joint_bins"]
    mass = joint_bin_masses(kin, product["area_quantile"], product["theta_cm_rad"],
                            bins["angle_edges_rad"], bins["recoil_edges_ev"])
    indices = np.asarray(bins["indices_angle_recoil"])
    if (indices.ndim != 2 or indices.shape[1] != 2 or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= np.asarray(mass.shape))
            or len(np.unique(indices, axis=0)) != len(indices)):
        raise ValueError("Invalid pair joint-cell indices")
    stored = np.zeros_like(mass); stored[tuple(indices.T)] = bins["probability_mass"]
    if not np.allclose(stored, mass, rtol=1e-10, atol=1e-12):
        raise ValueError("Pair joint bins disagree with correlated kinematic map")
    if not np.allclose(np.asarray(bins["probability_mass"]) * sigma,
                       bins["integrated_cross_section_cm2"], rtol=1e-12, atol=0):
        raise ValueError("Pair DCS area units disagree")
    angle, recoil = two_body_observables(kin, product["theta_cm_rad"])
    for actual, expected in ((product["theta_lab_rad"], angle), (product["recoil_energy_ev"], recoil),
                             (product["projectile_out_energy_ev"], product["energy_ev"] - recoil)):
        if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise ValueError("Pair energy-angle kinematics changed")
    if (not 0 < product["mean_recoil_energy_ev"] <= product["energy_ev"]
            or not 0 < product["mean_transport"] <= 2):
        raise ValueError("Pair moments violate kinematic bounds")
    for name, mean in (("recoil_moment_cross_section_ev_cm2", product["mean_recoil_energy_ev"]),
                       ("transport_cross_section_cm2", product["mean_transport"])):
        if not math.isclose(product[name], sigma * mean, rel_tol=1e-12, abs_tol=0):
            raise ValueError("Pair moment cross-section units disagree")


def export_differential(root, manifest, state):
    stage = state["differential"]
    verify_pair_coverage(manifest, stage)
    Table, _, _ = load_runtime(root)
    table = Table(manifest["kernel_manifest"])
    for key, node in stage["nodes"].items():
        validate_pair_product(pair_product(root, manifest, key), manifest, table)
        node["status"] = "qualified"
    # State is a journal, not numerical evidence. Recompute each leaf decision
    # from checked independent interior maps before issuing a consumer index.
    for interval in stage["intervals"]:
        if interval["status"] != "split":
            interval["status"] = "pending"
    if not advance_differential(root, manifest, state):
        raise ValueError("Pair DCS export requires the complete qualified energy domain")
    products = []
    for key, node in sorted(stage["nodes"].items()):
        product = pair_product(root, manifest, key)
        if not product["passes"]:
            raise ValueError("Unqualified pair product")
        if product["hard_cross_section_cm2"] > 0:
            masses = np.asarray(product["joint_bins"]["probability_mass"])
            sigma = np.asarray(product["joint_bins"]["integrated_cross_section_cm2"])
            if np.any(masses < 0) or not np.isclose(masses.sum(), 1., atol=1e-12, rtol=0):
                raise ValueError("Invalid pair DCS normalization")
            if not np.allclose(sigma, masses * product["hard_cross_section_cm2"], rtol=1e-12, atol=0):
                raise ValueError("Pair DCS area units disagree")
        products.append({"key": key, "target": node["target"], "energy_ev": node["energy_ev"],
                         "path": key + ".json", "sha256": file_hash(Path(root) / "differential" / (key + ".json"))})
    result = {"schema": SCHEMA, "campaign_signature": manifest["signature"], "projectile": manifest["projectile"],
              "status": "numerically_qualified_binary_differential_cross_sections", "products": products,
              "energy_intervals": stage["intervals"], "numerical_policy": manifest["policy"],
              "units": {"energy": "eV total projectile kinetic energy", "area": "cm2", "angle": "rad"},
              "interpolation": "Use qualified leaf endpoints: log(theta_CM) linear in log(E) at common area quantile; derive lab angle and recoil by exact kinematics",
              "total_cross_section": "Evaluate pi*b_max(E)^2 analytically at the requested energy; do not interpolate total area",
              "hard_boundary": {"minimum_turning_potential_ev": manifest["minimum_turning_potential_ev"],
                                "rule": "b_max=r_threshold*sqrt(1-V_min/E_cm) for E_cm>V_min; otherwise zero",
                                "target_threshold_radii_angstrom": {target: table.turning_threshold_radius_angstrom(manifest["projectile"], target) for target in ("H", "O")}},
              "joint_law": "q=(b/b_max)^2 uniform on [0,1]; lab angle and recoil share the same q. Never sample marginals independently",
              "azimuth": "Uniform for an isolated central-potential pair; explicit target geometry determines it in structured ice",
              "charge_state_dependence": "None in this retained nuclear pair potential; do not duplicate charge-state campaigns",
              "phase_operator": {"contract": manifest["production_contracts"]["structured_ice"],
                                 "structures": manifest["structures"], "runtime_commit": manifest["runtime_commit"],
                                 "homogeneous_density_only_closure_qualified": False},
              "limitations": ["Numerical refinement evidence, not a rigorous continuous-domain enclosure or physical-model accuracy bound",
                              "No phase-specific local cross section inferred by dividing finite-path counts by density",
                              "Recoil energy is transferred to a recoil; it is not automatically deposited locally",
                              "Frozen sequential binary model omits simultaneous multi-centre forces and recoil cascades"]}
    atomic_json(Path(root) / "differential" / "index.json", result)
    stage["index_sha256"] = file_hash(Path(root) / "differential" / "index.json")
    stage["tables_verified"] = True
    return result


def run_differential(root, manifest, state, workers, started):
    """Bounded, continuously replenished, restartable deterministic first stage."""
    from concurrent.futures import wait, FIRST_COMPLETED
    policy = Policy(**manifest["policy"]); stage = state["differential"]
    verify_pair_coverage(manifest, stage)
    limit = None if os.environ.get("PBS_JOBID") else 2
    submitted = 0; pending = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        with tqdm(total=len(stage["nodes"]), initial=sum(n["status"] != "pending" for n in stage["nodes"].values()),
                  desc="NLH pair-energy maps (adaptive)", unit="map") as progress:
            while not _STOP and time.monotonic() - started < policy.max_seconds:
                advance_differential(root, manifest, state)
                progress.total = len(stage["nodes"]); progress.refresh()
                live = set(pending.values())
                for key, node in stage["nodes"].items():
                    if len(pending) >= workers or (limit is not None and submitted >= limit):
                        break
                    if node["status"] == "pending" and key not in live:
                        pending[pool.submit(execute_pair, (root, manifest, key, node))] = key
                        submitted += 1
                atomic_json(Path(root) / "state.json", state)
                if not pending:
                    break
                completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    key = pending.pop(future)
                    try:
                        stage["nodes"][key].update(future.result())
                    except Exception as exc:
                        stage["nodes"][key].update(status="blocked", reason=f"{type(exc).__name__}: {exc}")
                    progress.update(1)
                    atomic_json(Path(root) / "state.json", state)
            # In-flight bounded maps commit their own products. If a PBS kill
            # intervenes, the next run adopts those receipts and point caches.
    if advance_differential(root, manifest, state):
        export_differential(root, manifest, state)
    atomic_json(Path(root) / "state.json", state)
    return stage.get("tables_verified", False)


def case_directions(manifest, structure):
    requested = manifest["structures"][structure].get("directions")
    return [i for i, d in enumerate(manifest["directions"]) if requested is None or d["name"] in requested]


def add_case(state, manifest, si, di, energy):
    key = digest([si, di, energy])[:20]
    if key in state["cases"]:
        return key
    if len(state["cases"]) >= manifest["policy"]["max_cases"]:
        return None
    state["cases"][key] = {"index": len(state["cases"]), "structure": si, "direction": di,
                           "energy_ev": energy, "phase": "training", "round": 0, "level": 0,
                           "target": manifest["policy"]["training_histories"], "proposal": {"boxes": [], "alpha": [1.]},
                           "cohorts": {}, "status": "pending",
                           "kernel_mode": "qualified_pair_maps"}
    if state.get("differential", {}).get("tables_verified"):
        state["cases"][key]["kernel"] = pair_gate(state)
    centres = seed_points(manifest, state["cases"][key])
    if centres:
        boxes = proposal_boxes(centres, Policy(**manifest["policy"]))
        state["cases"][key]["proposal"] = {"boxes": boxes, "alpha": [.5] + [.5 / len(boxes)] * len(boxes)}
    return key


def select_pilots(manifest, state):
    """One lowest- and highest-energy case per material, before broad sampling.

    These diagnose feasibility only: no pilot can qualify an untested case.
    Prefer a case with retained rare-event seeds when several directions exist.
    """
    selected = []
    phases = sorted({r["phase"] for r in manifest["structures"]})
    for phase in phases:
        for energy in (min(manifest["energies_ev"]), max(manifest["energies_ev"])):
            candidates = [(key, case) for key, case in state["cases"].items()
                          if manifest["structures"][case["structure"]]["phase"] == phase and case["energy_ev"] == energy]
            key, _ = max(candidates, key=lambda item: (len(seed_points(manifest, item[1])), -item[1]["index"]))
            selected.append(key)
    return selected


def proposal_boxes(centres, policy):
    # Deduplicate identical historical/training centres before evaluating q.
    centres = list({tuple(c): c for c in centres}.values())[:2 * policy.local_centres]
    return [{"centre": centre, "half_width": [2. ** -power] * len(centre)}
            for centre in centres for power in policy.proposal_scale_powers]


def box_density(points, boxes):
    points = np.atleast_2d(points)
    output = np.empty((len(points), len(boxes)))
    for j, box in enumerate(boxes):
        distance = np.abs((points - np.asarray(box["centre"]) + .5) % 1. - .5)
        half = np.asarray(box["half_width"])
        output[:, j] = np.all(distance < half, axis=1) / np.prod(2 * half)
    return output


def mixture_density(base_density, points, proposal):
    alpha = np.asarray(proposal["alpha"])
    local = box_density(points, proposal["boxes"])
    return alpha[0] * np.asarray(base_density) + local @ alpha[1:]


def isotropic_point(vector):
    v = np.asarray(vector)
    return [(v[2] + 1.) / 2., math.atan2(v[1], v[0]) % (2 * math.pi) / (2 * math.pi)]


def direction_from_point(point):
    mu, phi = 2 * point[-2] - 1, 2 * math.pi * point[-1]
    r = math.sqrt(max(0., 1 - mu * mu))
    return np.array([r * math.cos(phi), r * math.sin(phi), mu])


def pair_gate(state):
    return {"passes": True, "scope": "Independent pair-map and energy checks over the full reachable retained domain",
            "index_sha256": state["differential"]["index_sha256"]}


def qualified_pair_table(root, manifest, table):
    key = (str(root), "qualified_pair_table")
    if key not in _CACHE:
        index_path = Path(root) / "differential" / "index.json"
        stage = read(Path(root) / "state.json")["differential"]
        if not stage.get("tables_verified") or file_hash(index_path) != stage["index_sha256"]:
            raise ValueError("Qualified pair-table handoff missing or changed")
        index = read(index_path)
        if index["campaign_signature"] != manifest["signature"]:
            raise ValueError("Pair-table index belongs to another campaign")
        leaves = [i for i in index["energy_intervals"] if i["status"] == "qualified"]
        needed = {i[k] for i in leaves for k in ("a", "b")}
        entries = {entry["key"]: entry for entry in index["products"]}
        products = {}
        for pair in needed:
            entry = entries[pair]
            if entry["path"] != pair + ".json" or file_hash(index_path.parent / entry["path"]) != entry["sha256"]:
                raise ValueError("Pair-table consumer checksum mismatch")
            products[pair] = pair_product(root, manifest, pair)
        differential_backend()
        from _differential import PairMapTable
        _CACHE[key] = PairMapTable(table, products, leaves)
    return _CACHE[key]


def verify_differential(root, manifest=None):
    """Read-only consumer verification, including the actual exported map law."""
    root = Path(root); manifest = verify(root) if manifest is None else manifest
    state = read(root / "state.json"); stage = state["differential"]
    verify_pair_coverage(manifest, stage)
    path = root / "differential" / "index.json"
    if not stage.get("tables_verified") or file_hash(path) != stage["index_sha256"]:
        raise ValueError("Pair index is missing or differs from the production receipt")
    index = read(path)
    if (index["campaign_signature"] != manifest["signature"] or index["projectile"] != manifest["projectile"]
            or index["energy_intervals"] != stage["intervals"]):
        raise ValueError("Pair production identity or interval coverage mismatch")
    keys = [entry["key"] for entry in index["products"]]
    if len(keys) != len(set(keys)) or set(keys) != set(stage["nodes"]):
        raise ValueError("Incomplete pair production matrix")
    Table, _, _ = load_runtime(root); table = Table(manifest["kernel_manifest"])
    products = {}
    for entry in index["products"]:
        key = entry["key"]
        if entry["path"] != key + ".json" or file_hash(path.parent / entry["path"]) != entry["sha256"]:
            raise ValueError("Changed pair production file")
        products[key] = pair_product(root, manifest, key)
        validate_pair_product(products[key], manifest, table)
    _, compare = differential_backend(); policy = Policy(**manifest["policy"])
    for interval in stage["intervals"]:
        if interval["status"] == "split":
            continue
        if interval["status"] != "qualified" or not all(compare(products[interval["a"]], products[key], products[interval["b"]],
                relative_tolerance=policy.interpolation_relative_tolerance,
                tv_tolerance=policy.interpolation_tv_tolerance)["passes"] for key in interval["checks"]):
            raise ValueError("Unqualified production pair interpolation")
    return {"campaign_signature": manifest["signature"], "index_sha256": file_hash(path),
            "verified_pair_energy_maps": len(products), "binary_dcs_ready": True,
            "homogeneous_phase_closure_qualified": False, "new_trajectories": 0}


def get_transport(root, manifest, case):
    key = (str(root), case["structure"])
    if key not in _CACHE:
        Table, load_structure, Transport = load_runtime(root)
        record = manifest["structures"][case["structure"]]
        structure = load_structure(record["path"], frame_index=record["frame_index"], metadata_path=record["metadata"])
        transport = Transport(structure, Table(manifest["kernel_manifest"]))
        if structure.source_sha256 != record["sha256"] or transport.kernels.csv_sha256 != manifest["kernel_csv_sha256"]:
            raise ValueError("Worker input differs from the signed physical configuration")
        transport.kernels = qualified_pair_table(root, manifest, transport.kernels)
        _CACHE[key] = transport
    return _CACHE[key]


def sample_initial(transport, manifest, case, proposal, rng):
    direction = manifest["directions"][case["direction"]]["vector"]
    dim = 5 if direction is None else 3
    component = int(rng.choice(len(proposal["alpha"]), p=proposal["alpha"]))
    if component:
        box = proposal["boxes"][component - 1]
        point = (np.array(box["centre"]) + (2 * rng.random(dim) - 1) * box["half_width"]) % 1.
        direction = direction_from_point(point) if direction is None else np.asarray(direction)
        position = point[:3] @ transport.structure.lattice_angstrom
    else:
        point = rng.random(dim)
        direction = direction_from_point(point) if direction is None else np.asarray(direction)
        position, _ = transport.sample_collision_tube_mixture(
            manifest["projectile"], case["energy_ev"], direction, manifest["path_length_angstrom"], .5, rng)
        point[:3] = (position @ np.linalg.inv(transport.structure.lattice_angstrom)) % 1.
    base = .5 + .5 * transport._collision_tube_density_over_uniform(
        manifest["projectile"], case["energy_ev"], position, direction, manifest["path_length_angstrom"])
    density = float(mixture_density([base], [point], proposal)[0])
    if not math.isfinite(density) or density < .5 * proposal["alpha"][0] * (1 - 1e-12):
        raise ValueError("Invalid defensive proposal density")
    return point, position, direction, density, base


def relative_precision(estimate, interval, tolerance):
    """A simultaneous interval implies this relative error for every enclosed mean.

    Scaling by the positive lower endpoint avoids claiming relative accuracy
    for an unresolved zero/rare channel. The requirement is an error radius,
    not a full interval width. No data-derived rate floor is introduced.
    """
    low, high = interval
    if high is None or low <= 0 or not math.isfinite(estimate):
        return {"passes": False, "relative_error_bound": None, "allowed_relative_error": tolerance}
    radius = max(abs(estimate - low), abs(high - estimate))
    return {"passes": bool(radius <= tolerance * low), "relative_error_bound": float(radius / low),
            "allowed_relative_error": tolerance}


def total_variation_bound(probability, lower, upper):
    """Enclose TV = 1/2 sum |p-p_hat| on a declared finite partition.

    All cell intervals must hold simultaneously. Empty cells remain in the
    sum. Simplex conservation bounds positive and negative mass separately;
    taking the tightest of these valid bounds avoids summing both needlessly.
    This is not a TV guarantee for an unspecified density inside each cell.
    """
    p, lo, hi = map(lambda x: np.asarray(x, float), (probability, lower, upper))
    if p.shape != lo.shape or p.shape != hi.shape or np.any(lo > hi):
        raise ValueError("Invalid simultaneous probability intervals")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError("Point probabilities must form a simplex")
    positive, negative = np.maximum(0., hi - p), np.maximum(0., p - lo)
    return float(min(1., positive.sum(), negative.sum(), .5 * np.maximum(positive, negative).sum()))


def empirical_interval(n, total, total2, upper, alpha):
    """Two-sided Maurer--Pontil bounded-iid empirical Bernstein interval.

    The union bound over both signs uses log(4/alpha). Fixed-cohort doubling
    looks, cases, bins and proposal attempts receive disjoint error budgets.
    """
    total, total2, upper = np.asarray(total), np.asarray(total2), np.asarray(upper)
    if n < 2:
        return np.zeros_like(total), np.broadcast_to(upper, total.shape).copy()
    mean = total / n
    variance = np.maximum(0., (total2 - total * total / n) / (n - 1))
    log = math.log(4 / alpha)
    radius = np.sqrt(2 * variance * log / n) + 7 * upper * log / (3 * (n - 1))
    return np.maximum(0., mean - radius), np.minimum(upper, mean + radius)


def ratio_interval(lower, upper, numerator, denominator):
    if lower[denominator] <= 0:
        return [0., None]
    return [float(lower[numerator] / upper[denominator]), float(upper[numerator] / lower[denominator])]


def betting_interval(values, bound, alpha, grid_size=20, counts=None):
    """Two-sided, time-uniform constant-grid betting interval for a bounded mean.

    Waudby-Smith & Ramdas (2023), Appendix B.6: average products of
    1 + lambda * (X-m), with lambda = j/(G*m) or -j/(G*(1-m)). Each
    one-sided mixture is a nonnegative test martingale. Reject a candidate
    if either mixture exceeds 2/alpha (Ville plus a two-sided union bound).
    Monotonicity gives an interval containing the ordinary sample mean.

    Here X is an entire likelihood-weighted history divided by its declared
    support bound, NOT a collision or a self-normalized importance sample.
    Integer counts compress identical observations; they are not importance
    weights. The fixed grid/prior must not be fitted to validation outcomes.
    Zero events retain a positive upper limit. No empirical support truncation.
    """
    x = np.asarray(values, dtype=float)
    if (x.ndim != 1 or not math.isfinite(bound) or bound <= 0 or
            not 0 < alpha < 1 or type(grid_size) is not int or grid_size < 1 or
            not np.isfinite(x).all() or np.any(x < 0) or np.any(x > bound)):
        raise ValueError("Invalid bounded betting observations or policy")
    if counts is None:
        # Exact compression is valuable for zero-heavy rare-event observations.
        x, counts = np.unique(x, return_counts=True)
    else:
        counts = np.asarray(counts)
        if (counts.shape != x.shape or counts.dtype.kind not in "iu" or
                np.any(counts <= 0)):
            raise ValueError("Betting multiplicities must be positive integer counts")
    n = sum(int(v) for v in counts)
    if n < 2:
        return [0., float(bound)]
    x = x / bound
    counts = counts.astype(float)
    centre = float(np.dot(x, counts / n))
    fractions = np.arange(1, grid_size + 1, dtype=float) / grid_size
    threshold = math.log(2 / alpha)

    def capital(m, positive):
        # Work directly with m-X on the upper side: forming 1-X first loses
        # significant digits for near-zero normalized recoil/transport means.
        delta = (x - m) / m if positive else (m - x) / (1 - m)
        wealth = np.empty(grid_size)
        for j, fraction in enumerate(fractions):
            with np.errstate(divide="ignore", invalid="raise"):
                terms = np.log1p(fraction * delta)
            wealth[j] = np.dot(counts, terms)
        # Roundoff allowance enlarges acceptance; never reject on a numerical
        # equality. This is numerical guarding, not a physical error tolerance.
        guard = 64 * np.finfo(float).eps * n * max(1., abs(threshold))
        return float(logsumexp(wealth) - math.log(grid_size)) - guard

    def boundary(positive):
        outside, inside = (0., centre) if positive else (1., centre)
        if outside == inside:
            return outside
        for _ in range(160):
            middle = outside + (inside - outside) / 2
            if middle == outside or middle == inside:
                break
            if capital(middle, positive) > threshold:
                outside = middle
            else:
                inside = middle
            if abs(inside - outside) <= 64 * np.finfo(float).eps * max(abs(middle), np.finfo(float).tiny):
                break
        # The rejected side is an outward enclosure of the exact crossing.
        return outside

    return [float(bound * boundary(True)), float(bound * boundary(False))]


def scalar_confidence_alpha(policy):
    """One quarter of alpha: whole scores plus both scalar tail partitions.

    Betting is time-uniform and scalar rows are grid-independent: no extra
    spending over histogram levels or scheduled looks. The remaining budget
    covers assessment histograms (1/4) and interpolation histograms (1/2).
    """
    means = 5 + 2 * (len(policy.tail_score_fractions) - 1)
    return (1 - policy.confidence) / (4 * 2 * policy.max_cases * policy.max_proposal_rounds * means)


def physical_scalar_bounds(manifest, energy):
    """Whole-history support, not a bound inferred from sampled maxima.

    With rest energies m,M, momentum conservation gives
      2*p*p'*(1-cos(theta)) <= (2*M+E0)*(E-E').
    For E'/E >= 1/2 this bounds the angular score by
      (M/m+E0/(2*m))/sqrt(2) * log(E/E').
    For E'/E < 1/2, use 2 <= 2/log(2)*log(E/E').
    Logarithms telescope down to the kernel floor; at most one terminal
    collision crosses that floor, and contributes at most two. Thus angular
    support does not assume that the computational collision cap is physical.
    Total recoil is bounded by E0. Production count support uses the minimum
    fractional recoil of the qualified interpolant, not a computation cap.
    A missing recoil floor is permitted only for optimistic planning/test
    fixtures; assess obtains it from the qualified maps before certification.
    """
    policy = Policy(**manifest["policy"])
    support = manifest.get("tail_support")
    angular = 2. * policy.max_collisions
    count = float(policy.max_collisions)
    if support is not None:
        m = float(support["projectile_mass_c2_ev"])
        targets = np.asarray(support["target_mass_c2_ev"], float)
        floor = float(support["energy_floor_ev"])
        if (support.get("model") != "stationary_targets_nonenergizing_primary_sequential_binary"
                or not math.isfinite(m) or m <= 0 or targets.shape != (2,)
                or not np.isfinite(targets).all() or np.any(targets <= 0)
                or not math.isfinite(floor) or not math.isfinite(energy) or not 0 < floor <= energy):
            raise ValueError("Invalid physical tail support")
        coefficient = max((float(targets.max()) / m + energy / (2*m)) / math.sqrt(2), 2 / math.log(2))
        angular = coefficient * math.log(energy / floor) + 2.
        if "minimum_recoil_fraction" in support:
            fraction = support["minimum_recoil_fraction"]
            if not math.isfinite(fraction) or not 0 < fraction < 1:
                raise ValueError("No positive physical recoil floor for history-count support")
            count = math.ceil(math.log(energy/floor) / -math.log1p(-fraction)) + 1.
    # Outward arithmetic allowance, not a relaxation of a scientific tolerance.
    return np.nextafter(np.array([manifest["path_length_angstrom"], count,
                                   energy, angular]) * (1 + 128*np.finfo(float).eps), np.inf)


def assessment_physics(root, manifest, energy):
    """Enclose event-count support from the actual positive pair interpolant.

    T/E = beta(E)*sin(theta_CM/2)^2; beta is nondecreasing in E for positive
    stationary masses. Linear area interpolation and log-energy interpolation
    preserve endpoint minima. Each step therefore reduces energy by at least
    the enclosed fraction. This certifies the implemented pair law, not an
    uncomputed exact-potential enclosure. Numerical map checks remain separate.
    """
    index_path = Path(root)/"differential/index.json"
    checksum = file_hash(index_path)
    key = ("history_support", str(Path(root).resolve()), manifest["signature"], checksum, energy)
    if key not in _CACHE:
        index = read(index_path)
        if index["campaign_signature"] != manifest["signature"]:
            raise ValueError("Wrong pair index for physical tail support")
        entries = {row["key"]: row for row in index["products"]}
        needed = {interval[k] for interval in index["energy_intervals"]
                  if interval["status"] == "qualified" and entries[interval["a"]]["energy_ev"] <= energy
                  for k in ("a", "b")}
        minima = {"H": math.inf, "O": math.inf}
        for pair in needed:
            entry = entries[pair]
            if entry["path"] != pair+".json" or file_hash(index_path.parent/entry["path"]) != entry["sha256"]:
                raise ValueError("Changed pair evidence for physical history support")
            product = pair_product(root, manifest, pair)
            angles = product["theta_cm_rad"]
            if angles:
                minima[product["target"]] = min(minima[product["target"]], min(angles))
        support = dict(manifest["tail_support"])
        m, floor = support["projectile_mass_c2_ev"], support["energy_floor_ev"]
        fractions = []
        for target, mass in zip(("H", "O"), support["target_mass_c2_ev"]):
            theta = minima[target]
            if not 0 < theta <= math.pi:
                raise ValueError("Missing positive pair-angle floor; history support is unqualified")
            beta = (4*m*mass + 2*mass*floor) / ((m+mass)**2 + 2*mass*floor)
            fractions.append(beta*math.sin(theta/2)**2)
        # Guard basic interpolation/kinematic roundoff and outgoing-energy
        # subtraction. If this exhausts the floor, refuse a finite count bound.
        fraction = .5*(min(fractions) - 32*np.finfo(float).eps)
        if fraction <= 0:
            raise ValueError("Pair recoil floor unresolved at floating precision; count support unqualified")
        support.update(minimum_recoil_fraction=fraction, minimum_cm_angles_rad=minima,
                       pair_index_sha256=checksum)
        _CACHE[key] = support
    return {**manifest, "tail_support": _CACHE[key]}


def tail_scores(values, support, fractions):
    """Disjoint whole-history score bands; unknown band probabilities are estimated.

    Contributions sum exactly to the original score. This is not conditional
    sampling with assumed known stratum masses, nor a truncation of large scores.
    """
    x = np.asarray(values, float)
    edges = np.asarray(fractions, float) * support
    if (x.ndim != 1 or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > support)
            or not math.isfinite(support) or support <= 0 or edges[0] != 0
            or edges[-1] != support or np.any(np.diff(edges) <= 0)):
        raise ValueError("Score outside declared physical tail support")
    labels = np.minimum(np.searchsorted(edges, x, side="right") - 1, len(edges)-2)
    scores = np.zeros((len(x), len(edges)-1))
    scores[np.arange(len(x)), labels] = x
    return scores, edges[1:]


def scalar_betting_bounds(root, entries, bounds, alpha, grid_size, *, physical_bounds=None, tail_fractions=()):
    """Rebuild from the complete deterministic prefix of verified checkpoints."""
    parts, raw_parts, stop = [], [], 0
    for entry in sorted(entries, key=lambda e: e["identity"]["start"]):
        identity = entry["identity"]
        if identity["start"] != stop:
            raise ValueError("Scalar certification requires a contiguous history prefix")
        block = load_block(root, entry)
        weights = block["weights"]
        parts.append(np.column_stack((block["raw"][:, :4] * weights[:, None], weights)))
        if tail_fractions:
            raw_parts.append(block["raw"][:, 2:4])
        stop = identity["stop"]
    values = np.concatenate(parts)
    intervals = [betting_interval(values[:, j], bound, alpha, grid_size)
                 for j, bound in tqdm(enumerate(bounds), total=len(bounds),
                                      desc="Weighted scalar confidence bounds", unit="mean",
                                      disable=len(values) < 8192, leave=False)]
    lower, upper = np.asarray(intervals).T
    tails = {}
    if tail_fractions:
        raw = np.concatenate(raw_parts)
        for column, name in enumerate(("recoil", "angular")):
            score, caps = tail_scores(raw[:, column], physical_bounds[column+2], tail_fractions)
            score *= values[:, 4, None]
            caps = np.nextafter(caps * bounds[4], np.inf)
            band_intervals = np.asarray([betting_interval(score[:, j], cap, alpha, grid_size)
                for j, cap in tqdm(enumerate(caps), total=len(caps), desc=f"{name} tail bounds",
                                   unit="band", disable=len(raw) < 8192, leave=False)])
            lo, hi = band_intervals.sum(axis=0)
            index = column + 2
            lower[index], upper[index] = max(lower[index], lo), min(upper[index], hi)
            if lower[index] > upper[index]:
                raise ValueError("Inconsistent simultaneous whole-score and tail intervals")
            tails[name] = {"fractions": list(tail_fractions), "physical_support": float(physical_bounds[index]),
                "weighted_band_support": caps.tolist(), "means": score.mean(axis=0).tolist(),
                "intervals": band_intervals.tolist(), "nonzero_histories": np.count_nonzero(score, axis=0).tolist(),
                "scope": "Unconditional weighted score contributions; all bands, including unseen tails"}
    return lower, upper, tails


def feature_rows(raw, events, weights, edges):
    """Each row is one independent history, including all its correlated events."""
    n = len(raw)
    weights = np.asarray(weights)
    na, nb = len(edges[0]) - 1, len(edges[1]) - 1
    shape = (n, 2 * na * nb + na + nb)
    if len(events):
        # Columns: history, H/O, angular transport/2, recoil/E_entry, E_in/E_entry,
        #          incoming polar cosine, incoming azimuth, scattering azimuth.
        a = np.clip(np.searchsorted(edges[0], events[:, 2], side="right") - 1, 0, na - 1)
        b = np.clip(np.searchsorted(edges[1], events[:, 3], side="right") - 1, 0, nb - 1)
        columns = np.r_[(events[:, 1].astype(int) * na + a) * nb + b,
                        2 * na * nb + a, 2 * na * nb + na + b]
        rows = np.tile(events[:, 0].astype(int), 3)
        counts = sparse.coo_matrix((np.ones(len(rows)), (rows, columns)), shape=shape).tocsr()
    else:
        counts = sparse.csr_matrix(shape)
    return sparse.hstack((sparse.csr_matrix(raw), counts), format="csr").multiply(weights[:, None]).tocsr(), counts


def grid_edges(training_events, bins):
    # Training is independent of certification. The finite refinement family is
    # frozen before validation; no acceptance grid is fitted to that validation.
    result = []
    for col in (2, 3):
        values = training_events[:, col] if len(training_events) else np.array([0., 1.])
        weights = training_events[:, 8] if len(training_events) and training_events.shape[1] == 9 else np.ones(len(values))
        order = np.argsort(values)
        cumulative = np.cumsum(weights[order])
        quantiles = np.interp(np.linspace(0, 1, bins + 1) * cumulative[-1], cumulative, values[order])
        result.append(np.unique(np.r_[0., quantiles, 1.]).tolist())
    return result


def fit_proposal(points, base, density, raw, seed_points, policy, *, costs=None, events=None, tail_support=None):
    """Train an iid mixture for estimated error reduction per CPU-second.

    For generating densities g_i, scalar ratio influences u_ik give the pilot
    criterion V_k(alpha) = mean[u_ik**2/(g_i*q_alpha(x_i))]. Component CPU
    costs c_j are self-normalized pilot estimates using q_j/g_i. Minimize
    (sum alpha_j*c_j) * max_k V_k. With beta_j = alpha_j*c_j/sum(alpha*c),
    this is a convex epigraph in beta for frozen pilot influences and costs;
    the defensive alpha floor is also a linear constraint in beta.

    The optional sixth objective is the second moment of the per-history L1
    categorical ratio influence, scaled by twice the TV allowance. It guides
    distribution sampling; it is NOT an exact TV error bound or a proof of
    globally TV-optimal sampling. Selected boxes, ratios and CPU costs are
    fitted from training, so their predicted gain requires independent tests.
    He & Owen (2014), https://arxiv.org/html/1411.3954, supports mixture
    optimization; the cost reparameterization above is stated explicitly.
    Old validation observations retain their original proposal and weights.
    """
    points, base, density, raw = map(lambda x: np.asarray(x, float), (points, base, density, raw))
    n = len(raw)
    if (not n or points.ndim != 2 or len(points) != n or raw.ndim != 2 or raw.shape[1] < 4
            or base.shape != (n,) or density.shape != (n,)
            or any(not np.isfinite(x).all() for x in (points, base, density, raw))
            or np.any(base <= 0) or np.any(density <= 0)):
        raise ValueError("Invalid proposal-training observations")
    costs = np.ones(n) if costs is None else np.asarray(costs, float)
    if costs.shape != (n,) or not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError("Training CPU costs must be finite and positive")
    mean = np.mean(raw / density[:, None], axis=0)
    ratios = np.array([mean[a] / mean[b] if mean[b] > 0 else 0. for a, b in RATIOS])
    scales = np.where(ratios > 0, policy.statistical_relative_tolerance * ratios, 1.)
    residual = np.column_stack([(raw[:, a] - r * raw[:, b]) / (scales[j] * max(mean[b], 1e-300))
                                for j, ((a, b), r) in enumerate(zip(RATIOS, ratios))])
    tail_centres = []
    if tail_support is not None:
        # Each band is scored against the TOTAL observable's error allowance;
        # we do not demand arbitrary relative accuracy in vanishing channels.
        for column in (2, 3):
            scores, _ = tail_scores(raw[:, column], tail_support[column], policy.tail_score_fractions)
            band_means = np.mean(scores / density[:, None], axis=0)
            denominator = max(mean[0], 1e-300)
            scale = policy.statistical_relative_tolerance * max(mean[column], 1e-300)
            influence = (scores - raw[:, 0, None] * band_means[None, :] / denominator) / scale
            residual = np.column_stack((residual, influence))
            # At least one entrance seed per observed band, rather than only
            # the largest unstratified histories. The pilot selects seeds only.
            for j in np.flatnonzero(band_means > 0):
                index = int(np.argmax(influence[:, j] ** 2 / density))
                tail_centres.append(points[index].tolist())
    use_tv = policy.output_scope == "joint_distribution" and events is not None and len(events) > 0 and mean[1] > 0
    if use_tv:
        events = np.asarray(events, float)
        if (events.ndim != 2 or events.shape[1] < 8 or not np.isfinite(events).all()
                or np.any(events[:, 0] != events[:, 0].astype(int))
                or np.any(events[:, 0] < 0) or np.any(events[:, 0] >= n)):
            raise ValueError("Invalid training-event history indices")
        rows = events[:, 0].astype(int)
        if not np.array_equal(np.bincount(rows, minlength=n), raw[:, 1]):
            raise ValueError("Training events do not match complete trajectory counts")
        weighted_events = np.column_stack((events[:, :8], 1 / density[rows]))
        edges = grid_edges(weighted_events, policy.base_bins)
        size = 2 * (len(edges[0]) - 1) * (len(edges[1]) - 1)
        counts = feature_rows(raw, events, np.ones(n), edges)[1][:, :size].tocsr()
        probability = np.asarray(counts.T @ (1 / density)).ravel() / (n * mean[1])
        # Sparse identity: ||C_i - N_i*p||_1 = N_i plus occupied-cell changes.
        coo = counts.tocoo()
        expected = raw[coo.row, 1] * probability[coo.col]
        norm = raw[:, 1].copy()
        np.add.at(norm, coo.row, np.abs(coo.data - expected) - expected)
        residual = np.column_stack((residual, np.maximum(norm, 0) /
                                    (2 * policy.distribution_tv_tolerance * mean[1])))
    if not np.isfinite(residual).all():
        raise ValueError("Nonfinite tolerance-normalized training influence")
    magnitude = np.max(np.abs(residual))
    if magnitude == 0:
        return {"boxes": [], "alpha": [1.]}
    residual = residual / magnitude  # Common rescaling leaves every optimizer unchanged.
    score = np.max(residual * residual, axis=1)
    ranked = np.lexsort((np.arange(n), -score / density))[:policy.local_centres]
    centres = list(seed_points) + tail_centres + [points[i].tolist() for i in ranked]
    boxes = proposal_boxes(centres, policy)
    if not boxes or not np.any(score > 0):
        return {"boxes": [], "alpha": [1.]}
    densities = np.column_stack((base, box_density(points, boxes)))
    importance = densities / density[:, None]
    mass = importance.sum(0)
    # Unsupported components cannot be evaluated from this pilot. A selected
    # component's huge empirical mass must not masquerade as a huge CPU cost.
    keep = mass > 0
    boxes = [box for box, retained in zip(boxes, keep[1:]) if retained]
    densities, importance, mass = densities[:, keep], importance[:, keep], mass[keep]
    if not boxes:
        return {"boxes": [], "alpha": [1.]}
    component_cost = (costs @ importance) / mass
    component_cost /= component_cost[0]
    transformed = densities / component_cost
    squared = residual ** 2 / density[:, None]
    baseline_value = np.max(np.mean(squared / base[:, None], axis=0))
    squared /= baseline_value

    def objective(beta):
        q = transformed @ beta
        values = np.mean(squared / q[:, None], axis=0)
        gradients = -(squared / q[:, None] ** 2).T @ transformed / n
        return values, gradients

    floor = policy.defensive_fraction
    alpha = np.r_[max(.5, floor), np.full(len(boxes), (1 - max(.5, floor)) / len(boxes))]
    beta = alpha * component_cost; beta /= beta.sum()
    initial = np.r_[beta, max(objective(beta)[0])]
    constraint = (np.r_[1., np.zeros(len(boxes))] - floor) / component_cost
    result = minimize(lambda x: x[-1], initial, jac=lambda x: np.r_[np.zeros(len(beta)), 1.],
        method="SLSQP", bounds=[(0., 1.)] * len(beta) + [(0., None)],
        constraints=[{"type": "eq", "fun": lambda x: x[:-1].sum() - 1,
                      "jac": lambda x: np.r_[np.ones(len(beta)), 0.]},
                     {"type": "ineq", "fun": lambda x: constraint @ x[:-1],
                      "jac": lambda x: np.r_[constraint, 0.]},
                     {"type": "ineq", "fun": lambda x: x[-1] - objective(x[:-1])[0],
                      "jac": lambda x: np.column_stack((-objective(x[:-1])[1], np.ones(residual.shape[1])))}],
        options={"maxiter": 100, "ftol": 1e-8})
    candidates = [np.r_[1., np.zeros(len(boxes))], alpha]
    if result.success and np.isfinite(result.x).all():
        trial = np.maximum(result.x[:-1], 0.) / component_cost
        trial /= trial.sum()
        if trial[0] >= floor - 1e-8:
            if trial[0] < floor:
                trial[1:] *= (1 - floor) / trial[1:].sum(); trial[0] = floor
            candidates.append(trial)

    def value(candidate):
        return float((candidate @ component_cost) *
                     max(np.mean(squared / (densities @ candidate)[:, None], axis=0)))

    chosen = min(candidates, key=value)
    if chosen[0] == 1 or value(chosen) >= 1:
        return {"boxes": [], "alpha": [1.]}
    active = chosen[1:] > 0
    return {"boxes": [box for box, retained in zip(boxes, active) if retained],
            "alpha": np.r_[chosen[0], chosen[1:][active]].tolist(),
            "fit": {"criterion": "CPU cost times worst scalar influence variance / joint-TV L1 second-moment surrogate",
                    "joint_tv_surrogate_included": bool(use_tv), "estimated_gain": 1 / value(chosen),
                    "tail_score_objectives_included": tail_support is not None,
                    "independent_validation_required": True}}


def execute_block(task):
    root, manifest, case, identity = task
    if identity["phase"] == "verification":
        return execute_verification(root, manifest, case, identity)
    policy = Policy(**manifest["policy"])
    transport = get_transport(root, manifest, case)
    proposal = identity["proposal"]
    points, raw, weights, densities, bases, events, costs = [], [], [], [], [], [], []
    segments = []
    window_delta = []
    original_table = transport.kernels
    original_moment_cache = transport._moment_cache
    historical = []
    if identity["phase"] == "replay":
        # Historical replay must use its original kernel. It guides proposals
        # only and never contributes independent production exposure.
        transport.kernels = original_table.table
        transport._moment_cache = {}
    try:
        for local, index in enumerate(range(identity["start"], identity["stop"])):
            seed = [manifest["seed"], *np.frombuffer(bytes.fromhex(digest([identity["case"], identity["phase"],
                    identity["round"], index])), dtype="<u4").astype(int).tolist()]
            rng = np.random.default_rng(np.random.SeedSequence(seed))
            started = time.process_time()
            if identity["phase"] == "replay":
                request = manifest["historical_batch_replay"]; cfg = request["configuration"]
                runner = importlib.import_module("simulate_nlh_hard_collisions")
                if runner.TRAJECTORY_IMPLEMENTATION_VERSION != cfg["trajectory_implementation_version"]:
                    raise ValueError("Historical runtime version mismatch")
                runner._WORKER_TRANSPORT, runner._WORKER_STRUCTURE = transport, transport.structure
                _, result = runner._run_one((request["batch"]["trajectory_start"] + index, cfg["seed"], cfg["projectile"],
                    cfg["projectile_energy_ev"], cfg["path_length_angstrom"], cfg["fixed_direction"], cfg["max_collisions"],
                    cfg["control_variate"], cfg["initial_condition_sampling"], cfg["tube_mixture_fraction"]))
                weight = result.importance_sampling.target_over_proposal_weight if result.importance_sampling else 1.
                density = 1 / weight; base = density
                point = (np.asarray(result.initial_position_angstrom) @ np.linalg.inv(transport.structure.lattice_angstrom)) % 1.
                if cfg["fixed_direction"] is None:
                    point = np.r_[point, isotropic_point(result.initial_direction)]
                historical.append(np.array([result.traveled_path_length_angstrom, len(result.events), result.recoil_energy_ev,
                    sum(1. - math.cos(e.theta_projectile_lab_rad) for e in result.events)]) * weight)
            else:
                point, position, direction, density, base = sample_initial(transport, manifest, case, proposal, rng)
                trace_state = rng.bit_generator.state
                result = transport.trace(manifest["projectile"], case["energy_ev"], position, direction,
                                         manifest["path_length_angstrom"], rng=rng, max_collisions=policy.max_collisions)
            if result.termination == "maximum_collisions":
                raise RuntimeError("Collision-count cap reached; no truncated history can qualify")
            row = [result.traveled_path_length_angstrom, len(result.events), result.recoil_energy_ev,
                   sum(2 * math.sin(e.theta_projectile_lab_rad / 2) ** 2 for e in result.events),
                   result.ambiguous_event_count, float(result.termination != "path_complete")]
            if (identity["phase"] == "validation" or identity.get("calibration_geometry_check")) and index < policy.batch_size:
                transport.search_window_angstrom *= 2
                try:
                    rng.bit_generator.state = trace_state
                    check = transport.trace(manifest["projectile"], case["energy_ev"], position, direction,
                        manifest["path_length_angstrom"], rng=rng, max_collisions=policy.max_collisions)
                finally:
                    transport.search_window_angstrom /= 2
                other = [check.traveled_path_length_angstrom, len(check.events), check.recoil_energy_ev,
                         sum(2 * math.sin(e.theta_projectile_lab_rad / 2) ** 2 for e in check.events)]
                delta = np.abs(np.array(row[:4]) - other)
                if np.any(delta > 1e-10 * np.maximum(np.abs(row[:4]), 1e-20)):
                    raise RuntimeError("Search-window replay changed a trajectory; geometry requires investigation")
                window_delta.append(float(delta.max()))
            if not np.isfinite(row).all() or row[2] > case["energy_ev"] * (1 + 1e-12):
                raise RuntimeError("Nonfinite outcome or violated recoil-energy bound")
            previous_distance = 0.
            for event in result.events:
                if abs(event.projectile_energy_in_ev - event.projectile_energy_out_ev - event.recoil_energy_ev) > 1e-10 * event.projectile_energy_in_ev:
                    raise RuntimeError("Binary energy conservation failed")
                incoming = np.asarray(event.direction_in)
                segments.append([local, previous_distance, event.path_distance_angstrom,
                                 event.projectile_energy_in_ev / case["energy_ev"], *incoming,
                                 0 if event.target == "H" else 1, 1])
                previous_distance = event.path_distance_angstrom
                trial = np.array([1., 0., 0.]) if abs(incoming[0]) < .8 else np.array([0., 1., 0.])
                first = np.cross(incoming, trial); first /= np.linalg.norm(first)
                second = np.cross(incoming, first)
                outgoing = np.asarray(event.direction_out)
                azimuth = math.atan2(outgoing @ second, outgoing @ first) % (2 * math.pi)
                events.append([local, 0 if event.target == "H" else 1,
                               math.sin(event.theta_projectile_lab_rad / 2) ** 2,
                               event.recoil_energy_ev / case["energy_ev"],
                               event.projectile_energy_in_ev / case["energy_ev"],
                               incoming[2], math.atan2(incoming[1], incoming[0]), azimuth])
            segments.append([local, previous_distance, result.traveled_path_length_angstrom,
                             result.final_energy_ev / case["energy_ev"], *result.final_direction, -1, 0])
            raw.append(row); points.append(point); weights.append(1 / density)
            densities.append(density); bases.append(base); costs.append(time.process_time() - started)
    finally:
        transport.kernels = original_table
        transport._moment_cache = original_moment_cache
    return {"raw": np.asarray(raw), "points": np.asarray(points), "weights": np.asarray(weights),
            "density": np.asarray(densities), "base": np.asarray(bases), "cpu_seconds": np.asarray(costs),
            "events": np.asarray(events).reshape(-1, 8), "window_delta": np.asarray(window_delta),
            "segments": np.asarray(segments).reshape(-1, 9),
            "historical": np.asarray(historical).reshape(-1, 4)}


def cohort_key(case):
    return f"{case['phase']}_{case['round']}"


def cohort(case):
    return refresh_cohort(case["cohorts"].setdefault(cohort_key(case), {"blocks": [], "count": 0, "prefix": 0}))


def checked_ranges(ranges):
    """Return sorted, disjoint half-open history ranges; gaps are legitimate."""
    ordered = sorted(ranges)
    previous_stop = 0
    for start, stop in ordered:
        if type(start) is not int or type(stop) is not int or start < 0 or stop <= start:
            raise ValueError("Invalid cohort history range")
        if start < previous_stop:
            raise ValueError("Overlapping cohort checkpoint or pending range")
        previous_stop = stop
    return ordered


def refresh_cohort(c):
    """Keep committed count separate from the contiguous, assessable prefix.

    Workers may commit out of order. A missing earlier block is unfinished work,
    not a reason to discard a later checkpoint or replay its completed histories.
    """
    ranges = checked_ranges((entry["identity"]["start"], entry["identity"]["stop"])
                            for entry in c["blocks"])
    c["blocks"].sort(key=lambda entry: entry["identity"]["start"])
    c["count"] = sum(stop - start for start, stop in ranges)
    prefix = 0
    for start, stop in ranges:
        if start != prefix:
            break
        prefix = stop
    c["prefix"] = prefix
    return c


def missing_ranges(c, target, pending=()):
    """Return uncommitted, unreserved ranges in [0, target), in seed order.

    The caller splits these gaps into bounded tasks and passes their ranges as
    pending while futures are live. Pending work is not persisted as evidence;
    after interruption only checksum-committed blocks suppress resubmission.
    """
    if type(target) is not int or target < 0:
        raise ValueError("The cohort target must be a nonnegative integer")
    refresh_cohort(c)
    occupied = checked_ranges([(entry["identity"]["start"], entry["identity"]["stop"])
                               for entry in c["blocks"]] + list(pending))
    cursor, gaps = 0, []
    for start, stop in occupied:
        if cursor >= target:
            break
        if start > cursor:
            gaps.append((cursor, min(start, target)))
        cursor = stop
    if cursor < target:
        gaps.append((cursor, target))
    return gaps


def load_block(root, entry):
    path = Path(root) / entry["path"]
    if file_hash(path) != entry["sha256"]:
        raise ValueError(f"Corrupt checkpoint: {path}")
    with np.load(path, allow_pickle=False) as block:
        return {name: block[name] for name in block.files}


def save_block(root, identity, result):
    name = digest(identity)
    path = Path(root) / "blocks" / (name + ".npz")
    path.parent.mkdir(exist_ok=True)
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, **result)
        f.flush(); os.fsync(f.fileno())
    path.with_suffix(".tmp").replace(path)
    entry = {"identity": identity, "path": str(path.relative_to(root)), "sha256": file_hash(path)}
    # The receipt is the commit point. A restart adopts a valid orphan receipt;
    # an interrupted, receipt-less block is replayed with exactly the same seed.
    receipt = path.with_suffix(".json")
    atomic_json(receipt, entry)
    return compact_entry(entry, file_hash(receipt))


def compact_entry(entry, receipt_hash):
    slim = {k: v for k, v in entry["identity"].items() if k != "proposal"}
    slim["proposal_sha256"] = digest(entry["identity"]["proposal"])
    return {**entry, "identity": slim, "receipt_sha256": receipt_hash}


def full_identity(root, entry):
    path = (Path(root) / entry["path"]).with_suffix(".json")
    if file_hash(path) != entry["receipt_sha256"]:
        raise ValueError("Checkpoint receipt changed")
    return read(path)["identity"]


def training_data(root, case):
    names = sorted((key for key in case["cohorts"] if key.startswith("training_")),
                   key=lambda key: int(key.partition("_")[2]))
    selected = [entry for name in names for entry in
                sorted(case["cohorts"][name]["blocks"], key=lambda entry: entry["identity"]["start"])]
    parts = [load_block(root, entry) for entry in selected]
    data = {name: np.concatenate([p[name] for p in parts]) for name in
            ("points", "base", "density", "raw", "weights", "cpu_seconds")}
    # Include each history's likelihood weight when choosing physical, rather
    # than proposal-weighted, event quantiles. This is still training only.
    events, offset = [], 0
    for part in parts:
        local = part["events"][:, 0].astype(int)
        if (np.any(part["events"][:, 0] != local) or np.any(local < 0)
                or np.any(local >= len(part["raw"]))):
            raise ValueError("Invalid block-local training-event history index")
        rows = np.column_stack((part["events"], part["weights"][local]))
        rows[:, 0] += offset
        events.append(rows)
        offset += len(part["raw"])
    data["events"] = np.concatenate(events)
    return data


def seed_points(manifest, case):
    record = manifest["structures"][case["structure"]]
    direction = manifest["directions"][case["direction"]]["vector"]
    result = []
    for replay in manifest["historical_seeds"]:
        trajectory = replay["trajectory"]
        # Historical points guide sampling, not acceptance or pooled exposure.
        if trajectory["initial_energy_ev"] != case["energy_ev"]:
            continue
        cfg_hash = replay["identity"]["configuration_sha256"]
        if not cfg_hash or trajectory["projectile"] != manifest["projectile"]:
            continue
        # Seeds may come from another structure without affecting unbiasedness,
        # but restricting their labels avoids uninformative transferred centres.
        if record["name"] not in replay["identity"]["case"].split("/"):
            continue
        if direction is not None and not np.allclose(direction, trajectory["initial_direction"], atol=1e-12, rtol=0):
            continue
        point = (np.asarray(trajectory["initial_position_angstrom"]) @ np.linalg.inv(record["lattice_angstrom"])) % 1.
        result.append(point.tolist() + (isotropic_point(trajectory["initial_direction"]) if direction is None else []))
    return result


def aggregate(root, entries, edges):
    total = total2 = None
    n = 0
    cpu = 0.
    batch_records = []
    for entry in entries:
        block = load_block(root, entry)
        features, _ = feature_rows(block["raw"], block["events"], block["weights"], edges)
        if total is None:
            total, total2 = np.zeros(features.shape[1]), np.zeros(features.shape[1])
        # Do not materialize all empty grid cells for every small checkpoint.
        # CSR has already combined correlated events inside each history/bin.
        np.add.at(total, features.indices, features.data)
        np.add.at(total2, features.indices, features.data ** 2)
        count = len(block["raw"])
        n += count; cpu += float(block["cpu_seconds"].sum())
        weighted = block["raw"][:, :4] * block["weights"][:, None]
        batch_records.append((count, weighted.sum(0), (weighted ** 2).sum(0), float(block["weights"].sum()), float((block["weights"] ** 2).sum())))
    return n, total, total2, cpu, batch_records


def frozen_design(case):
    return {k: case[k] for k in ("structure", "direction", "energy_ev", "round", "proposal", "kernel_mode",
                                "kernel", "statistical_relative_tolerance", "grids")}


def verify_design(root, case):
    design = read(Path(root) / case["design_path"])
    if digest(design) != case["design_sha256"] or design != frozen_design(case):
        raise ValueError("Changed frozen statistical/proposal design")
    for name, c in case["cohorts"].items():
        if name.startswith(("baseline_", "validation_")):
            for entry in c["blocks"]:
                if entry["identity"].get("design_sha256") != case["design_sha256"]:
                    raise ValueError("A validation checkpoint belongs to another design")


def production_cohort(case):
    """Select independent evidence; never relabel a frozen proposal or receipt.

    Both baseline and targeted cohorts already receive simultaneous confidence
    coverage. Selection may use their pilot costs without an unbudgeted test.
    """
    selected = case.get("selected_cohort", "validation")
    if selected not in ("baseline", "validation"):
        raise ValueError("Unknown production evidence cohort")
    return selected


def cohort_weight_bound(case, phase=None):
    phase = production_cohort(case) if phase is None else phase
    return 2. if phase == "baseline" else 2. / case["proposal"]["alpha"][0]


def assess(root, manifest, case, phase=None):
    verify_design(root, case)
    phase = production_cohort(case) if phase is None else phase
    policy = Policy(**manifest["policy"])
    edges = ([[0., 1.], [0., 1.]] if policy.output_scope == "scalar"
             else case["grids"][case["level"]])
    entries = case["cohorts"].get(f"{phase}_{case['round']}", {}).get("blocks", [])
    if not entries:
        return None
    n, total, total2, cpu, batches = aggregate(root, entries, edges)
    # Histogram family: one quarter of global alpha over cases, attempts,
    # levels, scheduled looks and both cohorts. Scalars spend another quarter;
    # interpolation histograms spend the remaining half. Empty bins count.
    looks = 2 + math.ceil(math.log(max(1, policy.max_case_histories / policy.initial_validation_histories),
                                  1 + policy.sampling_growth if policy.output_scope == "scalar" else 2))
    alpha = ((1 - policy.confidence) / (2 * policy.max_cases * policy.max_proposal_rounds *
              (policy.max_bin_level + 1) * looks * 4 * (len(total) + 1)))
    weight_max = cohort_weight_bound(case, phase)
    physics = assessment_physics(root, manifest, case["energy_ev"]) if "tail_support" in manifest else manifest
    physical_bounds = physical_scalar_bounds(physics, case["energy_ev"])
    upper_bound = np.r_[physical_bounds, physical_bounds[1], 1.,
                       np.full(len(total) - 6, physical_bounds[1])] * weight_max
    lower, upper = empirical_interval(n, total, total2, upper_bound, alpha)
    scalar_alpha = scalar_confidence_alpha(policy)
    scalar_lower, scalar_upper, tails = scalar_betting_bounds(
        root, entries, np.r_[upper_bound[:4], weight_max], scalar_alpha, policy.betting_grid_size,
        physical_bounds=physical_bounds, tail_fractions=policy.tail_score_fractions)
    lower[:4], upper[:4] = scalar_lower[:4], scalar_upper[:4]
    weight_low, weight_high = scalar_lower[4], scalar_upper[4]
    mean = total / n
    scalar = {}
    rse = scalar_standard_errors(root, entries, policy.statistical_relative_tolerance)
    ratios = []
    for name, (a, b) in zip(OBSERVABLES, RATIOS):
        interval = ratio_interval(lower, upper, a, b)
        estimate = float(mean[a] / mean[b]) if mean[b] > 0 else 0.
        ratios.append(estimate)
        scalar[name] = {"estimate": estimate, "interval": interval,
                        **relative_precision(estimate, interval, policy.statistical_relative_tolerance)}
        scalar[name]["bounded_interval_passes"] = scalar[name]["passes"]
        scalar[name].update(rse[name])
        if policy.scalar_precision_mode == "relative_standard_error":
            scalar[name]["passes"] = rse[name]["standard_error_passes"]
    na, nb = len(edges[0]) - 1, len(edges[1]) - 1
    if lower[1] > 0:
        probability_lower = lower[6:] / upper[1]
        probability_upper = np.minimum(1., upper[6:] / lower[1])
        marginal_upper = probability_upper[2 * na * nb:]
        # For a joint CDF at arbitrary coordinates, the unrepresented mass is
        # bounded by the sum of the two intersected marginal strips.
        grid_bound = float(max(marginal_upper[:na]) + max(marginal_upper[na:]))
        probability_width = float(np.max(probability_upper[:2 * na * nb] - probability_lower[:2 * na * nb]))
        tv_bound = total_variation_bound(total[6:6 + 2 * na * nb] / total[1],
                                        probability_lower[:2 * na * nb], probability_upper[:2 * na * nb])
        recoil_compression = float(case["energy_ev"] * np.dot(marginal_upper[na:], np.diff(edges[1])) / 2)
        angular_compression = float(np.dot(marginal_upper[:na], np.diff(edges[0])))
    else:
        grid_bound, probability_width, tv_bound = 2., 1., 1.
        recoil_compression, angular_compression = case["energy_ev"], 2.
    recoil_scale = scalar[OBSERVABLES[3]]["interval"][0]
    angular_scale = scalar[OBSERVABLES[4]]["interval"][0]
    compression_pass = (recoil_compression <= policy.interpolation_relative_tolerance * recoil_scale and
                        angular_compression <= policy.interpolation_relative_tolerance * angular_scale)
    concentration = [max((b[2][j] for b in batches), default=0.) / max(total2[j], 1e-300) for j in range(4)]
    return {"n": n, "cpu_seconds": cpu, "scalar": scalar,
            "output_scope": policy.output_scope,
            "assessment_bin_edges": edges,
            "scalar_pass": all(v["passes"] for v in scalar.values()),
            "distribution_pass": tv_bound <= policy.distribution_tv_tolerance and grid_bound <= policy.grid_cdf_tolerance and compression_pass,
            "scalar_precision_mode": policy.scalar_precision_mode,
            "scalar_guarantee": "Estimated relative standard error; not a simultaneous or stopping-valid 95% guarantee"
                if policy.scalar_precision_mode == "relative_standard_error" else "Simultaneous bounded intervals",
            "tail_evidence": tails,
            "physical_history_support": physical_bounds.tolist(),
            "physical_support_provenance": physics.get("tail_support"),
            "statistical_pass": all(v["passes"] for v in scalar.values()) and
                                tv_bound <= policy.distribution_tv_tolerance,
            "joint_distribution_total_variation_upper_bound": tv_bound,
            "allowed_statistical_total_variation": policy.distribution_tv_tolerance,
            "distribution_scope": "H/O identity and joint polar-angle/recoil bin masses; not an unbinned density or azimuth law",
            "joint_bin_probability_maximum_width": probability_width,
            "grid_unresolved_mass_upper_bound": grid_bound,
            "grid_pass": grid_bound <= policy.grid_cdf_tolerance and compression_pass,
            "bin_compression_moment_error_bounds": {"mean_recoil_ev": recoil_compression,
                                                       "mean_one_minus_cosine": angular_compression,
                                                       "passes": compression_pass},
            "importance_normalization_interval": [float(weight_low), float(weight_high)],
            "largest_batch_raw_second_moment_shares": concentration,
            "ambiguity_fraction": float(mean[4] / mean[1]) if mean[1] else 0.,
            "floor_termination_probability_estimate": float(mean[5]),
            "individual_alpha": alpha, "scalar_alpha": scalar_alpha,
            "confidence_method": "time-uniform whole-history and tail-score grid betting; empirical Bernstein histograms; simultaneous family coverage",
            "means": mean[:6].tolist(), "scalar_lower": lower[:6].tolist(), "scalar_upper": upper[:6].tolist()}


def scalar_standard_errors(root, entries, tolerance):
    """Trajectory-clustered delta-method SE of ratios of weighted means.

    Calibration histories are excluded by the caller's independent cohort.
    Use complete histories, not correlated individual collisions. Two passes
    compute centred influences without subtracting nearly equal raw moments.
    These are estimated SEs, not finite-sample or optional-stopping guarantees.
    """
    n, sums = 0, np.zeros(4)
    for entry in entries:
        b = load_block(root, entry)
        x = b["raw"][:, :4]*b["weights"][:, None]
        n += len(x); sums += x.sum(axis=0)
    ratios = np.array([sums[a]/sums[b] if sums[b] > 0 else 0. for a, b in RATIOS])
    squared = np.zeros(len(RATIOS))
    for entry in entries:
        block = load_block(root, entry)
        x = block["raw"][:, :4]*block["weights"][:, None]
        for j, (a, b) in enumerate(RATIOS):
            squared[j] += np.dot(x[:, a]-ratios[j]*x[:, b], x[:, a]-ratios[j]*x[:, b])
    result = {}
    for j, (name, (_, b)) in enumerate(zip(OBSERVABLES, RATIOS)):
        se = float(math.sqrt(squared[j]*n/(n-1))/sums[b]) if n > 1 and sums[b] > 0 else None
        relative = float(se/abs(ratios[j])) if se is not None and ratios[j] != 0 else None
        result[name] = {"standard_error": se, "relative_standard_error": relative,
            "standard_error_passes": relative is not None and 0 < relative <= tolerance,
            "zero_variance_requires_review": se == 0.,
            "allowed_relative_standard_error": tolerance}
    return result


def efficiency_diagnostic(root, case):
    """Pilot variance-cost estimates, not confidence guarantees or forecasts."""
    result = {}
    for phase in ("baseline", "validation"):
        entries = case["cohorts"].get(f"{phase}_{case['round']}", {}).get("blocks", [])
        n = 0; total = np.zeros(4); second = np.zeros((4, 4)); seconds = 0.
        for entry in entries:
            block = load_block(root, entry)
            values = block["raw"][:, :4] * block["weights"][:, None]
            total += values.sum(0); second += values.T @ values
            n += len(values); seconds += float(block["cpu_seconds"].sum())
        mean = total / n
        covariance = (second - np.outer(total, total) / n) / (n - 1)
        costs = []
        for a, b in RATIOS:
            if mean[b] <= 0:
                return {"available": False}
            r = mean[a] / mean[b]
            if r <= 0:
                return {"available": False, "reason": "An observable is unresolved at zero"}
            variance = max(0., covariance[a, a] + r * r * covariance[b, b] - 2 * r * covariance[a, b]) / mean[b] ** 2
            costs.append(variance / r ** 2 * seconds / n)
        result[phase] = costs
    base, targeted = max(result["baseline"]), max(result["validation"])
    return {"available": True, "normalized_variance_times_cpu_seconds": result,
            "estimated_limiting_observable_gain": base / targeted if targeted > 0 else None,
            "demonstrated_speedup": False,
            "scope": "Independent frozen pilot; a point estimate only, vulnerable to unresolved tails"}


def observations_compatible(first, second):
    for name in OBSERVABLES:
        a, b = first["scalar"][name]["interval"], second["scalar"][name]["interval"]
        if a[1] is not None and b[1] is not None and (a[1] < b[0] or b[1] < a[0]):
            return False
    return True


def range_budget_diagnostic(manifest, case, report, weight_bound=None):
    """Optimistic planning calculation, not a lower bound on unknown physics.

    Hold the pilot means fixed, grant zero sample variance and a generous scalar
    error budget, then ask whether the bounded-range penalty alone fits inside
    the allowed workload. Failure prevents futile automatic bulk sampling.
    """
    policy = Policy(**manifest["policy"])
    mean = np.array(report["means"][:4])
    weight_bound = 2 / case["proposal"]["alpha"][0] if weight_bound is None else weight_bound
    bounds = np.asarray(report.get("physical_history_support",
                        physical_scalar_bounds(manifest, case["energy_ev"]))) * weight_bound
    if weight_bound == 1. and mean[0] > 0:
        # Historical planning rows are full-path ratios. Their estimated
        # importance normalization need not equal one. The ideal unit-weight
        # comparison fixes exposure to the full path while retaining ratios.
        mean *= manifest["path_length_angstrom"] / mean[0]
        mean[0] = manifest["path_length_angstrom"]
    n = policy.max_case_histories
    alpha = (1 - policy.confidence) / (2 * policy.max_cases * 4)
    low, high = np.asarray([betting_interval([value], bound, alpha, policy.betting_grid_size,
                                            counts=np.array([n]))
                            for value, bound in zip(mean, bounds)]).T
    # Retained aggregate means have no band allocation. Only a fresh complete
    # cohort can supply one; never invent a benign distribution of tail mass.
    for j, name in ((2, "recoil"), (3, "angular")):
        tail = report.get("tail_evidence", {}).get(name)
        if tail is not None:
            intervals = [betting_interval([value], cap, alpha, policy.betting_grid_size,
                         counts=np.array([n])) for value, cap in
                         zip(tail["means"], tail["weighted_band_support"])]
            lo, hi = np.sum(intervals, axis=0)
            low[j], high[j] = max(low[j], lo), min(high[j], hi)
    precision = []
    for a, b in RATIOS:
        interval = ratio_interval(low, high, a, b)
        estimate = mean[a] / mean[b] if mean[b] > 0 else 0.
        precision.append(relative_precision(estimate, interval, policy.statistical_relative_tolerance))
    return {"sampling_at_pilot_means_can_fit_budget": all(p["passes"] for p in precision),
            "zero_variance_precision_at_case_limit": dict(zip(OBSERVABLES, precision)),
            "weight_bound_used": weight_bound,
            "confidence_method": "grid betting; repeated pilot means are optimistic planning only",
            "scope": "Optimistic planning at current means, not a proof about unsampled physics or a required-trajectory forecast"}


def compare_scalar_intervals(root):
    """Compare old/new bounds without rewriting a signed campaign or sampling.

    Historical aggregates do not determine betting capital. Constant sequences
    at retained means provide only an optimistic feasibility sensitivity, never
    a certificate or a measured speed-up on actual rare-event trajectories.
    """
    root = Path(root)
    manifest = read(root / "manifest.json")
    signed = {k: v for k, v in manifest.items() if k != "signature"}
    if digest(signed) != manifest["signature"]:
        raise ValueError("Historical campaign signature mismatch")
    policy = Policy(**manifest["policy"])
    policy.validate()
    # Read-only reconstruction of physical support for older signed pair
    # campaigns. Do not rewrite their manifest or pretend to possess raw tails.
    Table, _, _ = load_runtime(root)
    for name, sha in manifest["files"].items():
        if not Path(name).is_absolute() and file_hash(root/name) != sha:
            raise ValueError("Historical runtime changed before support comparison")
    table = Table(manifest["kernel_manifest"])
    if table.csv_sha256 != manifest["kernel_csv_sha256"]:
        raise ValueError("Historical pair kernels changed before comparison")
    kin = [table.pair_kinematics(manifest["projectile"], t, manifest["energies_ev"][0]) for t in ("H", "O")]
    candidate_manifest = {**manifest, "tail_support": {
        "projectile_mass_c2_ev": kin[0].projectile_mass_c2_ev,
        "target_mass_c2_ev": [k.target_mass_c2_ev for k in kin],
        "energy_floor_ev": max(manifest["pair_energies_ev"][t][0] for t in ("H", "O")),
        "model": "stationary_targets_nonenergizing_primary_sequential_binary"}}
    checks = []
    for saved in tqdm(manifest.get("historical_planning", []), desc="Scalar estimator comparisons", unit="case"):
        mean = np.asarray(saved["means"][:4])
        mean *= manifest["path_length_angstrom"] / mean[0]
        mean[0] = manifest["path_length_angstrom"]
        bounds = np.array([manifest["path_length_angstrom"], policy.max_collisions,
                           saved["energy_ev"], 2 * policy.max_collisions])
        n = policy.max_case_histories
        alpha = (1 - policy.confidence) / (2 * policy.max_cases * 4)
        old_low, old_high = empirical_interval(n, n * mean, n * mean ** 2, bounds, alpha)
        candidate = range_budget_diagnostic(manifest, {"energy_ev": saved["energy_ev"]}, saved, weight_bound=1.)
        physics = assessment_physics(root, candidate_manifest, saved["energy_ev"])
        physical = range_budget_diagnostic(physics, {"energy_ev": saved["energy_ev"]}, saved, weight_bound=1.)
        old = {name: relative_precision(mean[a] / mean[b], ratio_interval(old_low, old_high, a, b),
                                        policy.statistical_relative_tolerance)
               for name, (a, b) in zip(OBSERVABLES, RATIOS)}
        checks.append({"case": saved["case"], "source_report_sha256": saved["report_sha256"],
                       "empirical_bernstein": old, "betting": candidate["zero_variance_precision_at_case_limit"],
                       "physical_support": physical["zero_variance_precision_at_case_limit"],
                       "physical_history_bounds": physical_scalar_bounds(physics, saved["energy_ev"]).tolist()})
    failures = {method: sum(not all(v["passes"] for v in check[method].values()) for check in checks)
                for method in ("empirical_bernstein", "betting", "physical_support")}
    result = {"source_campaign_signature": manifest["signature"], "source_manifest_sha256": file_hash(root / "manifest.json"),
              "controller_sha256": file_hash(Path(__file__)), "candidate_schema": SCHEMA,
              "relative_tolerance": policy.statistical_relative_tolerance, "confidence": policy.confidence,
              "statistical_distribution_tv": policy.distribution_tv_tolerance,
              "numerical_pair_distribution_tv": policy.pair_distribution_tv_tolerance,
              "tested_cases": len(checks), "failed_scalar_preflights": failures, "checks": checks,
              "scope": "Constant sequences at retained ratios, full-path exposure normalization, zero variance, ideal unit weights; NOT intervals for historical data",
              "new_trajectories": 0, "historical_trajectories_recertified": 0,
              "tail_band_statistics_available": False,
              "legacy_count_cap_is_not_physical_support": True,
              "broad_sampling_authorized": False,
              "next_step": "Review tail-specific estimator/support if betting still fails; otherwise independent weighted pilots"}
    atomic_json(root / "scalar_estimator_comparison.json", result)
    return result


def feasibility(root, manifest=None, state=None):
    """Reuse retained means before scheduling even a fresh pilot.

    Weight bound one and zero variance are optimistic limits, not properties
    claimed for the actual proposal. Failure diagnoses this interval design,
    not bad trajectories or the impossibility of a better estimator.
    """
    root = Path(root)
    manifest = verify(root) if manifest is None else manifest
    state = read(root / "state.json") if state is None else state
    checks = {}
    for key, case in state["cases"].items():
        record = manifest["structures"][case["structure"]]
        direction = manifest["directions"][case["direction"]]["vector"]
        for saved in manifest.get("historical_planning", []):
            if saved["structure_sha256"] == record["sha256"] and saved["direction"] == direction and saved["energy_ev"] == case["energy_ev"]:
                checks[key] = {**range_budget_diagnostic(manifest, case, saved, weight_bound=1.),
                               "historical_case": saved["case"], "source_report_sha256": saved["report_sha256"],
                               "retained_trajectories": saved["trajectories"], "new_trajectories": 0}
    blocked = [k for k, v in checks.items() if not v["sampling_at_pilot_means_can_fit_budget"]]
    result = {"campaign_signature": manifest["signature"], "checks": checks,
              "status": "independent_tail_pilots_required" if manifest.get("tail_support") else
                        ("estimator_review_required" if blocked else "independent_pilots_required"),
              "tail_pilot_required": bool(manifest.get("tail_support")),
              "broad_sampling_authorized_by_feasibility": False, "blocked_cases": blocked,
              "reason": "Betting intervals remain too wide at retained means, even with zero variance and ideal weight bound one" if blocked else "Historical planning does not replace independent proposal validation",
              "scope": "Computational planning only. No existing history is discarded or relabelled statistically certified."}
    atomic_json(root / "feasibility.json", result)
    return result


def verification_identity(manifest, case, key):
    selected = production_cohort(case)
    entries = case["cohorts"][f"{selected}_{case['round']}"]["blocks"]
    entry = min(entries, key=lambda item: item["identity"]["start"])
    original = entry["identity"]
    count = original["stop"] - original["start"]
    extra = max(0, min(original["stop"], manifest["policy"]["batch_size"]) - original["start"]) if selected == "validation" else 0
    return {"campaign": manifest["signature"], "case": key, "phase": "verification", "round": case["round"],
            "start": 0, "stop": count, "propagations": count + extra, "entry": entry,
            "selected_cohort": selected, "design_sha256": case["design_sha256"]}


def verification_result(root, identity):
    """Load one atomic replay proof; it never represents independent samples."""
    path = Path(root) / "verifications" / (digest(identity) + ".json")
    if not path.exists():
        return None
    proof = read(path)
    result = proof.get("result", {})
    if (proof.get("identity") != identity or result.get("passes") is not True or
            result.get("block_sha256") != identity["entry"]["sha256"] or
            result.get("histories_replayed") != identity["stop"] - identity["start"] or
            result.get("propagations") != identity["propagations"]):
        raise ValueError("Invalid deterministic-verification proof")
    # The proof is reusable only while the original data and receipt remain intact.
    full_identity(root, identity["entry"])
    if file_hash(Path(root) / identity["entry"]["path"]) != identity["entry"]["sha256"]:
        raise ValueError("Verified checkpoint changed")
    return {**result, "proof_path": str(path.relative_to(root)), "proof_sha256": file_hash(path),
            "verification_identity_sha256": digest(identity)}


def execute_verification(root, manifest, case, identity):
    if identity["campaign"] != manifest["signature"]:
        raise ValueError("Verification task belongs to another campaign")
    existing = verification_result(root, identity)
    if existing is not None:
        return existing
    entry = identity["entry"]
    actual = execute_block((root, manifest, case, full_identity(root, entry)))
    expected = load_block(root, entry)
    if expected.keys() != actual.keys() or len(expected["raw"]) != identity["stop"] - identity["start"]:
        raise RuntimeError("Checkpoint replay has a different product shape")
    for name in expected:
        if name != "cpu_seconds" and not np.array_equal(expected[name], actual[name]):
            raise RuntimeError(f"Checkpoint replay differs in {name}")
    result = {"passes": True, "block_sha256": entry["sha256"], "histories_replayed": len(expected["raw"]),
              "propagations": identity["propagations"], "new_independent_samples": 0}
    atomic_json(Path(root) / "verifications" / (digest(identity) + ".json"), {"identity": identity, "result": result})
    return verification_result(root, identity)


def adopt_verification(state, case, result):
    key = result["verification_identity_sha256"]
    accounted = state.setdefault("verification_work", {})
    if key not in accounted:
        # Recover a worker proof committed before its controller reservation was
        # checkpointed. Account once; repeated resumes do not invent more work.
        accounted[key] = result["propagations"]
        state["reserved_histories"] += result["propagations"]
    case["replay"] = result
    case["phase"] = production_cohort(case)
    case["status"] = "qualified"


def advance_case(root, manifest, state, key):
    case = state["cases"][key]
    policy = Policy(**manifest["policy"])
    if case["status"] in ("qualified", "blocked") or not case.get("kernel", {}).get("passes"):
        return
    if case["phase"] == "verification":
        result = verification_result(root, verification_identity(manifest, case, key))
        if result is not None:
            adopt_verification(state, case, result)
        return
    current = cohort(case)
    if current["prefix"] < case["target"]:
        return
    if case["phase"] == "replay":
        sums = np.zeros(4); products = np.zeros((4, 4)); best = []
        expected = manifest["historical_batch_replay"]["batch"]["raw_statistics"]
        for entry in current["blocks"]:
            block = load_block(root, entry); values = block["historical"]
            sums += values.sum(0); products += values.T @ values
            score = np.zeros(len(values))
            for name, (a, b) in zip(OBSERVABLES, RATIOS):
                r = expected[name]["numerator_sum"] / expected[name]["denominator_sum"]
                scale = policy.statistical_relative_tolerance * r if r > 0 else 1.
                score = np.maximum(score, ((values[:, a] - r * values[:, b]) / scale) ** 2)
            best.extend((float(score[i]), block["points"][i].tolist()) for i in np.argsort(score)[-policy.local_centres:])
            best = sorted(best, key=lambda row: row[0], reverse=True)[:policy.local_centres]
        for name, (a, b) in zip(OBSERVABLES, RATIOS):
            reconstructed = [sums[a], sums[b], products[a, a], products[b, b], products[a, b]]
            reference = [expected[name][field] for field in ("numerator_sum", "denominator_sum", "numerator_square_sum",
                                                            "denominator_square_sum", "numerator_denominator_sum")]
            if not np.allclose(reconstructed, reference, rtol=1e-9, atol=0):
                case.update(status="blocked", reason="Historical batch replay does not reproduce its raw sufficient statistics")
                return
        case["historical_replay"] = {"passes": True, "trajectories": current["count"], "new_independent_samples": 0,
                                     "batch_sha256": manifest["historical_batch_replay"]["batch_sha256"],
                                     "centres": [point for _, point in best]}
        boxes = proposal_boxes(case["historical_replay"]["centres"], policy)
        case["proposal"] = {"boxes": boxes, "alpha": [.5] + [.5 / len(boxes)] * len(boxes)}
        case.update(phase="training", target=policy.training_histories)
    elif case["phase"] == "training":
        data = training_data(root, case)
        centres = case.get("historical_replay", {}).get("centres", []) + seed_points(manifest, case)
        case["proposal"] = fit_proposal(data["points"], data["base"], data["density"], data["raw"],
                                        centres, policy, costs=data["cpu_seconds"], events=data["events"],
                                        tail_support=physical_scalar_bounds(manifest, case["energy_ev"]))
        if case["round"] + 1 < policy.max_proposal_rounds:
            case["round"] += 1
            return
        case["statistical_relative_tolerance"] = policy.statistical_relative_tolerance
        case["grids"] = [grid_edges(data["events"], policy.base_bins * 2 ** level)
                         for level in range(policy.max_bin_level + 1)]
        case["design_path"] = f"designs/{key}.json"
        case["design_sha256"] = digest(frozen_design(case))
        atomic_json(Path(root) / case["design_path"], frozen_design(case))
        case["phase"], case["target"] = "baseline", policy.initial_validation_histories
    elif case["phase"] == "baseline" and production_cohort(case) != "baseline":
        case["baseline"] = assess(root, manifest, case, "baseline")
        case["phase"] = "validation"
    else:
        case["assessment"] = assess(root, manifest, case)
        report = case["assessment"]
        case["efficiency"] = efficiency_diagnostic(root, case)
        if not report["importance_normalization_interval"][0] <= 1. <= report["importance_normalization_interval"][1]:
            case.update(status="blocked", reason="Importance normalization check failed")
            return
        reference_phase = "validation" if production_cohort(case) == "baseline" else "baseline"
        reference = assess(root, manifest, case, reference_phase)
        if not observations_compatible(report, reference):
            case.update(status="blocked", reason="Independent baseline and targeted estimates disagree")
            return
        if acceptance_passes(report, policy):
            identity = verification_identity(manifest, case, key)
            result = verification_result(root, identity)
            if result is not None:
                adopt_verification(state, case, result)
                return
            if state["reserved_histories"] + identity["propagations"] > policy.max_histories:
                case.update(status="blocked", reason="No remaining budget for mandatory deterministic replay")
                return
            case["phase"] = "verification"
            return
        gain = case["efficiency"].get("estimated_limiting_observable_gain")
        if production_cohort(case) == "validation" and gain is not None and gain <= 1.:
            # The original baseline proposal and its independent evidence are
            # retained. Existing simultaneous bounds cover choosing either
            # cohort; no samples are relabelled or pooled after this selection.
            case["selected_cohort"] = "baseline"
            case["sampling_decision"] = {
                "selected": "baseline", "estimated_targeted_gain": gain,
                "reason": "The independent targeted pilot did not improve variance per CPU-second",
                "tolerances_unchanged": True,
            }
            case["phase"] = "baseline"
            case["target"] = refresh_cohort(case["cohorts"][f"baseline_{case['round']}"])["count"]
            return
        case["range_budget_diagnostic"] = range_budget_diagnostic(
            manifest, case, report, weight_bound=cohort_weight_bound(case))
        if (policy.scalar_precision_mode == "simultaneous_bound" and
                not case["range_budget_diagnostic"]["sampling_at_pilot_means_can_fit_budget"]):
            case.update(status="blocked", reason="The finite-bound range penalty is too large in an optimistic budget projection; review the estimator before further sampling")
            return
        case["feasibility_passed"] = True
        if policy.output_scope == "joint_distribution" and not report["grid_pass"] and case["level"] < policy.max_bin_level:
            case["level"] += 1
            # The same weighted event products resolve finer training-defined
            # bins. No trajectories are rerun merely to change output bins.
            return
        target = next_sample_target(case["target"], policy)
        if target > case["target"]:
            case["target"] = target
        else:
            case.update(status="blocked", reason="Case budget exhausted without relative scalar / total-variation / grid certification")


def acceptance_passes(report, policy):
    """Never relabel scalar precision as distribution qualification."""
    if policy.output_scope == "scalar":
        return bool(report.get("scalar_pass", False))
    return report["statistical_pass"] and report["grid_pass"]


def next_sample_target(current, policy):
    factor = 1 + policy.sampling_growth if policy.output_scope == "scalar" else 2
    return min(policy.max_case_histories,
               math.ceil(current * factor / policy.batch_size) * policy.batch_size)


def scalar_interpolation_check(left, middle, right, policy):
    """Observed midpoint change plus independent-cohort SE, as in CTMC pilots.

    This is numerical refinement evidence at tested energies, not a confidence
    enclosure of interpolation error throughout the interval.
    """
    checks = {}
    for name in OBSERVABLES:
        a, m, b = [r["scalar"][name] for r in (left, middle, right)]
        pred = (a["estimate"] + b["estimate"]) / 2
        scale = min(pred, m["estimate"])
        errors = [v.get("standard_error") for v in (a, m, b)]
        valid = scale > 0 and all(e is not None and math.isfinite(e) and e > 0 for e in errors)
        se = math.sqrt(errors[1]**2 + (errors[0]**2 + errors[2]**2)/4) if valid else None
        delta = abs(m["estimate"] - pred)
        checks[name] = {"difference": delta, "comparison_standard_error": se,
            "passes": bool(valid and delta <= policy.scalar_interpolation_tolerance * scale
                           and se <= policy.statistical_relative_tolerance * scale),
            "resolved_failure": bool(valid and delta > policy.scalar_interpolation_tolerance * scale + 2 * se)}
    return {"passes": all(v["passes"] for v in checks.values()),
            "resolved_failure": any(v["resolved_failure"] for v in checks.values()),
            "observables": checks, "scope": "Observed midpoint convergence with estimated SE; no simultaneous guarantee"}


def interpolation_bound(left, middle, right, tolerance):
    definite_failure, uncertain = False, False
    worst = 0.
    for name in OBSERVABLES:
        a, m, b = [r["scalar"][name] for r in (left, middle, right)]
        intervals = [v["interval"] for v in (a, m, b)]
        if any(v[1] is None or v[0] <= 0 for v in intervals):
            uncertain = True
            continue
        lower_pred, upper_pred = .5 * (intervals[0][0] + intervals[2][0]), .5 * (intervals[0][1] + intervals[2][1])
        upper_error = max(abs(intervals[1][0] - upper_pred), abs(intervals[1][1] - lower_pred))
        lower_error = max(0., intervals[1][0] - upper_pred, lower_pred - intervals[1][1])
        allowed = tolerance * intervals[1][0]
        worst = max(worst, upper_error / allowed)
        definite_failure |= lower_error > allowed
        uncertain |= upper_error > allowed
    return {"passes": not uncertain, "resolved_failure": definite_failure, "maximum_bound_to_allowance": worst}


def advance_grid(root, manifest, state):
    policy = Policy(**manifest["policy"])
    for interval in list(state["intervals"]):
        if interval["status"] != "pending":
            continue
        a, b = [state["cases"][interval[name]] for name in ("a", "b")]
        if any(c["status"] == "blocked" for c in (a, b)):
            interval.update(status="blocked", reason="An endpoint is not qualified")
            continue
        if not all(c["status"] == "qualified" for c in (a, b)):
            continue
        if "mid" not in interval:
            mid = add_case(state, manifest, a["structure"], a["direction"], math.sqrt(a["energy_ev"] * b["energy_ev"]))
            if mid is None:
                interval.update(status="blocked", reason="Case limit prevents an independent energy midpoint")
                continue
            interval["mid"] = mid
        mid = state["cases"][interval["mid"]]
        if mid["status"] == "blocked":
            interval.update(status="blocked", reason="The midpoint is not qualified")
            continue
        if mid["status"] != "qualified":
            continue
        report = (scalar_interpolation_check(a["assessment"], mid["assessment"], b["assessment"], policy)
                  if policy.output_scope == "scalar" else
                  interpolation_bound(a["assessment"], mid["assessment"], b["assessment"], policy.interpolation_relative_tolerance))
        # A deterministic grid family prevents data-selected midpoint grids from
        # retrospectively choosing confidence queries on endpoint histories.
        level = interval.get("bin_level", 0)
        number = policy.base_bins * 2 ** level
        axis = np.unique(np.r_[0., np.geomspace(1e-16, 1., number + 1), np.linspace(0., 1., number + 1)]).tolist()
        edges = [axis, axis]
        bound = ({"passes": True, "resolved_failure": False, "qualified": False,
                  "scope": "Distribution interpolation is not requested by scalar production"}
                 if policy.output_scope == "scalar" else tv_interpolation_check(root, manifest, (a, mid, b), edges))
        report["joint_distribution"] = bound
        interval["assessment"] = report
        if bound.get("unresolved_grid", False) and level < policy.max_bin_level:
            interval["bin_level"] = level + 1
            continue
        if report["passes"] and bound["passes"]:
            interval["status"] = "qualified"
        elif (report["resolved_failure"] or bound["resolved_failure"]) and interval["depth"] < policy.max_energy_depth:
            interval["status"] = "split"
            for left, right in ((interval["a"], interval["mid"]), (interval["mid"], interval["b"])):
                state["intervals"].append({"a": left, "b": right, "depth": interval["depth"] + 1, "status": "pending"})
        elif report["resolved_failure"] or bound["resolved_failure"]:
            interval.update(status="blocked", reason="Energy refinement limit exhausted")
        else:
            enlarged = False
            for c in (a, mid, b):
                target = next_sample_target(c["target"], policy)
                if target > c["target"]:
                    c["target"] = target; c["status"] = "pending"; enlarged = True
            if not enlarged:
                interval.update(status="blocked", reason="Energy interpolation remains statistically unresolved at the case budget")


def tv_interpolation_check(root, manifest, cases, edges):
    """A common partition, clustered weighted uncertainty and the TV triangle inequality."""
    policy = Policy(**manifest["policy"])
    na, nb = len(edges[0]) - 1, len(edges[1]) - 1
    probabilities, radii, strips = [], [], []
    # The grid is deterministic; all permitted refinement levels are budgeted.
    looks = 2 + math.ceil(math.log(max(1, policy.max_case_histories / policy.initial_validation_histories),
                                  1 + policy.sampling_growth if policy.output_scope == "scalar" else 2))
    alpha = (1 - policy.confidence) / (4 * policy.max_cases * policy.max_proposal_rounds *
             (policy.max_bin_level + 1) * looks * (2 * na * nb + na + nb + 6))
    for case in cases:
        entries = case["cohorts"][f"{production_cohort(case)}_{case['round']}"]["blocks"]
        n, total, total2, _, _ = aggregate(root, entries, edges)
        physics = assessment_physics(root, manifest, case["energy_ev"]) if "tail_support" in manifest else manifest
        bounds = physical_scalar_bounds(physics, case["energy_ev"])
        support = np.r_[bounds, bounds[1], 1., np.full(len(total)-6, bounds[1])]
        low, high = empirical_interval(n, total, total2, support * cohort_weight_bound(case), alpha)
        if low[1] <= 0:
            return {"passes": False, "resolved_failure": False, "upper_bound": None}
        p = total[6:6 + 2 * na * nb] / total[1]
        lower = low[6:6 + 2 * na * nb] / high[1]
        upper = np.minimum(1., high[6:6 + 2 * na * nb] / low[1])
        probabilities.append(p); radii.append(total_variation_bound(p, lower, upper))
        upper_marg = np.minimum(1., high[6 + 2 * na * nb:] / low[1])
        strips.append(float(max(upper_marg[:na]) + max(upper_marg[na:])))
    a, m, b = probabilities
    observed = float(.5 * np.abs(m - .5 * (a + b)).sum())
    uncertainty = radii[1] + .5 * (radii[0] + radii[2])
    upper, lower = min(1., observed + uncertainty), max(0., observed - uncertainty)
    grid_ok = max(strips) <= policy.grid_cdf_tolerance
    return {"passes": upper <= policy.interpolation_tv_tolerance and grid_ok,
            "resolved_failure": lower > policy.interpolation_tv_tolerance,
            "unresolved_grid": not grid_ok, "upper_bound": upper, "lower_bound": lower,
            "observed_total_variation": observed, "statistical_radius": uncertainty,
            "maximum_cdf_grid_mass_bound": max(strips),
            "scope": "TV between bin masses; CDF grid resolution checked separately, not continuous-density TV"}


def verify_coverage(manifest, state):
    if not state["cases"]:
        raise ValueError("The declared physical matrix is missing")
    intervals = {(i["a"], i["b"]): i for i in state["intervals"]}
    if len(intervals) != len(state["intervals"]):
        raise ValueError("Duplicated energy intervals")
    for si in range(len(manifest["structures"])):
        for di in case_directions(manifest, si):
            keys = [digest([si, di, e])[:20] for e in manifest["energies_ev"]]
            if not all(key in state["cases"] for key in keys):
                raise ValueError("The declared physical matrix is incomplete")
            if not all((a, b) in intervals for a, b in zip(keys[:-1], keys[1:])):
                raise ValueError("An original energy interval is missing")
    for key, case in state["cases"].items():
        if key != digest([case["structure"], case["direction"], case["energy_ev"]])[:20]:
            raise ValueError("Changed physical case identity")
        if case["direction"] not in case_directions(manifest, case["structure"]):
            raise ValueError("Case uses a direction outside its declared material scope")
    for interval in state["intervals"]:
        a, b = [state["cases"][interval[x]] for x in ("a", "b")]
        if a["structure"] != b["structure"] or a["direction"] != b["direction"] or a["energy_ev"] >= b["energy_ev"]:
            raise ValueError("Invalid energy interval")
        if "mid" in interval:
            m = state["cases"][interval["mid"]]
            if not math.isclose(m["energy_ev"], math.sqrt(a["energy_ev"] * b["energy_ev"]), rel_tol=1e-14):
                raise ValueError("Changed geometric midpoint")
        if interval["status"] == "split" and ("mid" not in interval or
                (interval["a"], interval["mid"]) not in intervals or (interval["mid"], interval["b"]) not in intervals):
            raise ValueError("A required refinement child is missing")


def reconcile(root, manifest, state):
    for path in sorted((Path(root) / "blocks").glob("*.json")):
        entry = read(path); identity = entry["identity"]
        if identity["campaign"] != manifest["signature"] or path.stem != digest(identity):
            raise ValueError("Foreign or renamed checkpoint receipt")
        case = state["cases"][identity["case"]]
        c = case["cohorts"].setdefault(f"{identity['phase']}_{identity['round']}", {"blocks": [], "count": 0})
        if not any(b["path"] == entry["path"] for b in c["blocks"]):
            load_block(root, entry)
            c["blocks"].append(compact_entry(entry, file_hash(path)))
    used = 0
    for case in state["cases"].values():
        for c in case["cohorts"].values():
            refresh_cohort(c)
            used += c["count"]
    # Failed reservations and completed verification replays also consume work.
    state["reserved_histories"] = max(state["reserved_histories"], used)


def summary(state):
    cases = list(state["cases"].values())
    differential = state.get("differential", {})
    return {"status": state["status"], "cases": len(cases),
            "workflow_stage": state.get("workflow_stage", "not_started"),
            "calibration_histories": sum(v["count"] for c in cases for name, v in c["cohorts"].items() if name.startswith("training_")),
            "production_histories": sum(v["count"] for c in cases for name, v in c["cohorts"].items() if name.startswith(("baseline_", "validation_"))),
            "binary_dcs": {"status": differential.get("status", "not_started"),
                           "qualified_pair_energy_maps": sum(n["status"] == "qualified" for n in differential.get("nodes", {}).values()),
                           "required_pair_energy_maps": len(differential.get("nodes", {})),
                           "qualified_energy_intervals": sum(i["status"] == "qualified" for i in differential.get("intervals", [])),
                           "tables_verified": differential.get("tables_verified", False)},
            "reason": state.get("reason"), "pause_reason": state.get("pause_reason"),
            "qualified_cases": sum(c["status"] == "qualified" for c in cases),
            "blocked_cases": sum(c["status"] == "blocked" for c in cases),
            "committed_histories": sum(v["count"] for c in cases for v in c["cohorts"].values()),
            "new_independent_histories": sum(v["count"] for c in cases for name, v in c["cohorts"].items() if not name.startswith("replay_")),
            "historical_histories_replayed": sum(v["count"] for c in cases for name, v in c["cohorts"].items() if name.startswith("replay_")),
            "reserved_or_consumed_histories": state["reserved_histories"],
            "qualified_energy_intervals": sum(i["status"] == "qualified" for i in state["intervals"]),
            "blockers": {k: c.get("reason") for k, c in state["cases"].items() if c["status"] == "blocked"},
            "pilot_cases": state.get("pilot_cases", []),
            "output_scope": state.get("output_scope", "joint_distribution"),
            "scalar_response_ready": state.get("tables_verified", False),
            "response_tables_ready": state.get("tables_verified", False) and state.get("output_scope", "joint_distribution") == "joint_distribution",
            "production_release": False}


def active_cases(state):
    """Do not allocate the broad matrix until independent pilots are feasible."""
    pilots = state.get("pilot_cases", [])
    if pilots and not all(state["cases"][k].get("feasibility_passed") or state["cases"][k]["status"] == "qualified" for k in pilots):
        return [(k, state["cases"][k]) for k in pilots]
    return list(state["cases"].items())


def readiness(manifest, state):
    """Separate a reusable numerical product from a physical transport release."""
    return {"schema": SCHEMA, "campaign_signature": manifest["signature"],
            "projectile": manifest["projectile"], "status": state["status"],
            "materials": sorted({r.get("material", "specified_snapshot") for r in manifest["structures"]}),
            "required_base_cases": sum(len(case_directions(manifest, i)) for i in range(len(manifest["structures"]))) * len(manifest["energies_ev"]),
            "qualified_cases": sum(c["status"] == "qualified" for c in state["cases"].values()),
            "output_scope": Policy(**manifest["policy"]).output_scope,
            "scalar_response_production_ready": state.get("tables_verified", False),
            "response_table_production_ready": state.get("tables_verified", False) and Policy(**manifest["policy"]).output_scope == "joint_distribution",
            "binary_differential_cross_sections_ready": state.get("differential", {}).get("tables_verified", False),
            "local_elastic_cross_section_production_ready": False,
            "statistical_policy": {name: manifest["policy"][name] for name in
                                   ("confidence", "statistical_relative_tolerance", "distribution_tv_tolerance")},
            "numerical_policy": {name: manifest["policy"][name] for name in
                                 ("kernel_relative_tolerance", "grid_cdf_tolerance", "interpolation_relative_tolerance", "interpolation_tv_tolerance")},
            "numerical_blockers": summary(state)["blockers"],
            "coverage": "Only declared energies/interpolation intervals, snapshots and incident directions; no automatic transfer to another particle",
            "reusable_evidence": "Signed validation event blocks and frozen designs; training/replays guide sampling but are not pooled as independent validation",
            "physical_requirements_outside_this_certifier": [
                "Specify and validate a local/history-dependent collision-table contract if local transport is required",
                "Check the sequential binary approximation against overlapping multi-centre encounters",
                "Assess structural-ensemble uncertainty; one amorphous snapshot is not independent replicas",
                "Qualify an azimuth law before using these polar-angle/recoil tables as a complete directional event generator"],
            "completion_contract": "Pair success exports reusable microscopic DCS; separately qualified ice response tables are exported after their own gates. Neither silently authorizes a homogeneous phase-specific local law"}


@contextmanager
def campaign_lock(root):
    with (Path(root) / "controller.lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another controller owns this campaign") from exc
        yield


def stop_handler(signum, frame):
    global _STOP
    _STOP = True


def next_case_task(root, manifest, state, pending_ranges, cursor=0, smoke_remaining=None):
    """Choose one bounded task fairly, excluding committed and in-flight work."""
    policy = Policy(**manifest["policy"])
    calibration = state.get("workflow_stage") == "calibrate"
    cases = list(state["cases"].items()) if calibration else active_cases(state)
    if not cases or state["reserved_histories"] >= policy.max_histories or smoke_remaining == 0:
        return None, cursor
    for offset in range(len(cases)):
        position = (cursor + offset) % len(cases)
        key, case = cases[position]
        if calibration and case["phase"] not in ("replay", "training"):
            continue
        if case["status"] in ("blocked", "qualified") or not case.get("kernel", {}).get("passes"):
            continue
        worker_case = {name: case[name] for name in ("structure", "direction", "energy_ev", "kernel_mode")}
        if case["phase"] == "verification":
            pending_key = (key, "verification", case["round"])
            if pending_ranges.get(pending_key):
                continue
            identity = verification_identity(manifest, case, key)
            if identity["propagations"] > policy.max_histories - state["reserved_histories"]:
                continue
            if smoke_remaining is not None and identity["propagations"] > smoke_remaining:
                continue
            return (root, manifest, worker_case, identity), (position + 1) % len(cases)
        c = cohort(case)
        pending_key = (key, case["phase"], case["round"])
        gaps = missing_ranges(c, case["target"], pending_ranges.get(pending_key, ()))
        if not gaps:
            continue
        start, stop = gaps[0]
        geometry_check = state.get("workflow_stage") in ("calibrate", "produce") and case["phase"] == "training"
        multiplier = 2 if (case["phase"] == "validation" or geometry_check) and start < policy.batch_size else 1
        count = min(policy.batch_size, stop - start,
                    (policy.max_histories - state["reserved_histories"]) // multiplier)
        if smoke_remaining is not None:
            count = min(count, smoke_remaining)
        if count <= 0:
            continue
        identity = {"campaign": manifest["signature"], "case": key, "phase": case["phase"],
                    "round": case["round"], "start": start, "stop": start + count,
                    "proposal": case["proposal"] if case["phase"] != "baseline" else {"boxes": [], "alpha": [1.]},
                    "kernel_mode": case["kernel_mode"]}
        if geometry_check:
            identity["calibration_geometry_check"] = True
        if case["phase"] in ("baseline", "validation"):
            identity["design_sha256"] = case["design_sha256"]
        return (root, manifest, worker_case, identity), (position + 1) % len(cases)
    return None, cursor


def calibration_index(root, manifest, state):
    """A separate frozen-design handoff, not a phase-output certificate."""
    designs = {}
    for key, case in state["cases"].items():
        if "design_path" in case and case["status"] != "blocked":
            verify_design(root, case)
            designs[key] = {"path": case["design_path"], "sha256": case["design_sha256"]}
    ready = bool(state["cases"]) and len(designs) == len(state["cases"]) and state["differential"].get("tables_verified", False)
    result = {"campaign_signature": manifest["signature"], "ready_for_independent_sampling": ready,
              "designs": designs, "pair_index_sha256": state["differential"].get("index_sha256"),
              "scalar_precision_mode": manifest["policy"]["scalar_precision_mode"],
              "scope": "Pair numerics and per-case sampling/grid families frozen; final phase distribution, interpolation and model adequacy are not certified"}
    result["sha256"] = digest(result)
    atomic_json(Path(root)/"calibration/index.json", result)
    return result


def verify_calibration(root, manifest, state):
    path = Path(root)/"calibration/index.json"
    if not path.exists():
        raise ValueError("Run calibrate before produce: no frozen calibration handoff")
    result = read(path); signature = result.pop("sha256")
    if (digest(result) != signature or result["campaign_signature"] != manifest["signature"]
            or not result["ready_for_independent_sampling"]
            or result["pair_index_sha256"] != state["differential"].get("index_sha256")):
        raise ValueError("Calibration is incomplete or its handoff changed")
    for key, item in result["designs"].items():
        case = state["cases"][key]
        verify_design(root, case)
        if item != {"path": case["design_path"], "sha256": case["design_sha256"]}:
            raise ValueError("Production settings differ from frozen calibration")
    # Adaptive energy children get their own training and frozen design before
    # any independent sample; previously frozen cases must never be retrained.
    return result


def run(root, workers=1, max_new_histories=None, stage="combined"):
    global _STOP
    _STOP = False
    root = Path(root).resolve()
    manifest = verify(root); policy = Policy(**manifest["policy"])
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    if max_new_histories is not None and (type(max_new_histories) is not int or max_new_histories <= 0):
        raise ValueError("The optional smoke budget must be a positive integer")
    if not os.environ.get("PBS_JOBID") and (workers > 2 or max_new_histories is None or max_new_histories > 64):
        raise ValueError("Outside PBS only an explicit <=64-history, <=2-worker smoke test is allowed")
    available = int(os.environ.get("NCPUS", workers))
    if workers > available:
        raise ValueError("Requested workers exceed the PBS allocation")
    signal.signal(signal.SIGTERM, stop_handler); signal.signal(signal.SIGINT, stop_handler)
    started, submitted = time.monotonic(), 0
    with campaign_lock(root):
        state = read(root / "state.json"); verify_coverage(manifest, state); reconcile(root, manifest, state)
        if stage not in ("calibrate", "produce", "combined"):
            raise ValueError("Unknown workflow stage")
        if stage == "produce":
            verify_calibration(root, manifest, state)
        state["workflow_stage"] = stage
        state["output_scope"] = policy.output_scope
        if not run_differential(root, manifest, state, workers, started):
            state.update(status="paused_restartable" if state["differential"]["status"] == "pending" else "not_qualified",
                         reason="binary_differential_stage_incomplete")
            if state["status"] == "paused_restartable":
                state["pause_reason"] = "signal" if _STOP else ("smoke_budget" if not os.environ.get("PBS_JOBID") else "time_limit")
            atomic_json(root / "state.json", state)
            atomic_json(root / "report.json", summary(state))
            atomic_json(root / "readiness.json", readiness(manifest, state))
            return summary(state)
        for case in state["cases"].values():
            case["kernel"] = pair_gate(state)
        preflight = feasibility(root, manifest, state)
        if preflight["blocked_cases"] and not preflight.get("tail_pilot_required", False):
            state.update(status="not_qualified", reason="estimator_review_required_before_sampling")
            for key in preflight["blocked_cases"]:
                state["cases"][key].update(status="blocked", reason=preflight["reason"])
            atomic_json(root / "state.json", state)
            atomic_json(root / "report.json", summary(state))
            atomic_json(root / "readiness.json", readiness(manifest, state))
            return summary(state)
        state["status"] = "running"
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            with tqdm(total=policy.max_histories, initial=state["reserved_histories"], desc="NLH propagations (budget)", unit="history",
                      bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]") as progress:
                futures, pending_ranges = {}, {}
                cursor = 0
                while True:
                    dispatch = (not _STOP and time.monotonic() - started < policy.max_seconds and
                                (max_new_histories is None or submitted < max_new_histories))
                    changed = False
                    if dispatch:
                        for key in list(state["cases"]):
                            if stage == "calibrate" and state["cases"][key]["phase"] not in ("replay", "training"):
                                continue
                            before = digest(state["cases"][key])
                            advance_case(root, manifest, state, key)
                            changed |= before != digest(state["cases"][key])
                        before_grid = digest(state["intervals"])
                        if stage != "calibrate":
                            advance_grid(root, manifest, state)
                        changed |= before_grid != digest(state["intervals"])
                        # Replenish each released worker immediately. Fair case
                        # rotation prevents one long case from monopolizing the
                        # queue; immutable history indices preserve exact resume.
                        while len(futures) < workers and not _STOP:
                            if time.monotonic() - started >= policy.max_seconds:
                                break
                            remaining = None if max_new_histories is None else max_new_histories - submitted
                            task, cursor = next_case_task(root, manifest, state, pending_ranges, cursor, remaining)
                            if task is None:
                                break
                            identity = task[3]
                            count = identity["stop"] - identity["start"]
                            if identity["phase"] == "verification":
                                work = identity["propagations"]
                                accounting = state.setdefault("verification_work", {})
                                accounting[digest(identity)] = accounting.get(digest(identity), 0) + work
                                state["reserved_histories"] += work
                                submitted += work  # Bound local smoke work; no independent evidence is added.
                            else:
                                multiplier = 2 if (identity["phase"] == "validation" or identity.get("calibration_geometry_check")) and identity["start"] < policy.batch_size else 1
                                state["reserved_histories"] += multiplier * count
                                submitted += count
                            pending_key = (identity["case"], identity["phase"], identity["round"])
                            pending_ranges.setdefault(pending_key, []).append((identity["start"], identity["stop"]))
                            futures[pool.submit(execute_block, task)] = task
                            changed = True
                    if changed:
                        atomic_json(root / "state.json", state)
                    if not futures:
                        if dispatch and changed:
                            continue
                        break
                    # A timeout observes cancellation without a wave barrier.
                    # On a stop, drain only the bounded tasks already in flight.
                    done, _ = wait(futures, timeout=1., return_when=FIRST_COMPLETED)
                    for future in done:
                        task = futures.pop(future)
                        identity = task[3]; case = state["cases"][identity["case"]]
                        try:
                            result = future.result()
                            if identity["phase"] == "verification":
                                proof = verification_result(root, identity)
                                if proof is None or proof != result:
                                    raise ValueError("Verification worker did not commit its matching proof")
                                adopt_verification(state, case, proof)
                                progress.update(identity["propagations"])
                            else:
                                entry = save_block(root, identity, result)
                                c = case["cohorts"][f"{identity['phase']}_{identity['round']}"]
                                c["blocks"].append(entry); refresh_cohort(c)
                                progress.update(identity["stop"] - identity["start"])
                        except Exception as exc:
                            case.update(status="blocked", reason=f"{type(exc).__name__}: {exc}")
                        finally:
                            pending_key = (identity["case"], identity["phase"], identity["round"])
                            pending_ranges[pending_key].remove((identity["start"], identity["stop"]))
                        atomic_json(root / "state.json", state)
        complete = stage != "calibrate" and all(c["status"] == "qualified" for c in state["cases"].values()) and all(
            i["status"] in ("qualified", "split") for i in state["intervals"])
        if complete:
            state["status"] = "numerically_qualified_for_declared_protocol"
        elif _STOP or (max_new_histories is not None and submitted >= max_new_histories) or time.monotonic() - started >= policy.max_seconds:
            state["status"] = "paused_restartable"
            state["pause_reason"] = "signal" if _STOP else ("smoke_budget" if max_new_histories is not None and submitted >= max_new_histories else "time_limit")
        else:
            state["status"] = "not_qualified"
            state["reason"] = "campaign_budget" if state["reserved_histories"] >= policy.max_histories else "unresolved_case_or_grid_gate"
        if stage == "calibrate":
            handoff = calibration_index(root, manifest, state)
            if handoff["ready_for_independent_sampling"]:
                state.update(status="calibration_ready", reason="Frozen settings; independent production not yet certified")
        elif stage == "produce":
            atomic_json(root/"production/report.json", {"campaign_signature": manifest["signature"],
                "summary": summary(state), "scalar_precision_mode": policy.scalar_precision_mode,
                "calibration_samples_pooled": False})
        atomic_json(root / "state.json", state)
        atomic_json(root / "report.json", summary(state))
        atomic_json(root / "readiness.json", readiness(manifest, state))
    if complete:
        try:
            export(root)
            state["tables_verified"] = True
            atomic_json(root / "state.json", state)
            atomic_json(root / "report.json", summary(state))
            atomic_json(root / "readiness.json", readiness(manifest, state))
        except Exception as exc:
            state.update(status="not_qualified", reason=f"Export verification failed: {exc}")
            atomic_json(root / "state.json", state)
            atomic_json(root / "report.json", summary(state))
            atomic_json(root / "readiness.json", readiness(manifest, state))
            raise
    return summary(state)


def export(root):
    root = Path(root).resolve(); manifest = verify(root)
    with campaign_lock(root):
        state = read(root / "state.json"); verify_coverage(manifest, state); reconcile(root, manifest, state)
        # Recompute every acceptance decision from checksum-verified blocks;
        # an edited report cannot promote an incomplete scientific campaign.
        for key, case in state["cases"].items():
            if "design_path" not in case:
                raise ValueError("Export refused: a required case is not qualified")
            report = assess(root, manifest, case)
            if report is None or not acceptance_passes(report, Policy(**manifest["policy"])):
                raise ValueError("Export refused: a required case is not qualified")
            proof = verification_result(root, verification_identity(manifest, case, key))
            if proof is None:
                raise ValueError("Export refused: the current selected cohort has no matching replay proof")
            case["replay"] = proof
            reference = assess(root, manifest, case, "validation" if production_cohort(case) == "baseline" else "baseline")
            if not observations_compatible(report, reference) or not report["importance_normalization_interval"][0] <= 1. <= report["importance_normalization_interval"][1]:
                raise ValueError("Export refused: baseline or normalization verification failed")
            case["assessment"] = report; case["status"] = "qualified"
        for interval in state["intervals"]:
            if interval["status"] != "split":
                interval["status"] = "pending"
        advance_grid(root, manifest, state)
        if any(i["status"] not in ("qualified", "split") for i in state["intervals"]):
            raise ValueError("Export refused: energy interpolation is not qualified")
        if Policy(**manifest["policy"]).output_scope == "scalar":
            return export_scalars(root, manifest, state)
        directory = root / "tables"; directory.mkdir(exist_ok=True)
        tables = []
        for key, case in state["cases"].items():
            edges = case["grids"][case["level"]]
            entries = case["cohorts"][f"{production_cohort(case)}_{case['round']}"]["blocks"]
            n, total, total2, _, _ = aggregate(root, entries, edges)
            na, nb = len(edges[0]) - 1, len(edges[1]) - 1
            rate = total[6:6 + 2 * na * nb].reshape(2, na, nb) / total[0]
            record = manifest["structures"][case["structure"]]
            density = record["water_molecules"] / record["volume_angstrom3"]
            occupied = np.argwhere(rate > 0)
            selected = tuple(occupied.T)
            policy = Policy(**manifest["policy"])
            low, high = empirical_interval(n, total[6:6 + 2 * na * nb], total2[6:6 + 2 * na * nb],
                    case["assessment"]["physical_history_support"][1] * cohort_weight_bound(case),
                    case["assessment"]["individual_alpha"])
            counts_low, counts_high = case["assessment"]["scalar_lower"][1], case["assessment"]["scalar_upper"][1]
            probability_low = (low / counts_high).reshape(rate.shape)
            probability_high = np.minimum(1., high / counts_low).reshape(rate.shape)
            payload = {"schema": SCHEMA, "campaign_signature": manifest["signature"], "case": key,
                       "projectile": manifest["projectile"],
                       "estimand": manifest["estimand"], "structure": record,
                       "entrance_energy_ev": case["energy_ev"], "entrance_direction": manifest["directions"][case["direction"]],
                       "path_length_angstrom": manifest["path_length_angstrom"], "targets": ["H", "O"],
                       "angular_coordinate": "sin(theta_lab/2)^2", "recoil_coordinate": "recoil_energy / entrance_energy",
                       "angular_edges": edges[0], "recoil_fraction_edges": edges[1],
                       "shape": list(rate.shape), "nonzero_bin_indices_target_angle_recoil": occupied.tolist(),
                       "joint_probability": (rate[selected] / rate.sum()).tolist(),
                       "joint_probability_lower": probability_low[selected].tolist(),
                       "joint_probability_upper": probability_high[selected].tolist(),
                       "unsampled_bin_probability_upper_bound": float(np.max(probability_high[rate == 0])) if np.any(rate == 0) else 0.,
                       "macroscopic_bin_rate_per_cm": (rate[selected] * 1e8).tolist(),
                       "effective_bin_area_per_water_cm2": (rate[selected] / density * 1e-16).tolist(),
                       "uncertainty": case["assessment"], "raw_weighted_event_blocks": entries,
                       "selected_evidence_cohort": production_cohort(case),
                       "scope": "Per-event observables averaged over this finite-path entrance ensemble; not local Markov cross sections",
                       "geant4_drop_in": False, "physical_model_validated": False,
                       "exclusions": ["soft scattering", "simultaneous many-atom forces", "recoil cascades", "electronic stopping", "charge exchange"],
                       "within_bin_sampling_law_qualified": False,
                       "recoil_is_local_energy_deposition": False}
            path = directory / (key + ".json"); atomic_json(path, payload)
            tables.append({"case": key, "path": path.name, "sha256": file_hash(path)})
        index = {"schema": SCHEMA, "campaign_signature": manifest["signature"], "tables": tables,
                 "status": "numerically_qualified_model_conditional_response", "production_release": False,
                 "orientation_domain": manifest["orientation_domain"], "energy_intervals": state["intervals"],
                 "interpolation": "linear total rates and mixtures of endpoint conditional distributions in log entrance energy, inside tested intervals only",
                 "interpolation_grid": "common predetermined bin partition tested in TV; no guarantee of continuous-density TV",
                 "sampling_contract": "H/O, polar-angle and recoil bin masses jointly; independent marginal sampling is forbidden",
                 "physical_release_blockers": ["binary-overlap approximation lacks an independent multicentre comparison",
                     "a finite-path response is not a history-independent local transport operator",
                     "snapshot Monte Carlo precision is not uncertainty of the physical ice ensemble"]}
        atomic_json(directory / "index.json", index)
        state["status"] = "numerically_qualified_for_declared_protocol"
        atomic_json(root / "state.json", state)
        verified = verify_tables(root, manifest)
        atomic_json(root / "production_handoff.json", verified)
        state["tables_verified"] = True
        atomic_json(root / "state.json", state)
        atomic_json(root / "readiness.json", readiness(manifest, state))
        return index


def export_scalars(root, manifest, state):
    """Called under the export lock after evidence and interpolation verification."""
    directory = root / "tables"; directory.mkdir(exist_ok=True)
    tables = []
    for key, case in state["cases"].items():
        payload = {"schema": SCHEMA, "campaign_signature": manifest["signature"],
            "case": key, "projectile": manifest["projectile"], "output_scope": "scalar",
            "entrance_energy_ev": case["energy_ev"], "structure": manifest["structures"][case["structure"]],
            "entrance_direction": manifest["directions"][case["direction"]],
            "path_length_angstrom": manifest["path_length_angstrom"],
            "uncertainty": case["assessment"], "estimand": manifest["estimand"],
            "joint_distribution_qualified": False, "physical_model_validated": False,
            "scope": "Whole-history finite-path scalar ratios; not a differential collision generator",
            "raw_weighted_event_blocks": case["cohorts"][f"{production_cohort(case)}_{case['round']}"]["blocks"]}
        path = directory / (key + ".json"); atomic_json(path, payload)
        tables.append({"case": key, "path": path.name, "sha256": file_hash(path)})
    index = {"schema": SCHEMA, "campaign_signature": manifest["signature"], "tables": tables,
        "output_scope": "scalar", "status": "numerically_qualified_scalar_response",
        "joint_distribution_qualified": False, "energy_intervals": state["intervals"],
        "interpolation": "Linear scalar ratios in log entrance energy; tested midpoint convergence only"}
    atomic_json(directory / "index.json", index)
    state["status"] = "numerically_qualified_for_declared_protocol"
    atomic_json(root / "state.json", state)
    verified = verify_tables(root, manifest)
    atomic_json(root / "production_handoff.json", verified)
    state["tables_verified"] = True
    atomic_json(root / "state.json", state)
    atomic_json(root / "readiness.json", readiness(manifest, state))
    return index


def verify_scalar_table(table, case, manifest):
    policy = Policy(**manifest["policy"])
    if table.get("output_scope") != "scalar" or table.get("joint_distribution_qualified") is not False:
        raise ValueError("Scalar table falsely claims distribution qualification")
    if table.get("path_length_angstrom") != manifest["path_length_angstrom"]:
        raise ValueError("Scalar path length mismatch")
    report = table["uncertainty"]
    if not acceptance_passes(report, policy):
        raise ValueError("Scalar precision failed")
    for name in OBSERVABLES:
        v = report["scalar"][name]
        estimate, se = v["estimate"], v.get("standard_error")
        if (not math.isfinite(estimate) or estimate <= 0 or se is None or
                not math.isfinite(se) or not 0 < se / estimate <= policy.statistical_relative_tolerance):
            raise ValueError("Scalar table fails signed relative standard error")


def verify_tables(root, manifest=None):
    """Read-only production handoff check; never substitutes an unchecked table."""
    root = Path(root).resolve()
    manifest = verify(root) if manifest is None else manifest
    state = read(root / "state.json")
    verify_coverage(manifest, state)
    index_path = root / "tables" / "index.json"
    index = read(index_path)
    if index["campaign_signature"] != manifest["signature"] or index["schema"] != SCHEMA:
        raise ValueError("Table index belongs to another campaign")
    handoff = root / "production_handoff.json"
    if handoff.exists() and read(handoff)["index_sha256"] != file_hash(index_path):
        raise ValueError("Table index differs from the recorded production handoff")
    keys = [t["case"] for t in index["tables"]]
    if len(keys) != len(set(keys)) or set(keys) != set(state["cases"]):
        raise ValueError("Production handoff is missing or duplicating required cases")
    if index["energy_intervals"] != state["intervals"] or any(i["status"] not in ("qualified", "split") for i in index["energy_intervals"]):
        raise ValueError("Energy coverage is not qualified for production handoff")
    if Policy(**manifest["policy"]).output_scope == "scalar" and (index.get("output_scope") != "scalar" or index.get("joint_distribution_qualified") is not False):
        raise ValueError("Scalar index falsely claims distribution qualification")
    for entry in index["tables"]:
        if Path(entry["path"]).name != entry["path"]:
            raise ValueError("Invalid table path")
        path = root / "tables" / entry["path"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError("Production table checksum changed")
        table = read(path)
        case = state["cases"][entry["case"]]
        if (table["campaign_signature"] != manifest["signature"] or table["case"] != entry["case"]
                or table["projectile"] != manifest["projectile"]
                or table["entrance_energy_ev"] != case["energy_ev"]
                or table["structure"] != manifest["structures"][case["structure"]]
                or table["entrance_direction"] != manifest["directions"][case["direction"]]):
            raise ValueError("Production table identity mismatch")
        if Policy(**manifest["policy"]).output_scope == "scalar":
            verify_scalar_table(table, case, manifest)
            continue
        probability = np.asarray(table["joint_probability"])
        rate = np.asarray(table["macroscopic_bin_rate_per_cm"])
        area = np.asarray(table["effective_bin_area_per_water_cm2"])
        if probability.shape != rate.shape or area.shape != rate.shape or np.any(probability < 0) or not np.isclose(probability.sum(), 1., rtol=0, atol=1e-12):
            raise ValueError("Invalid table normalization")
        indices = np.asarray(table["nonzero_bin_indices_target_angle_recoil"])
        expected_shape = [2, len(table["angular_edges"]) - 1, len(table["recoil_fraction_edges"]) - 1]
        if table["shape"] != expected_shape or indices.shape != (len(probability), 3) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Invalid joint table shape")
        if np.any(indices < 0) or np.any(indices >= np.array(expected_shape)) or len(np.unique(indices, axis=0)) != len(indices):
            raise ValueError("Invalid or duplicate joint table cell")
        for axis in (table["angular_edges"], table["recoil_fraction_edges"]):
            if axis[0] != 0. or axis[-1] != 1. or not np.all(np.diff(axis) > 0):
                raise ValueError("Table axes do not cover the normalized physical domain")
        record = table["structure"]
        density_per_cm3 = record["water_molecules"] / record["volume_angstrom3"] * 1e24
        if not np.allclose(area * density_per_cm3, rate, rtol=1e-12, atol=0):
            raise ValueError("Macroscopic/microscopic area units disagree")
        if not np.allclose(probability * rate.sum(), rate, rtol=1e-12, atol=0):
            raise ValueError("Joint probabilities and bin rates disagree")
        if not table["uncertainty"]["statistical_pass"] or not table["uncertainty"]["grid_pass"]:
            raise ValueError("Unqualified uncertainty in production table")
        policy = Policy(**manifest["policy"])
        for name in OBSERVABLES:
            scalar = table["uncertainty"]["scalar"][name]
            accepted = (scalar.get("relative_standard_error") is not None and
                        0 < scalar["relative_standard_error"] <= policy.statistical_relative_tolerance)
            if policy.scalar_precision_mode == "simultaneous_bound":
                accepted = relative_precision(scalar["estimate"], scalar["interval"], policy.statistical_relative_tolerance)["passes"]
            if not accepted:
                raise ValueError("Production table fails the signed relative precision")
        if not 0 <= table["uncertainty"]["joint_distribution_total_variation_upper_bound"] <= policy.distribution_tv_tolerance:
            raise ValueError("Production table fails the signed TV precision")
        if not np.isclose(rate.sum(), table["uncertainty"]["scalar"][OBSERVABLES[0]]["estimate"] * 1e8, rtol=1e-12, atol=0):
            raise ValueError("Total rate differs from the certified rate")
    return {"schema": SCHEMA, "campaign_signature": manifest["signature"],
            "index_path": str(index_path), "index_sha256": file_hash(index_path),
            "projectile": manifest["projectile"], "policy": manifest["policy"],
            "verified_case_count": len(keys),
            "scalar_response_ready": True,
            "response_tables_ready": Policy(**manifest["policy"]).output_scope == "joint_distribution",
            "output_scope": Policy(**manifest["policy"]).output_scope,
            "local_elastic_collision_law_ready": False, "geant4_drop_in": False,
            "reuse": "Consume this verified index and its weighted event blocks; no duplicate production trajectory campaign is required for the same estimand",
            "scope": manifest["estimand"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare"); prepare_parser.add_argument("root", type=Path); prepare_parser.add_argument("spec", type=Path)
    prepare_parser.add_argument("--projectile")
    prepare_parser.add_argument("--material", choices=("hexagonal", "amorphous"))
    prepare_parser.add_argument("--reuse-pairs", type=Path,
                                help="Reuse unchanged binary products, not old phase acceptance")
    for name in ("calibrate", "produce", "run"):
        runner = sub.add_parser(name, help="Combined diagnostic workflow" if name == "run" else None)
        runner.add_argument("root", type=Path); runner.add_argument("--workers", type=int, default=1)
        runner.add_argument("--max-new-histories", type=int)
    for name in ("status", "export", "verify-tables", "verify-differential", "feasibility", "compare-scalar-intervals"):
        child = sub.add_parser(name); child.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        answer = summary(prepare(args.root, args.spec, args.projectile, args.material, args.reuse_pairs))
    elif args.command in ("run", "calibrate", "produce"):
        answer = run(args.root, args.workers, args.max_new_histories,
                     "combined" if args.command == "run" else args.command)
    elif args.command == "status":
        verify(args.root); answer = summary(read(args.root / "state.json"))
    elif args.command == "verify-tables":
        answer = verify_tables(args.root)
    elif args.command == "verify-differential":
        answer = verify_differential(args.root)
    elif args.command == "feasibility":
        answer = feasibility(args.root)
    elif args.command == "compare-scalar-intervals":
        answer = compare_scalar_intervals(args.root)
    else:
        answer = export(args.root)
    print(json.dumps(answer, indent=2))
    return 0


if __name__ == "__main__":
    main()
