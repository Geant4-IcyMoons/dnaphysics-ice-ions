"""Prepare phase-resolved handoff experiments; compare independent cohorts.

Preparation does not submit jobs or imply that a combined kernel exists.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

from physics.elastic.zbl.generate_backend import (
    PHASE_ORIENTATIONS, REPOSITORY_ROOT, _read_registry,
)

DEFAULT_CONFIG = Path(__file__).with_name("carbon_ice.json")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                   allow_nan=False).encode()).hexdigest()


def compare_independent(left, right, *, tolerance=0.05):
    """Observed sensitivity and SE, not a confidence/equivalence guarantee.

    Each input contains mean and standard_error for a whole-history estimate.
    Paired cohorts require a covariance-aware estimator instead.
    """
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("Tolerance must be between zero and one.")
    if any(v.get(k) is None for v in (left, right) for k in ("mean", "standard_error")):
        return {"status": "unresolved", "reason": "missing_estimate_or_se"}
    a, b = float(left["mean"]), float(right["mean"])
    sa, sb = float(left["standard_error"]), float(right["standard_error"])
    if not all(math.isfinite(x) and x > 0 for x in (a, b, sa, sb)):
        return {"status": "unresolved", "reason": "nonpositive_or_nonfinite_estimate_or_se"}
    scale = min(a, b)
    change = abs(a-b)/scale
    se = math.hypot(sa, sb)/scale
    return {"status": "screen_pass" if max(change, se) <= tolerance else "unresolved",
            "relative_change": change, "relative_comparison_standard_error": se,
            "target": tolerance, "confidence_guarantee": False,
            "physical_validation": False}


def prepare(configuration):
    config = dict(configuration)
    if config["schema_version"] != 1 or config["projectile"] != "C":
        raise ValueError("This study currently defines carbon only.")
    for key in ("phases", "energies_ev", "boundary_potential_ev", "soft_recoil_cutoffs_ev"):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(f"{key} must be nonempty and unique.")
    for key in ("energies_ev", "boundary_potential_ev", "soft_recoil_cutoffs_ev"):
        if any(not math.isfinite(x) or x <= 0 for x in config[key]):
            raise ValueError(f"{key} must be finite and positive.")
    if min(config["boundary_potential_ev"]) < 30:
        raise ValueError("The primary study stays within the retained >=30 eV boundary domain.")
    for key in ("relative_standard_error_target", "relative_boundary_change_target"):
        if not math.isfinite(config[key]) or not 0 < config[key] < 1:
            raise ValueError(f"Invalid {key}.")
    if not math.isfinite(config["path_length_angstrom"]) or config["path_length_angstrom"] <= 0:
        raise ValueError("Invalid path length.")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    sources = {}
    for directory in ("bca", "nlh", "zbl", "handoff"):
        for path in sorted((REPOSITORY_ROOT/"physics/elastic"/directory).glob("*")):
            if path.suffix in (".py", ".csv"):
                sources[str(path.relative_to(REPOSITORY_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    conditions = []
    for phase in config["phases"]:
        registry_path, registry = _read_registry(phase)
        sources[str(registry_path.relative_to(REPOSITORY_ROOT))] = hashlib.sha256(registry_path.read_bytes()).hexdigest()
        for index, record in enumerate(registry["structures"]):
            path = (registry_path.parent/record["path"]).resolve()
            metadata = path.with_suffix(".json")
            if not metadata.is_file():
                raise ValueError(f"Missing structure metadata: {metadata}")
            sources[str(metadata.relative_to(REPOSITORY_ROOT))] = hashlib.sha256(metadata.read_bytes()).hexdigest()
            for energy, direction in itertools.product(config["energies_ev"], PHASE_ORIENTATIONS[phase]):
                conditions.append({"phase": phase, "structure": str(path.relative_to(REPOSITORY_ROOT)),
                                   "structure_sha256": record["sha256"], "energy_ev": energy,
                                   "direction": direction, "stage": "pilot" if index == 0 else "replicas"})
    cohorts = []
    for condition in conditions:
        for boundary in [*config["boundary_potential_ev"], None]:
            case = dict(condition, model="combined_nlh_zbl" if boundary is not None else "zbl_full",
                        boundary_potential_ev=boundary,
                        soft_recoil_cutoff_ev=None,
                        prerequisite="binary_convergence_and_soft_cutoff_selection")
            case["id"] = digest(case)[:20]
            case["seed"] = int(digest([config["seed"], case["id"]])[:16], 16)
            cohorts.append(case)
    payload = {"configuration": config, "sources": sources, "cohorts": cohorts}
    return dict(payload, signature=digest(payload),
                status="prepared", handoff_qualified=False,
                runtime="physics.elastic.handoff.run",
                binary_tasks=[{"target": t, "energy_ev": e,
                               "boundary_potential_ev": config["boundary_potential_ev"],
                               "soft_recoil_cutoffs_ev": config["soft_recoil_cutoffs_ev"]}
                              for t, e in itertools.product(("H", "O"), config["energies_ev"])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(json.loads(args.configuration.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text()) != manifest:
            raise RuntimeError("Existing study differs; choose a new output, preserving provenance.")
    else:
        temporary = args.output.with_suffix(args.output.suffix+".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        temporary.replace(args.output)
    print(f"Prepared {len(manifest['binary_tasks'])} pair tasks and {len(manifest['cohorts'])} planned ice cohorts; no jobs submitted.")


if __name__ == "__main__":
    main()
