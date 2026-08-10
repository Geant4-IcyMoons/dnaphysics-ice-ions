"""Generate phase-attested full-ZBL atomistic backend manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from .backend import DEFAULT_ENERGY_BOUNDS_EV, DEFAULT_PROJECTILES, FullZBLKernel
from .kernel import PROJECTILES

from bca.structure import file_sha256, load_ice_structure  # noqa: E402
from bca.trajectory import PeriodicHardCollisionTransport  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
ICE_STRUCTURES_ROOT = REPOSITORY_ROOT / "python_scripts/physics_ice/ice_structures"
PHASE_REGISTRIES = {
    "hexagonal_ih_100k": (
        ICE_STRUCTURES_ROOT
        / "hexagonal_ih_100K_experimental/collision_structures.json"
    ),
    "amorphous_lda_80k": (
        ICE_STRUCTURES_ROOT
        / "epsr_lda80k/artifacts/collision_structures.json"
    ),
}
PHASE_ORIENTATIONS = {
    "hexagonal_ih_100k": ("c_axis", "basal_a_axis", "isotropic"),
    "amorphous_lda_80k": ("isotropic",),
}
DEFAULT_CUTOFFS_EV = (1.0, 10.0, 30.0)


def _repository_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))


def _read_registry(phase_id: str) -> tuple[Path, dict[str, object]]:
    registry_path = PHASE_REGISTRIES[phase_id].resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if (
        registry.get("schema_version") != 1
        or registry.get("all_collision_ready") is not True
    ):
        raise ValueError(f"{phase_id} registry is not collision-ready.")
    records = registry.get("structures")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{phase_id} registry contains no structures.")
    for record in records:
        if not isinstance(record, dict) or record.get("collision_ready") is not True:
            raise ValueError(f"{phase_id} contains an unattested structure.")
        source = (registry_path.parent / str(record["path"])).resolve()
        if file_sha256(source) != record.get("sha256"):
            raise ValueError(f"Structure checksum mismatch: {source}")
    return registry_path, registry


def build_manifest(
    phase_id: str,
    *,
    projectiles: Sequence[str] = DEFAULT_PROJECTILES,
    energy_bounds_ev: tuple[float, float] = DEFAULT_ENERGY_BOUNDS_EV,
    minimum_transfer_cutoffs_ev: Sequence[float] = DEFAULT_CUTOFFS_EV,
) -> dict[str, object]:
    """Return a validated phase/backend manifest without running trajectories."""
    cutoffs = tuple(float(value) for value in minimum_transfer_cutoffs_ev)
    if not cutoffs or any(value <= 0.0 for value in cutoffs):
        raise ValueError("Minimum-transfer cutoffs must be positive.")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("Minimum-transfer cutoffs must be unique.")
    # Construction validates projectile names, energy bounds, and each cutoff.
    kernels = [
        FullZBLKernel(
            minimum_transfer_ev=cutoff,
            projectiles=projectiles,
            energy_bounds_ev=energy_bounds_ev,
        )
        for cutoff in cutoffs
    ]
    registry_path, registry = _read_registry(phase_id)
    structures = []
    for record in registry["structures"]:
        source = (registry_path.parent / str(record["path"])).resolve()
        metadata = source.with_suffix(".json")
        structures.append(
            {
                "path": _repository_relative(source),
                "metadata": _repository_relative(metadata),
                "sha256": record["sha256"],
                "frame_index": int(record.get("frame_index", 0)),
                "density_g_cm3": float(record["density_g_cm3"]),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "backend": "atomistic_periodic_full_zbl",
        "validation_state": "validation_pending",
        "production_enabled": False,
        "phase_id": phase_id,
        "structure_registry": _repository_relative(registry_path),
        "structures": structures,
        "orientations": list(PHASE_ORIENTATIONS[phase_id]),
        "projectiles": list(projectiles),
        "charge_state_dependence": "none",
        "charge_state_aliases": {
            symbol: list(range(PROJECTILES[symbol].atomic_number + 1))
            for symbol in projectiles
        },
        "total_kinetic_energy_bounds_ev": list(energy_bounds_ev),
        "minimum_recoil_transfer_cutoffs_ev": sorted(cutoffs),
        "interaction_domain": (
            "full retained ZBL disk from b=0 to the selected recoil cutoff"
        ),
        "runtime_exclusivity": (
            "zbl_full replaces nlh_hard and HTran in a trajectory run"
        ),
        "excluded_physics": [
            "electronic_stopping",
            "excitation",
            "ionisation",
            "charge_exchange",
            "molecular_polarisation",
            "dynamic_target_relaxation",
        ],
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["configuration_sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def write_manifests(
    output_directory: Path,
    phases: Sequence[str],
    *,
    projectiles: Sequence[str] = DEFAULT_PROJECTILES,
    energy_bounds_ev: tuple[float, float] = DEFAULT_ENERGY_BOUNDS_EV,
    minimum_transfer_cutoffs_ev: Sequence[float] = DEFAULT_CUTOFFS_EV,
) -> tuple[Path, ...]:
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for phase_id in phases:
        manifest = build_manifest(
            phase_id,
            projectiles=projectiles,
            energy_bounds_ev=energy_bounds_ev,
            minimum_transfer_cutoffs_ev=minimum_transfer_cutoffs_ev,
        )
        path = output_directory / f"zbl_full_{phase_id}.manifest.json"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_directory,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
        paths.append(path)
    return tuple(paths)


def load_transport(
    manifest_path: Path,
    *,
    structure_index: int,
    minimum_transfer_ev: float,
) -> PeriodicHardCollisionTransport:
    """Instantiate the existing atomistic trajectory engine with full ZBL."""
    manifest = json.loads(manifest_path.resolve().read_text(encoding="utf-8"))
    if manifest.get("backend") != "atomistic_periodic_full_zbl":
        raise ValueError("Manifest does not describe the atomistic full-ZBL backend.")
    cutoff = float(minimum_transfer_ev)
    if cutoff not in manifest["minimum_recoil_transfer_cutoffs_ev"]:
        raise ValueError("Requested cutoff is absent from the backend manifest.")
    records = manifest["structures"]
    try:
        record = records[structure_index]
    except (IndexError, TypeError) as exc:
        raise ValueError("Invalid structure index.") from exc
    source = REPOSITORY_ROOT / record["path"]
    if file_sha256(source) != record["sha256"]:
        raise ValueError("Backend structure checksum mismatch.")
    structure = load_ice_structure(
        source,
        frame_index=int(record["frame_index"]),
        metadata_path=REPOSITORY_ROOT / record["metadata"],
    )
    kernel = FullZBLKernel(
        minimum_transfer_ev=cutoff,
        projectiles=tuple(manifest["projectiles"]),
        energy_bounds_ev=tuple(manifest["total_kinetic_energy_bounds_ev"]),
    )
    return PeriodicHardCollisionTransport(structure, kernel)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=tuple(PHASE_REGISTRIES),
        default=tuple(PHASE_REGISTRIES),
    )
    parser.add_argument(
        "--projectiles",
        nargs="+",
        choices=DEFAULT_PROJECTILES,
        default=DEFAULT_PROJECTILES,
    )
    parser.add_argument("--energy-min-ev", type=float, default=1.0e4)
    parser.add_argument("--energy-max-ev", type=float, default=1.0e8)
    parser.add_argument(
        "--transfer-cutoffs-ev", nargs="+", type=float, default=DEFAULT_CUTOFFS_EV
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    paths = write_manifests(
        arguments.output_directory,
        tuple(arguments.phases),
        projectiles=tuple(arguments.projectiles),
        energy_bounds_ev=(arguments.energy_min_ev, arguments.energy_max_ev),
        minimum_transfer_cutoffs_ev=tuple(arguments.transfer_cutoffs_ev),
    )
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
