"""Retained CHILL+ structural diagnostic; extracted without formula changes."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy import special
from scipy.spatial import cKDTree
CHILL_NEIGHBOR_CUTOFF_A = 3.5
CHILL_MINIMUM_HEXAGONAL_FRACTION = 0.99

def chill_plus(
    oxygen: NDArray[np.float64], lengths: NDArray[np.float64]
) -> tuple[dict[str, object], NDArray[np.int64], NDArray[np.float64], cKDTree]:
    """Apply the published l=3 CHILL+ bulk-polymorph definitions."""

    tree = cKDTree(oxygen, boxsize=lengths)
    distances, indices = tree.query(oxygen, k=5)
    neighbors = np.asarray(indices[:, 1:], dtype=np.int64)
    neighbor_distances = np.asarray(distances[:, 1:], dtype=np.float64)
    displacement = oxygen[neighbors] - oxygen[:, None, :]
    displacement -= np.rint(displacement / lengths) * lengths
    radii = np.linalg.norm(displacement, axis=2)
    theta = np.arccos(np.clip(displacement[:, :, 2] / radii, -1.0, 1.0))
    phi = np.mod(np.arctan2(displacement[:, :, 1], displacement[:, :, 0]), 2.0 * np.pi)
    q_lm = np.stack(
        [
            np.mean(special.sph_harm_y(3, m, theta, phi), axis=1)
            for m in range(-3, 4)
        ],
        axis=1,
    )
    norms = np.linalg.norm(q_lm, axis=1)
    correlations = np.empty_like(neighbor_distances)
    for neighbor_slot in range(4):
        neighbor_index = neighbors[:, neighbor_slot]
        correlations[:, neighbor_slot] = (
            np.sum(q_lm * np.conjugate(q_lm[neighbor_index]), axis=1).real
            / (norms * norms[neighbor_index])
        )
    staggered = np.sum(correlations <= -0.8, axis=1)
    eclipsed = np.sum(
        (correlations >= -0.35) & (correlations <= 0.25), axis=1
    )
    coordination = tree.query_ball_point(
        oxygen, CHILL_NEIGHBOR_CUTOFF_A, return_length=True
    ) - 1
    four_neighbors = coordination == 4
    hexagonal = four_neighbors & (staggered == 3) & (eclipsed == 1)
    cubic = four_neighbors & (staggered == 4) & (eclipsed == 0)
    clathrate = four_neighbors & (staggered == 0) & (eclipsed == 4)
    metrics: dict[str, object] = {
        "water_molecules": int(oxygen.shape[0]),
        "hexagonal_count": int(np.count_nonzero(hexagonal)),
        "hexagonal_fraction": float(np.mean(hexagonal)),
        "cubic_count": int(np.count_nonzero(cubic)),
        "cubic_fraction": float(np.mean(cubic)),
        "clathrate_count": int(np.count_nonzero(clathrate)),
        "other_count": int(
            oxygen.shape[0]
            - np.count_nonzero(hexagonal)
            - np.count_nonzero(cubic)
            - np.count_nonzero(clathrate)
        ),
        "four_neighbor_count": int(np.count_nonzero(four_neighbors)),
        "coordination_minimum": int(np.min(coordination)),
        "coordination_maximum": int(np.max(coordination)),
        "correlation_minimum": float(np.min(correlations)),
        "correlation_maximum": float(np.max(correlations)),
        "four_nearest_oo_distance_a": {
            "minimum": float(np.min(neighbor_distances)),
            "mean": float(np.mean(neighbor_distances)),
            "maximum": float(np.max(neighbor_distances)),
        },
        "passes_published_bulk_ih_benchmark": bool(
            np.mean(hexagonal) >= CHILL_MINIMUM_HEXAGONAL_FRACTION
            and np.all(four_neighbors)
        ),
    }
    return metrics, neighbors, neighbor_distances, tree
