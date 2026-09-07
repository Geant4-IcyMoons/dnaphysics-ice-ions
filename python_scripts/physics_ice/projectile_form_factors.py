"""Frozen, spherical projectile electron densities for first-Born tables.

F_Q(k) = 4*pi*integral r^2*n_Q(r)*j0(k*r) dr; F_Q(0) = Z-Q.
The coherent projectile charge amplitude is Z-F_Q(k), not a scalar Zeff.
See PROJECTILE_FORM_FACTORS.md for the fixed-state approximation, sources,
RPWBA momentum convention, and processes that this model does not describe.
Lengths and wave numbers here are in bohr and inverse bohr. No target-ELF
normalization, charge-exchange physics, or Barkas correction is included.
"""

from dataclasses import dataclass
from functools import cached_property, lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import gamma, poch

from constants import C_AU, EH, PROJECTILE_LIBRARY

VERSION = "frozen-spherical-hf-v1"
DATA_PATH = Path(__file__).with_name("projectile_atomic_data.json")
ELEMENTS = {cfg["element"]: int(cfg["charge"]) for cfg in PROJECTILE_LIBRARY.values()}
DEFAULT_WORKERS = 10


def element_symbol(projectile):
    key = str(projectile).strip().lower()
    for name, cfg in PROJECTILE_LIBRARY.items():
        if key in (name, cfg["element"].lower(), *cfg["aliases"]):
            return cfg["element"]
    raise ValueError(f"Unsupported projectile {projectile!r}; use H, He, C, O, S.")


def validate_charge(element, charge):
    z = ELEMENTS[element_symbol(element)]
    value = float(charge)
    if not np.isfinite(value) or value != int(value) or not 0 <= value <= z:
        raise ValueError(f"Charge state must be an integer from 0 to {z}, got {charge!r}.")
    return int(value)


def screening_momentum(k_au, loss_eV=0.0, *, relativistic=False):
    """PWBA k, or RPWBA spacelike invariant sqrt(k^2-(W/hbar*c)^2).

    RPWBA uses an electric elastic form factor in the Breit/static-density
    approximation. In the negligible-projectile-recoil limit this is also
    the rest-frame wave number of the Lorentz-contracted rigid cloud.
    This is not a covariant many-electron calculation of internal transitions.
    """
    k, w = np.broadcast_arrays(np.asarray(k_au, float), np.asarray(loss_eV, float))
    if np.any(~np.isfinite(k)) or np.any(k < 0) or np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("Momentum and energy transfer must be finite and nonnegative.")
    if not relativistic:
        return k
    wc = w / (C_AU * EH)
    delta = (k - wc) * (k + wc)
    if np.any(delta < -1e-13 * np.maximum(k*k, wc*wc)):
        raise ValueError("RPWBA projectile form factor requires spacelike momentum transfer.")
    return np.sqrt(np.maximum(delta, 0.0))


@dataclass(frozen=True)
class ProjectileDensity:
    element: str
    charge: int
    record: dict

    @property
    def z(self):
        return ELEMENTS[self.element]

    @property
    def electrons(self):
        return self.z - self.charge

    @cached_property
    def terms(self):
        return np.asarray(self.record.get("gaussian_terms", []), float).reshape(-1, 3)

    def moment(self, order):
        """Integral of r**order times the electron density (not per electron)."""
        if order < 0:
            raise ValueError("Only nonnegative radial moments are supported.")
        if self.electrons == 0:
            return 0.0
        if self.electrons == 1:
            return float(gamma(order + 3) / (2.0 * (2*self.z)**order))
        l, a, weight = self.terms.T
        return float(np.sum(weight * gamma(l + 1.5 + order/2) / gamma(l + 1.5) / a**(order/2)))

    def density(self, r_bohr):
        """Spherical electron number density in bohr**-3."""
        r = np.asarray(r_bohr, float)
        if np.any(~np.isfinite(r)) or np.any(r < 0):
            raise ValueError("Radius must be finite and nonnegative.")
        if self.electrons == 0:
            return np.zeros_like(r)
        if self.electrons == 1:
            return self.z**3 / np.pi * np.exp(-2*self.z*r)
        out = np.zeros_like(r)
        for l, a, weight in self.terms:
            out += weight * a**(l+1.5)/(2*np.pi*gamma(l+1.5)) * r**(2*l) * np.exp(-a*r*r)
        return out

    def _deficit_direct(self, k):
        """N-F(k), evaluated without subtracting nearly equal neutral charges."""
        k = np.asarray(k, float)
        if self.electrons == 0:
            return np.zeros_like(k)
        if self.electrons == 1:
            x = (k/(2*self.z))**2
            # 1 - (1+x)**-2, stable at k=0 and at large k.
            return -np.expm1(-2*np.log1p(x))
        out = np.empty_like(k)
        small = k*np.sqrt(self.moment(2)/self.electrons) < 1e-3
        out[small] = k[small]**2 * (self.moment(2)/6 - k[small]**2*self.moment(4)/120)
        q = k[~small]
        value = np.zeros_like(q)
        for l, a, weight in self.terms:
            x = q*q/(4*a)
            # Kummer transform: 1F1(l+3/2;3/2;-x) = exp(-x) P_l(x).
            # P_l-1 is evaluated separately to retain its small-x precision.
            xm = np.minimum(x, 700.0)
            pminus = np.zeros_like(xm)
            for j in range(1, int(l)+1):
                pminus += poch(-int(l), j)/poch(1.5, j)/gamma(j+1) * xm**j
            deficit = -np.expm1(-xm) - np.exp(-xm)*pminus
            value += weight * deficit
        out[~small] = value
        return out

    @cached_property
    def _interpolator(self):
        k = np.geomspace(1e-5, 1e5, 8193)
        deficit = self._deficit_direct(k)
        if np.any(~np.isfinite(deficit)) or np.any(deficit <= 0):
            raise ValueError("Invalid atomic form-factor deficit; rebuild the atomic data.")
        return PchipInterpolator(np.log(k), np.log(deficit), extrapolate=False)

    def electron_deficit(self, k_au, *, direct=False):
        k = np.asarray(k_au, float)
        if np.any(~np.isfinite(k)) or np.any(k < 0):
            raise ValueError("Wave number must be finite and nonnegative.")
        if direct or self.electrons <= 1:
            return self._deficit_direct(k)
        out = np.empty_like(k)
        inside = (k >= 1e-5) & (k <= 1e5)
        out[inside] = np.exp(self._interpolator(np.log(k[inside])))
        out[~inside] = self._deficit_direct(k[~inside])
        return out

    def form_factor(self, k_au, *, direct=False):
        return self.electrons - self.electron_deficit(k_au, direct=direct)

    def charge_amplitude(self, k_au):
        return self.charge + self.electron_deficit(k_au)

    def squared_charge(self, k_au):
        return self.charge_amplitude(k_au)**2


@lru_cache(maxsize=1)
def _atomic_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing {DATA_PATH}; run generate_projectile_atomic_data.py.")
    raw = DATA_PATH.read_bytes()
    data = json.loads(raw)
    if data.get("version") != VERSION:
        raise ValueError("Unsupported projectile atomic-data version.")
    return data, hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=64)
def load_density(projectile, charge):
    element = element_symbol(projectile)
    charge = validate_charge(element, charge)
    n = ELEMENTS[element] - charge
    if n <= 1:
        record = {"method": "bare" if n == 0 else "hydrogenic-1s", "energy_hartree": -ELEMENTS[element]**2/2 if n else 0.0}
    else:
        data, _ = _atomic_data()
        key = f"{element}:{charge}"
        if key not in data["states"]:
            raise ValueError(f"Missing atomic density for {key}; no effective-charge fallback is permitted.")
        record = data["states"][key]
    density = ProjectileDensity(element, charge, record)
    if n > 1 and (
        not len(density.terms) or not np.all(np.isfinite(density.terms))
        or np.any(density.terms[:, 1] <= 0)
        or np.any(density.terms[:, 0] != np.floor(density.terms[:, 0]))
        or np.any(density.terms[:, 0] < 0)
        or not np.isfinite(record["energy_hartree"])
    ):
        raise ValueError(f"Invalid atomic density record for {element}:{charge}.")
    if not np.isclose(density.moment(0), n, rtol=0, atol=1e-8):
        raise ValueError(f"Incorrect electron normalization for {element}:{charge}.")
    return density


def density_metadata(density, *, relativistic=False):
    return {
        "projectile_form_factor_model": VERSION,
        "projectile_element": density.element,
        "projectile_nuclear_charge": density.z,
        "projectile_charge_state": density.charge,
        "projectile_bound_electrons": density.electrons,
        "projectile_electronic_state": "ground-configuration-spherical-frozen-density",
        "projectile_atomic_method": density.record["method"],
        "projectile_atomic_data_sha256": _atomic_data()[1] if density.electrons > 1 else "analytic",
        "projectile_screening_momentum": "spacelike-invariant-Breit-static" if relativistic else "lab-wave-number",
        "projectile_internal_transitions_included": False,
        "projectile_charge_exchange_included": False,
    }
