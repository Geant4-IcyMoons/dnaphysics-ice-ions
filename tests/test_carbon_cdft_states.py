from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


NEP_DIR = (
    Path(__file__).resolve().parents[1]
    / "python_scripts"
    / "physics_ice"
    / "nep_mbpol"
)
if str(NEP_DIR) not in sys.path:
    sys.path.insert(0, str(NEP_DIR))

from ion_ice.schema import load_projectile_definition


CARBON_PATH = NEP_DIR / "ion_ice" / "species" / "C.json"
HELIUM_PATH = NEP_DIR / "ion_ice" / "species" / "He.json"


def _carbon_payload() -> dict[str, object]:
    return json.loads(CARBON_PATH.read_text(encoding="utf-8"))


def _write_payload(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "C.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_carbon_states_declare_spin_mode_and_exact_fresh_atomic_guesses():
    carbon = load_projectile_definition(CARBON_PATH)

    assert [state.scf_spin_mode for state in carbon.states] == [
        "UNRESTRICTED",
        "UNRESTRICTED",
        "RESTRICTED",
        "UNRESTRICTED",
        "RESTRICTED",
        "UNRESTRICTED",
        "RESTRICTED",
    ]
    assert [state.atomic_guess_dict() for state in carbon.states] == [
        {
            "alpha": [{"n": 2, "l": 1, "nel": 2}],
            "beta": [{"n": 2, "l": 1, "nel": -2}],
        },
        {
            "alpha": [],
            "beta": [{"n": 2, "l": 1, "nel": -2}],
        },
        {
            "alpha": [{"n": 2, "l": 1, "nel": -2}],
            "beta": [{"n": 2, "l": 1, "nel": -2}],
        },
        {
            "alpha": [{"n": 2, "l": 1, "nel": -2}],
            "beta": [
                {"n": 2, "l": 0, "nel": -2},
                {"n": 2, "l": 1, "nel": -2},
            ],
        },
        {
            "alpha": [
                {"n": 2, "l": 0, "nel": -2},
                {"n": 2, "l": 1, "nel": -2},
            ],
            "beta": [
                {"n": 2, "l": 0, "nel": -2},
                {"n": 2, "l": 1, "nel": -2},
            ],
        },
        {
            "alpha": [
                {"n": 2, "l": 0, "nel": -2},
                {"n": 2, "l": 1, "nel": -2},
            ],
            "beta": [
                {"n": 1, "l": 0, "nel": -2},
                {"n": 2, "l": 0, "nel": -2},
                {"n": 2, "l": 1, "nel": -2},
            ],
        },
        {
            "alpha": [
                {"n": 1, "l": 0, "nel": -2},
                {"n": 2, "l": 0, "nel": -2},
                {"n": 2, "l": 1, "nel": -2},
            ],
            "beta": [
                {"n": 1, "l": 0, "nel": -2},
                {"n": 2, "l": 0, "nel": -2},
                {"n": 2, "l": 1, "nel": -2},
            ],
        },
    ]


def test_carbon_guesses_initialize_declared_shell_populations_and_spin():
    carbon = load_projectile_definition(CARBON_PATH)
    neutral_spherical = {
        "alpha": {(1, 0): 1.0, (2, 0): 1.0, (2, 1): 1.0},
        "beta": {(1, 0): 1.0, (2, 0): 1.0, (2, 1): 1.0},
    }
    expected = [
        ({(1, 0): 1, (2, 0): 1, (2, 1): 2}, {(1, 0): 1, (2, 0): 1}),
        ({(1, 0): 1, (2, 0): 1, (2, 1): 1}, {(1, 0): 1, (2, 0): 1}),
        ({(1, 0): 1, (2, 0): 1}, {(1, 0): 1, (2, 0): 1}),
        ({(1, 0): 1, (2, 0): 1}, {(1, 0): 1}),
        ({(1, 0): 1}, {(1, 0): 1}),
        ({(1, 0): 1}, {}),
        ({}, {}),
    ]

    for state, (expected_alpha, expected_beta) in zip(carbon.states, expected):
        occupations = {
            spin: dict(values) for spin, values in neutral_spherical.items()
        }
        for spin, changes in (
            ("alpha", state.atomic_guess_alpha),
            ("beta", state.atomic_guess_beta),
        ):
            for n, angular_momentum, electrons in changes:
                orbital = (n, angular_momentum)
                # CP2K 2025.2 init_atom_electronic_state applies NEL/2 to
                # the spin-resolved half of the neutral spherical shell.
                occupations[spin][orbital] += electrons / 2
                if occupations[spin][orbital] == 0:
                    occupations[spin].pop(orbital)

        assert occupations["alpha"] == expected_alpha
        assert occupations["beta"] == expected_beta
        alpha_count = sum(occupations["alpha"].values())
        beta_count = sum(occupations["beta"].values())
        assert alpha_count + beta_count == state.electrons_on_projectile
        assert alpha_count - beta_count == state.multiplicity - 1

    q1_guess = carbon.state(1).atomic_guess_dict()
    assert q1_guess == {
        "alpha": [],
        "beta": [{"n": 2, "l": 1, "nel": -2}],
    }
    assert all(
        "m" not in change
        for spin_changes in q1_guess.values()
        for change in spin_changes
    )


def test_carbon_requires_explicit_spin_modes_and_ionized_atomic_guesses(tmp_path):
    payload = _carbon_payload()
    states = payload["components"]["soft_dft"]["states"]
    states[0].pop("scf_spin_mode")
    with pytest.raises(ValueError, match="explicit scf_spin_mode"):
        load_projectile_definition(_write_payload(tmp_path, payload))

    payload = _carbon_payload()
    states = payload["components"]["soft_dft"]["states"]
    states[0].pop("cp2k_atomic_guess")
    with pytest.raises(ValueError, match="explicit CP2K atomic-guess"):
        load_projectile_definition(_write_payload(tmp_path, payload))

    payload = _carbon_payload()
    states = payload["components"]["soft_dft"]["states"]
    states[1]["cp2k_atomic_guess"]["beta"][0]["nel"] = -1
    with pytest.raises(ValueError, match="sum to -2q"):
        load_projectile_definition(_write_payload(tmp_path, payload))

    payload = _carbon_payload()
    states = payload["components"]["soft_dft"]["states"]
    states[1]["cp2k_atomic_guess"] = {
        "alpha": [{"n": 2, "l": 1, "nel": -2}],
        "beta": [],
    }
    with pytest.raises(ValueError, match="alpha-majority multiplicity"):
        load_projectile_definition(_write_payload(tmp_path, payload))


def test_spin_mode_validation_and_noncarbon_backward_compatibility(tmp_path):
    helium = load_projectile_definition(HELIUM_PATH)
    assert [state.scf_spin_mode for state in helium.states] == [
        "RESTRICTED",
        "UNRESTRICTED",
        "RESTRICTED",
    ]

    payload = _carbon_payload()
    states = payload["components"]["soft_dft"]["states"]
    states[0]["scf_spin_mode"] = "RESTRICTED"
    with pytest.raises(ValueError, match="multiplicity-one"):
        load_projectile_definition(_write_payload(tmp_path, payload))

    payload = _carbon_payload()
    states = payload["components"]["soft_dft"]["states"]
    states[0]["scf_spin_mode"] = "UKS"
    with pytest.raises(ValueError, match="RESTRICTED or UNRESTRICTED"):
        load_projectile_definition(_write_payload(tmp_path, payload))
