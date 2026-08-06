"""Species-general definitions of low-energy single-electron capture channels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from ion_ice import IonChargeState, ProjectileDefinition


@dataclass(frozen=True)
class ElectronDonorState:
    """One target state reached after donating one electron to a projectile."""

    target: str
    initial_charge: int
    initial_multiplicity: int
    product_charge: int
    product_multiplicity: int
    donor_orbital: str
    product_electronic_state: str
    provenance: dict[str, str]

    def validate(self) -> None:
        if self.product_charge != self.initial_charge + 1:
            raise ValueError(
                "A single-electron donor state must increase target charge by one."
            )
        if self.initial_multiplicity < 1 or self.product_multiplicity < 1:
            raise ValueError("Target multiplicities must be positive.")
        if not self.target.strip() or not self.donor_orbital.strip():
            raise ValueError("Target and donor-orbital labels cannot be empty.")
        if not self.product_electronic_state.strip():
            raise ValueError("The ionized target electronic state cannot be empty.")
        if not self.provenance.get("source") or not any(
            self.provenance.get(key) for key in ("doi", "url")
        ):
            raise ValueError(
                "Target-state provenance requires a source and DOI or URL."
            )

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


WATER_GROUND_STATE_DONOR = ElectronDonorState(
    target="H2O",
    initial_charge=0,
    initial_multiplicity=1,
    product_charge=1,
    product_multiplicity=2,
    donor_orbital="1b1",
    product_electronic_state="X 2B1",
    provenance={
        "source": "NIST Chemistry WebBook SRD 69, H2O+ spectroscopy",
        "url": "https://webbook.nist.gov/cgi/cbook.cgi?ID=C56583621&Mask=1980",
        "definition": (
            "The smallest capture model removes one electron from neutral H2O "
            "and represents the ground X 2B1 state of H2O+."
        ),
    },
)


def coupled_multiplicities(
    first_multiplicity: int, second_multiplicity: int
) -> tuple[int, ...]:
    """Return every total multiplicity allowed by coupling two fragment spins."""

    first = int(first_multiplicity)
    second = int(second_multiplicity)
    if first < 1 or second < 1:
        raise ValueError("Spin multiplicities must be positive integers.")
    twice_first_spin = first - 1
    twice_second_spin = second - 1
    return tuple(
        twice_total_spin + 1
        for twice_total_spin in range(
            abs(twice_first_spin - twice_second_spin),
            twice_first_spin + twice_second_spin + 1,
            2,
        )
    )


def _ion_label(symbol: str, charge: int) -> str:
    if charge == 0:
        return symbol
    if charge == 1:
        return f"{symbol}+"
    return f"{symbol}{charge}+"


@dataclass(frozen=True)
class SingleElectronCaptureChannel:
    """One spin-conserving ``X(q+) -> X((q-1)+)`` capture candidate."""

    projectile: str
    atomic_number: int
    incident_state: IonChargeState
    product_state: IonChargeState
    target_state: ElectronDonorState
    total_charge: int
    total_multiplicity: int
    allowed_product_total_multiplicities: tuple[int, ...]
    spin_allowed: bool

    @property
    def channel_id(self) -> str:
        return (
            f"{self.projectile}_q{self.incident_state.charge}_to_"
            f"q{self.product_state.charge}_capture_{self.target_state.donor_orbital}"
        )

    @property
    def reaction(self) -> str:
        initial_target = _ion_label(
            self.target_state.target, self.target_state.initial_charge
        )
        product_target = _ion_label(
            self.target_state.target, self.target_state.product_charge
        )
        return (
            f"{_ion_label(self.projectile, self.incident_state.charge)} + "
            f"{initial_target} -> "
            f"{_ion_label(self.projectile, self.product_state.charge)} + "
            f"{product_target}"
        )

    def validate(self) -> None:
        self.target_state.validate()
        if self.incident_state.charge != self.product_state.charge + 1:
            raise ValueError("Single capture must reduce projectile charge by one.")
        if self.total_charge != (
            self.incident_state.charge + self.target_state.initial_charge
        ):
            raise ValueError("Entrance fragments do not conserve total charge.")
        product_total_charge = (
            self.product_state.charge + self.target_state.product_charge
        )
        if self.total_charge != product_total_charge:
            raise ValueError("Capture products do not conserve total charge.")
        if self.total_multiplicity != self.incident_state.multiplicity:
            raise ValueError(
                "Neutral singlet H2O requires the entrance total multiplicity "
                "to equal the incident projectile multiplicity."
            )
        expected = coupled_multiplicities(
            self.product_state.multiplicity,
            self.target_state.product_multiplicity,
        )
        if self.allowed_product_total_multiplicities != expected:
            raise ValueError("Stored product spin couplings are inconsistent.")
        if self.spin_allowed != (self.total_multiplicity in expected):
            raise ValueError("Stored spin-allowance flag is inconsistent.")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "channel_id": self.channel_id,
            "process": "single_electron_capture",
            "reaction": self.reaction,
            "projectile": self.projectile,
            "atomic_number": self.atomic_number,
            "incident_state": self.incident_state.as_dict(),
            "product_state": self.product_state.as_dict(),
            "target_state": self.target_state.as_dict(),
            "total_charge": self.total_charge,
            "total_multiplicity": self.total_multiplicity,
            "allowed_product_total_multiplicities": list(
                self.allowed_product_total_multiplicities
            ),
            "spin_allowed": self.spin_allowed,
            "status": (
                "runnable_ground_channel"
                if self.spin_allowed
                else "blocked_no_spin_conserving_ground_channel"
            ),
            "state_definition": (
                "Both diabatic states have the same nuclei, total charge, and "
                "total multiplicity. Only the constrained projectile electron "
                "population changes by one."
            ),
        }


def build_single_capture_channels(
    projectile: ProjectileDefinition,
    incident_charges: Iterable[int] | None = None,
    *,
    target_state: ElectronDonorState = WATER_GROUND_STATE_DONOR,
    require_spin_allowed: bool = True,
) -> tuple[SingleElectronCaptureChannel, ...]:
    """Build all requested non-negative single-capture steps from a charge ladder."""

    projectile.validate()
    target_state.validate()
    requested = tuple(
        range(1, projectile.atomic_number + 1)
        if incident_charges is None
        else (int(charge) for charge in incident_charges)
    )
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Select at least one unique incident charge state.")
    if any(charge < 1 or charge > projectile.atomic_number for charge in requested):
        raise ValueError(
            f"Single capture for {projectile.symbol} requires incident q=1.."
            f"{projectile.atomic_number}; q=0 is the neutral endpoint."
        )

    channels: list[SingleElectronCaptureChannel] = []
    for charge in sorted(requested):
        incident = projectile.state(charge)
        product = projectile.state(charge - 1)
        allowed = coupled_multiplicities(
            product.multiplicity, target_state.product_multiplicity
        )
        channel = SingleElectronCaptureChannel(
            projectile=projectile.symbol,
            atomic_number=projectile.atomic_number,
            incident_state=incident,
            product_state=product,
            target_state=target_state,
            total_charge=charge + target_state.initial_charge,
            total_multiplicity=incident.multiplicity,
            allowed_product_total_multiplicities=allowed,
            spin_allowed=incident.multiplicity in allowed,
        )
        channel.validate()
        channels.append(channel)

    blocked = [channel.channel_id for channel in channels if not channel.spin_allowed]
    if blocked and require_spin_allowed:
        raise ValueError(
            "No spin-conserving ground-channel representation for: "
            + ", ".join(blocked)
        )
    return tuple(channels)
