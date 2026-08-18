"""General low-energy single-electron capture framework."""

from .channels import (
    WATER_GROUND_STATE_DONOR,
    ElectronDonorState,
    SingleElectronCaptureChannel,
    build_single_capture_channels,
    coupled_multiplicities,
)
from .landau_zener import (
    impact_parameter_cross_section_cm2,
    transition_probability,
)
from .workflow import (
    BRANCH_HANDOFF_PENDING,
    SCHEMA_VERSION,
    build_workflow,
    collect_workflow,
    load_workflow_manifest,
    run_workflow_tasks,
)

__all__ = [
    "BRANCH_HANDOFF_PENDING",
    "ElectronDonorState",
    "SCHEMA_VERSION",
    "SingleElectronCaptureChannel",
    "WATER_GROUND_STATE_DONOR",
    "build_single_capture_channels",
    "build_workflow",
    "collect_workflow",
    "coupled_multiplicities",
    "impact_parameter_cross_section_cm2",
    "load_workflow_manifest",
    "run_workflow_tasks",
    "transition_probability",
]
