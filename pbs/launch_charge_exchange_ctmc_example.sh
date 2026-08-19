#!/usr/bin/env bash
#
# Example multi-node launcher for the element-specific CTMC PBS scripts.
#
# Usage:
#   export START_SEPARATION_AU='convergence-derived values'
#   export TARGET_BMAX_AU='five convergence-derived values'
#   export LOSS_BMAX_AU='element-specific convergence-derived values'
#
#   bash pbs/launch_charge_exchange_ctmc_example.sh plan carbon 84
#   bash pbs/launch_charge_exchange_ctmc_example.sh submit carbon 84
#   # After all compute shards finish successfully:
#   bash pbs/launch_charge_exchange_ctmc_example.sh merge carbon 84
#   bash pbs/launch_charge_exchange_ctmc_example.sh refine-base carbon 84
#   # After all unique base curves finish successfully:
#   bash pbs/launch_charge_exchange_ctmc_example.sh refine-intervals carbon 84
#   # After all adaptive intervals finish successfully:
#   bash pbs/launch_charge_exchange_ctmc_example.sh merge-refined carbon 84
#
# This launcher intentionally contains no default boundary/cutoff values.

set -euo pipefail

usage() {
    printf '%s\n' \
        "Usage: $0 {plan|submit|merge|refine-base|refine-intervals|merge-refined} {carbon|lithium|oxygen|sulfur} SHARD_COUNT" \
        "" \
        "Required exported variables:" \
        "  START_SEPARATION_AU  one convergence-derived value per energy" \
        "  TARGET_BMAX_AU       five convergence-derived H2O-orbital values" \
        "  LOSS_BMAX_AU         element-specific projectile-loss values" \
        "" \
        "PBS_ARRAY_MAX defaults to 50 and controls array splitting."
}

if (( $# != 3 )); then
    usage >&2
    exit 2
fi

action="$1"
atom="$2"
shard_count="$3"

case "${action}" in
    plan|submit|merge|refine-base|refine-intervals|merge-refined) ;;
    *)
        printf 'Unknown action: %s\n' "${action}" >&2
        usage >&2
        exit 2
        ;;
esac

case "${atom}" in
    carbon|lithium|oxygen|sulfur) ;;
    *)
        printf 'Unknown element: %s\n' "${atom}" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ ! "${shard_count}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'SHARD_COUNT must be a positive integer: %s\n' "${shard_count}" >&2
    exit 2
fi

array_max="${PBS_ARRAY_MAX:-50}"
if [[ ! "${array_max}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'PBS_ARRAY_MAX must be a positive integer: %s\n' "${array_max}" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
pbs_script="${repo_root}/pbs/generate_${atom}_charge_exchange_ctmc.pbs"
if [[ ! -f "${pbs_script}" ]]; then
    printf 'Missing element PBS script: %s\n' "${pbs_script}" >&2
    exit 1
fi

required_variables=(
    START_SEPARATION_AU
    TARGET_BMAX_AU
    LOSS_BMAX_AU
)
for variable_name in "${required_variables[@]}"; do
    if [[ -z "${!variable_name:-}" ]]; then
        printf 'Required variable is empty or unset: %s\n' \
            "${variable_name}" >&2
        exit 2
    fi
    export "${variable_name}"
done

optional_variables=(
    MINIMUM_INTEGRATION_TIME_AU
    CONDA_SH
    CONDA_ENV
    PYTHON_BIN
    TRAJECTORY_CHUNK_SIZE
    PENDING_FACTOR
    CHECKPOINT_EVERY
    CHECKPOINT_SECONDS
    RTOL
    ATOL
    RETRY_RTOL
    RETRY_ATOL
    MAXIMUM_INTEGRATION_STEPS
    MAXIMUM_RELATIVE_ENERGY_DRIFT
    BOUNDARY_EXTENSION_FACTOR
    ADAPTIVE_AXIS_RELATIVE_TOLERANCE
    ADAPTIVE_COMBINED_RELATIVE_TOLERANCE
    ADAPTIVE_STATISTICAL_RELATIVE_TOLERANCE
    ADAPTIVE_STATISTICAL_CONFIDENCE
    ADAPTIVE_MAX_IMPACT_LEVELS
    ADAPTIVE_MAX_ENERGY_LEVELS
    ADAPTIVE_MAX_SAMPLING_LEVELS
    INITIAL_ENSEMBLE
    OWNERSHIP_MANIFEST
    EXTRA_ARGS
    REPO_ROOT
)

forwarded_variables=("${required_variables[@]}")
for variable_name in "${optional_variables[@]}"; do
    if [[ -n "${!variable_name+x}" ]]; then
        export "${variable_name}"
        forwarded_variables+=("${variable_name}")
    fi
done

variable_list="$(IFS=,; printf '%s' "${forwarded_variables[*]}")"

if [[ "${action}" != "plan" ]] && ! command -v qsub >/dev/null 2>&1; then
    printf 'qsub is not available in PATH\n' >&2
    exit 127
fi

cd "${repo_root}"

if [[ "${action}" == "merge" || "${action}" == "merge-refined" ]]; then
    if (( shard_count < 2 )); then
        printf '%s\n' \
            "A one-shard run writes final outputs directly; no merge is needed." >&2
        exit 2
    fi
    if [[ "${action}" == "merge" ]]; then
        mode_variable="MERGE_ONLY=1"
        merge_label="base-grid merge"
    else
        mode_variable="MERGE_ADAPTIVE_ONLY=1"
        merge_label="adaptive-interval merge"
    fi
    export_spec="${variable_list},SHARD_COUNT=${shard_count},${mode_variable}"
    printf 'Submitting one-core %s for %s (%s shards)\n' \
        "${merge_label}" "${atom}" "${shard_count}"
    merge_job="$(qsub \
        -l select=1:ncpus=1:mem=4gb \
        -v "${export_spec}" \
        "${pbs_script}")"
    printf '%s\n' "${merge_job}"
    if [[ "${action}" == "merge-refined" && "${atom}" == "carbon" ]]; then
        benchmark_input="${BENCHMARK_INPUT_DIR:-${repo_root}/cross_sections/carbon_charge_exchange}"
        benchmark_output="${BENCHMARK_OUTPUT_DIR:-${repo_root}/python_scripts/physics_ice/process_evidence/charge_exchange_ctmc/benchmarking/runs/carbon_charge_exchange}"
        formal_archive="${FORMAL_REFERENCE_ARCHIVE:-${repo_root}/C3_100keVpu.zip}"
        benchmark_job="$(qsub \
            -N carbon_ctmc_benchmark \
            -W "depend=afterok:${merge_job}" \
            -v "REPO_ROOT=${repo_root},INPUT_DIR=${benchmark_input},OUTPUT_DIR=${benchmark_output},FORMAL_REFERENCE_ARCHIVE=${formal_archive}" \
            "${repo_root}/pbs/benchmark_carbon_charge_exchange_ctmc.pbs")"
        printf 'Carbon validation benchmark: %s after %s\n' \
            "${benchmark_job}" "${merge_job}"
    fi
    exit 0
fi

printf '%s\n' \
    "Element: ${atom}" \
    "Total shards: ${shard_count}" \
    "Cores per running shard: 64" \
    "Maximum requested cores: $((shard_count * 64))" \
    "PBS array batch limit: ${array_max}" \
    "PBS script: ${pbs_script}"

offset=0
while (( offset < shard_count )); do
    remaining=$((shard_count - offset))
    batch_size="${array_max}"
    if (( remaining < batch_size )); then
        batch_size="${remaining}"
    fi
    array_last=$((batch_size - 1))
    export_spec="${variable_list},SHARD_COUNT=${shard_count},SHARD_OFFSET=${offset}"
    if [[ "${action}" == "refine-base" ]]; then
        export_spec="${export_spec},ADAPTIVE_BASE_CURVES_ONLY=1"
    elif [[ "${action}" == "refine-intervals" ]]; then
        export_spec="${export_spec},ADAPTIVE_INTERVALS_ONLY=1"
    fi

    if [[ "${action}" == "plan" ]]; then
        if (( batch_size == 1 )); then
            printf 'Would submit shard index %d as a standalone job\n' \
                "${offset}"
        else
            printf 'Would submit shard indices %d--%d as array 0-%d\n' \
                "${offset}" "$((offset + batch_size - 1))" "${array_last}"
        fi
    else
        if (( batch_size == 1 )); then
            printf 'Submitting shard index %d as a standalone job: ' "${offset}"
            qsub \
                -v "${export_spec}" \
                "${pbs_script}"
        else
            printf 'Submitting shard indices %d--%d as array 0-%d: ' \
                "${offset}" "$((offset + batch_size - 1))" "${array_last}"
            qsub \
                -J "0-${array_last}" \
                -v "${export_spec}" \
                "${pbs_script}"
        fi
    fi
    offset=$((offset + batch_size))
done

if [[ "${action}" == "submit" ]]; then
    printf '%s\n' \
        "Base-grid shards submitted. After they finish, run merge, refine-base," \
        "refine-intervals, and merge-refined in that order."
elif [[ "${action}" == "refine-base" ]]; then
    printf '%s\n' \
        "Unique adaptive base curves submitted. Do not submit refine-intervals" \
        "until every base-curve array element has completed successfully."
elif [[ "${action}" == "refine-intervals" ]]; then
    printf '%s\n' \
        "Adaptive intervals submitted. Do not run merge-refined until every" \
        "interval array element has completed successfully."
fi
