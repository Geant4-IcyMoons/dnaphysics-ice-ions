#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TABLE="${SCRIPT_DIR}/per_energy_runs.csv"
ROOT_DIR="${SCRIPT_DIR}/root"
CACHE_FILE="${SCRIPT_DIR}/run_per_energy.cache.csv"
mkdir -p "${ROOT_DIR}"

DNAPHYSICS_BIN_INPUT="${1:-${SCRIPT_DIR}/../build/dnaphysics}"
THREADS="${2:-10}"
PHYSICS="${3:-${DNA_PHYSICS:-ice_am}}"

if [[ "${DNAPHYSICS_BIN_INPUT}" = /* ]]; then
  DNAPHYSICS_BIN="${DNAPHYSICS_BIN_INPUT}"
else
  DNAPHYSICS_BIN="${PWD}/${DNAPHYSICS_BIN_INPUT}"
fi

if [[ ! -f "${RUN_TABLE}" ]]; then
  echo "Missing run table: ${RUN_TABLE}" >&2
  exit 1
fi

if [[ ! -x "${DNAPHYSICS_BIN}" ]]; then
  echo "dnaphysics binary not executable: ${DNAPHYSICS_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CACHE_FILE}" ]]; then
  echo "root_basename,energy_index,e_center_mev,sim_particles,threads,physics,utc_timestamp" > "${CACHE_FILE}"
fi

have_outputs() {
  local base="$1"
  compgen -G "${ROOT_DIR}/${base}*.root" > /dev/null
}

is_cached() {
  local base="$1"
  awk -F, -v b="${base}" 'NR>1 && $1==b {found=1; exit} END {exit !found}' "${CACHE_FILE}"
}

mark_cached() {
  local base="$1"
  local idx="$2"
  local ecenter="$3"
  local np="$4"
  local stamp
  stamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf "%s,%s,%s,%s,%s,%s,%s\n" "${base}" "${idx}" "${ecenter}" "${np}" "${THREADS}" "${PHYSICS}" "${stamp}" >> "${CACHE_FILE}"
}

echo "Running per-energy Europa library with DNA_PHYSICS=${PHYSICS}, threads=${THREADS}"
echo "Cache file: ${CACHE_FILE}"

tail -n +2 "${RUN_TABLE}" | while IFS=, read -r energy_index e_low_mev e_high_mev e_center_mev dE_mev sim_particles macro_relpath root_basename root_relpath csv_physics csv_density; do
  macro_path="${SCRIPT_DIR}/${macro_relpath}"
  if [[ ! -f "${macro_path}" ]]; then
    echo "Missing macro: ${macro_path}" >&2
    exit 1
  fi
  if have_outputs "${root_basename}"; then
    if ! is_cached "${root_basename}"; then
      mark_cached "${root_basename}" "${energy_index}" "${e_center_mev}" "${sim_particles}"
    fi
    echo "[energy ${energy_index}] SKIP cached/output ${root_basename}*.root"
    continue
  fi
  if is_cached "${root_basename}"; then
    echo "[energy ${energy_index}] cache entry exists but outputs are missing; rerunning ${root_basename}"
  fi
  echo "[energy ${energy_index}] E=${e_center_mev} MeV -> ${root_basename}*.root"
  (
    cd "${ROOT_DIR}"
    DNA_PHYSICS="${PHYSICS}" DNA_NTUPLE_FILES=0 DNA_ROOT_BASENAME="${root_basename}" "${DNAPHYSICS_BIN}" "${macro_path}" "${THREADS}"
  )
  if have_outputs "${root_basename}"; then
    if ! is_cached "${root_basename}"; then
      mark_cached "${root_basename}" "${energy_index}" "${e_center_mev}" "${sim_particles}"
    fi
  else
    echo "[energy ${energy_index}] WARNING: run finished but no ${root_basename}*.root was found." >&2
  fi
done
