#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TABLE="${SCRIPT_DIR}/per_energy_runs.csv"
ROOT_DIR="${SCRIPT_DIR}/root"
mkdir -p "${ROOT_DIR}"

DNAPHYSICS_BIN_INPUT="${1:-${SCRIPT_DIR}/../build/dnaphysics}"
THREADS="${2:-12}"
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

echo "Running per-energy Europa library with DNA_PHYSICS=${PHYSICS}, threads=${THREADS}"

tail -n +2 "${RUN_TABLE}" | while IFS=, read -r energy_index e_low_mev e_high_mev e_center_mev dE_mev sim_particles macro_relpath root_basename root_relpath csv_physics csv_density; do
  macro_path="${SCRIPT_DIR}/${macro_relpath}"
  root_path="${SCRIPT_DIR}/${root_relpath}"
  if [[ ! -f "${macro_path}" ]]; then
    echo "Missing macro: ${macro_path}" >&2
    exit 1
  fi
  if [[ -f "${root_path}" ]]; then
    echo "[energy ${energy_index}] SKIP existing ${root_relpath}"
    continue
  fi
  echo "[energy ${energy_index}] E=${e_center_mev} MeV -> ${root_relpath}"
  (
    cd "${ROOT_DIR}"
    DNA_PHYSICS="${PHYSICS}" DNA_NTUPLE_FILES=0 DNA_ROOT_BASENAME="${root_basename}" "${DNAPHYSICS_BIN}" "${macro_path}" "${THREADS}"
  )
done
