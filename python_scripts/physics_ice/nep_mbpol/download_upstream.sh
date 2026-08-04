#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_DIR="${SCRIPT_DIR}/downloads"
UPSTREAM_DIR="${SCRIPT_DIR}/upstream"
MODEL_DIR="${SCRIPT_DIR}/model"
ZENODO_API="https://zenodo.org/api/records/15033656/files"

mkdir -p "${DOWNLOAD_DIR}" "${UPSTREAM_DIR}" "${MODEL_DIR}"

download_file() {
    local filename="$1"
    local expected_md5="$2"
    local destination="${DOWNLOAD_DIR}/${filename}"

    if [[ ! -f "${destination}" ]]; then
        echo "Downloading ${filename}"
        curl --fail --location --retry 3 --retry-delay 2 \
            "${ZENODO_API}/${filename}/content" \
            --output "${destination}"
    else
        echo "Using existing ${filename}"
    fi

    local actual_md5
    if command -v md5sum >/dev/null 2>&1; then
        actual_md5="$(md5sum "${destination}" | awk '{print $1}')"
    elif command -v md5 >/dev/null 2>&1; then
        actual_md5="$(md5 -q "${destination}")"
    else
        echo "Neither md5sum nor md5 is available; cannot verify ${filename}." >&2
        exit 1
    fi

    if [[ "${actual_md5}" != "${expected_md5}" ]]; then
        echo "Checksum mismatch for ${filename}" >&2
        echo "Expected: ${expected_md5}" >&2
        echo "Actual:   ${actual_md5}" >&2
        exit 1
    fi
    echo "Verified ${filename}"
}

download_file "GPUMD-v3.9.3.zip" "e9c7f3de78f5d60b25098ed659f0bb53"
download_file "03-Demo-MD.zip" "60e3ff7c7b6174df6121d1ede0f06bc4"
download_file "README-v1.md" "786ff0716022153c2f1a7a0b510b16a7"

echo "Extracting GPUMD and the NEP-MB-pol MD demonstrations"
if [[ ! -d "${UPSTREAM_DIR}/GPUMD-v3.9.3" ]]; then
    unzip -q "${DOWNLOAD_DIR}/GPUMD-v3.9.3.zip" -d "${UPSTREAM_DIR}"
    mv "${UPSTREAM_DIR}/brucefan1983-GPUMD-139ffbf" \
       "${UPSTREAM_DIR}/GPUMD-v3.9.3"
else
    echo "Using existing extracted GPUMD-v3.9.3"
fi

if [[ ! -d "${UPSTREAM_DIR}/03-Demo-MD" ]]; then
    unzip -q "${DOWNLOAD_DIR}/03-Demo-MD.zip" -d "${UPSTREAM_DIR}"
else
    echo "Using existing extracted 03-Demo-MD"
fi

cp "${UPSTREAM_DIR}/03-Demo-MD/CMD/Density/nep.txt" \
   "${MODEL_DIR}/nep-mbpol.nep.txt"
cp "${DOWNLOAD_DIR}/README-v1.md" "${UPSTREAM_DIR}/README-v1.md"

if command -v python3 >/dev/null 2>&1; then
    echo "Downloading the pinned GenIce2 source package/wheel"
    python3 -m pip download --no-deps --disable-pip-version-check \
        --dest "${DOWNLOAD_DIR}" "genice2==2.2.13.3"
else
    echo "python3 was not found; skipped the optional GenIce2 package download."
fi

echo
echo "NEP-MB-pol runtime files are ready in:"
echo "  ${SCRIPT_DIR}"
echo
echo "The 1.5 GB training/test corpus was intentionally omitted because it is"
echo "not required to run the published pretrained potential."
