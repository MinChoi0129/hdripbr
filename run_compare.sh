#!/usr/bin/env bash
# 채널별 방법(OmniX / Marigold / 직접구현 / DeepBump / DA-V2) 전부 실행해 비교 저장.
# 사용: ./run_compare.sh 01_cyberpunk [legacy_panorama.png]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/conda)"
PY="${HDRIPBR_PY:-${CONDA_BASE}/envs/dit360/bin/python}"
PY_MAT="${HDRIPBR_PY_MAT:-${CONDA_BASE}/envs/pbrhq/bin/python}"
DB_DIR="${HERE}/third_party/DeepBump"

S="${1:?scene name}"
PANO="${2:-${HERE}/legacy/old_outputs/dit360/${S}/${S}.png}"
OMNIX_DIR="${HERE}/legacy/old_outputs/dit360/${S}"
CDIR="${HERE}/outputs_v2/dit360/${S}/compare"
mkdir -p "${CDIR}"
cp "${PANO}" "${CDIR}/_panorama.png"

echo "[compare:${S}] Marigold (albedo/roughness/metallic/normal/depth)"
${PY_MAT} "${HERE}/pipeline/stage_marigold_compare.py" "${PANO}" "${CDIR}" --res 2048 --ensemble 8 --steps 4

echo "[compare:${S}] OmniX(legacy) + DA-V2 + DeepBump + derived"
${PY} "${HERE}/pipeline/stage_compare_rest.py" "${PANO}" "${CDIR}" "${OMNIX_DIR}" "${DB_DIR}"

echo "[compare:${S}] channel sheets"
${PY} "${HERE}/pipeline/stage_compare_sheets.py" "${CDIR}"

echo "Done -> ${CDIR}"
ls -1 "${CDIR}"