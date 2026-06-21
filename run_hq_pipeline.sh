#!/usr/bin/env bash
# 단일 문자열 모드: 프롬프트 1개 -> 360 HDRI + PBR 전 채널.
# 사용: ./run_hq_pipeline.sh <이름> "<프롬프트>"
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/conda)"
PY="${HDRIPBR_PY:-${CONDA_BASE}/envs/dit360/bin/python}"
PY_MAT="${HDRIPBR_PY_MAT:-${CONDA_BASE}/envs/pbrhq/bin/python}"

NAME="${1:?사용: ./run_hq_pipeline.sh <이름> \"<프롬프트>\"}"
TEXT="${2:?프롬프트 문자열이 필요합니다}"
OUT_DIR="${HERE}/outputs_v2/dit360/${NAME}"
mkdir -p "${OUT_DIR}"
PANO="${OUT_DIR}/panorama.png"

echo "[1/6] DiT360 생성"
( cd "${HERE}/DiT360" && ${PY} inference.py "This is a panorama. ${TEXT}" "${PANO}" )

echo "[2/6] ExpandNet HDR (EXR + 3노출)"
( cd "${HERE}/ExpandNet" && OPENCV_IO_ENABLE_OPENEXR=1 ${PY} expand.py "${PANO}" --use_exr True --out "${OUT_DIR}" )
mv -f "${OUT_DIR}/panorama_prediction.exr" "${OUT_DIR}/hdri.exr"
${PY} "${HERE}/pipeline/stage_exr_preview.py" "${OUT_DIR}/hdri.exr" "${OUT_DIR}"

echo "[3/6] OmniX (BaseColor / Roughness / Normal)"
( cd "${HERE}/OmniX" && ${PY} scripts/run_pano_perception.py --panorama "${PANO}" --output_dir "${OUT_DIR}" )
${PY} "${HERE}/pipeline/stage_omnix_post.py" "${OUT_DIR}" 4096 2048

echo "[4/6] Marigold (Metallic / Normal / Depth)"
${PY_MAT} "${HERE}/pipeline/stage_marigold_final.py" "${PANO}" "${OUT_DIR}" --res 2048 --ensemble 8 --steps 4

echo "[5/6] AO / Height / Displacement"
${PY} "${HERE}/pipeline/stage_pbr_derive.py" "${OUT_DIR}" "${PANO}"

echo "[6/6] 합본 프리뷰"
${PY} "${HERE}/pipeline/stage_assemble.py" "${OUT_DIR}" --panorama "${PANO}"
cp -f "${OUT_DIR}/preview_stitched.png" "${OUT_DIR}/pbr_preview.png"

echo "완료 -> ${OUT_DIR}"
