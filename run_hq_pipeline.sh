#!/usr/bin/env bash
# hdripbr 최종 파이프라인 (학습 없음, 사전학습 weight 추론만)
# 채널별 확정 방법:
#   LDR        : DiT360 inference.py (원본, seed0, 2048x1024 -> 4096x2048)
#   HDRI EXR   : ExpandNet (내부 수정으로 진짜 HDR>1) + 3노출 프리뷰(p50/p95/p99)
#   BaseColor  : OmniX            Roughness : OmniX
#   Normal     : OmniX + Marigold (둘 다 저장)
#   Metallic   : Marigold         Depth     : Marigold
#   AO/Height/Displacement : 직접구현(derived, depth+normal+luma)
#
# 사용:
#   ./run_hq_pipeline.sh "11_metal_factory" "A polished steel factory hangar ..."
#   ./run_hq_pipeline.sh "11_metal_factory" --reuse /path/to/panorama.png
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/conda)"
PY="${HDRIPBR_PY:-${CONDA_BASE}/envs/dit360/bin/python}"        # DiT360/ExpandNet/OmniX/derive
PY_MAT="${HDRIPBR_PY_MAT:-${CONDA_BASE}/envs/pbrhq/bin/python}" # Marigold

OUT_NAME="${1:?scene name required}"; shift
OUT_DIR="${HERE}/outputs_v2/dit360/${OUT_NAME}"
mkdir -p "${OUT_DIR}"
PANO="${OUT_DIR}/panorama.png"

# ---- [1/7] LDR (DiT360 원본, seed0) ----
if [ "${1:-}" = "--reuse" ]; then
    cp "${2}" "${PANO}"; echo "[1/7] reuse panorama: ${2}"
else
    PROMPT="This is a panorama. ${1:?prompt required}"
    echo "[1/7] DiT360 generate (inference.py, seed0, 2048x1024 -> 4K)"
    ( cd "${HERE}/DiT360" && ${PY} inference.py "${PROMPT}" "${PANO}" )
fi

# ---- [2/7] ExpandNet HDR (진짜 HDR>1) + 3노출 프리뷰 ----
echo "[2/7] ExpandNet LDR->HDR EXR"
( cd "${HERE}/ExpandNet" && OPENCV_IO_ENABLE_OPENEXR=1 ${PY} expand.py "${PANO}" --use_exr True --out "${OUT_DIR}" )
mv -f "${OUT_DIR}/panorama_prediction.exr" "${OUT_DIR}/hdri.exr"
${PY} "${HERE}/pipeline/stage_exr_preview.py" "${OUT_DIR}/hdri.exr" "${OUT_DIR}"

# ---- [3/7] OmniX -> BaseColor / Roughness / Normal(omnix) ----
echo "[3/7] OmniX -> basecolor/roughness/normal_omnix"
( cd "${HERE}/OmniX" && ${PY} scripts/run_pano_perception.py --panorama "${PANO}" --output_dir "${OUT_DIR}" )
${PY} "${HERE}/pipeline/stage_omnix_post.py" "${OUT_DIR}" 4096 2048

# ---- [4/7] Marigold -> Metallic / Normal(marigold) / Depth ----
echo "[4/7] Marigold -> metallic/normal_marigold/depth"
${PY_MAT} "${HERE}/pipeline/stage_marigold_final.py" "${PANO}" "${OUT_DIR}" --res 2048 --ensemble 8 --steps 4

# ---- [5/7] derived -> AO / Height / Displacement ----
echo "[5/7] derived AO/Height/Displacement"
${PY} "${HERE}/pipeline/stage_pbr_derive.py" "${OUT_DIR}" "${PANO}"

# ---- [6/7] 합본 프리뷰 ----
echo "[6/7] assemble preview"
${PY} "${HERE}/pipeline/stage_assemble.py" "${OUT_DIR}" --panorama "${PANO}"
cp -f "${OUT_DIR}/preview_stitched.png" "${OUT_DIR}/pbr_preview.png"

# ---- [7/7] 정리 ----
echo "[7/7] cleanup"
rm -rf "${OUT_DIR}/compare" "${OUT_DIR}/pbr_depth8.png" "${OUT_DIR}/pbr_normal.png"

echo "Done -> ${OUT_DIR}"
ls -1 "${OUT_DIR}" | grep -vE "panorama_(hires|native)|hunyuan"