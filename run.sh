#!/usr/bin/env bash

# 사용:
#   ./run.sh                                   # inputs/text/*.txt 전체 실행 + 총괄 검증
#   ./run.sh inputs/text/01_sunny_plaza.txt    # 단일 입력
#   ./run.sh inputs/text/01_sunny_plaza.txt --reuse <flat_dir>   # 생성 재사용(검증만)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/conda)"
export CONDA_BASE
PY="${CONDA_BASE}/envs/dit360/bin/python"

if [ $# -ge 1 ]; then
    ${PY} "${HERE}/scripts/run.py" "$@"
else
    for f in "${HERE}"/inputs/text/*.txt; do
        echo "################ $(basename "$f" .txt) ################"
        ${PY} "${HERE}/scripts/run.py" "$f" || echo "  (실패: $f)"
    done
    echo "################ 총괄 검증 ################"
    ${PY} "${HERE}/scripts/validate.py" "${HERE}/outputs"
fi
