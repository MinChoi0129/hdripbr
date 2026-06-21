#!/usr/bin/env bash
# hdripbr 외부 구성요소 설치: 3개 repo clone + 패치 적용.
# (conda env / pip 설치는 INSTALL.md 참고. 이 스크립트는 '코드' 배치만 담당)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "${HERE}"

clone() { [ -d "$2" ] && echo "  exists: $2" || git clone "$1" "$2"; }

echo "[1/3] 외부 repo clone"
clone https://github.com/Insta360-Research-Team/DiT360.git        DiT360
clone https://github.com/dmarnerides/hdr-expandnet.git            ExpandNet   # weights.pth 포함
clone https://github.com/HKU-MMLab/OmniX.git                      OmniX

echo "[2/3] 패치 적용 (patches/ -> 각 repo, 전체파일 교체)"
cp patches/DiT360/inference.py            DiT360/inference.py
cp patches/ExpandNet/expand.py            ExpandNet/expand.py
cp patches/OmniX/src/systems/omnix.py     OmniX/src/systems/omnix.py
echo "  patched: DiT360/inference.py, ExpandNet/expand.py, OmniX/src/systems/omnix.py"

echo "[3/3] (선택) 비교 도구 — 메인 파이프라인엔 불필요"
if [ "${WITH_COMPARE:-0}" = "1" ]; then
  clone https://github.com/HugoTini/DeepBump.git third_party/DeepBump
fi

echo "OK. 다음: conda env 2개 생성 + pip 설치 (INSTALL.md Step 2), 그리고 huggingface-cli login (FLUX gated)."
