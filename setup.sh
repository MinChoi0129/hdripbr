#!/usr/bin/env bash
# hdripbr 외부 구성요소 설치: 3개 repo clone + 패치 적용.
# (conda env / pip 설치는 INSTALL.md 참고. 이 스크립트는 '코드' 배치만 담당)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "${HERE}"

clone() { [ -d "$2" ] && echo "  exists: $2" || git clone "$1" "$2"; }

echo "[1/2] 외부 repo clone"
clone https://github.com/Insta360-Research-Team/DiT360.git        DiT360
clone https://github.com/dmarnerides/hdr-expandnet.git            ExpandNet   # weights.pth 포함
clone https://github.com/HKU-MMLab/OmniX.git                      OmniX

echo "[2/2] 패치 적용 (patches/ -> 각 repo, 전체파일 교체)"
cp patches/DiT360/inference.py            DiT360/inference.py
cp patches/ExpandNet/expand.py            ExpandNet/expand.py
cp patches/OmniX/src/systems/omnix.py     OmniX/src/systems/omnix.py
echo "  patched: DiT360/inference.py, ExpandNet/expand.py, OmniX/src/systems/omnix.py"
