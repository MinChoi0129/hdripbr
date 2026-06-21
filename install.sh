#!/usr/bin/env bash
# 한 방 설치: conda env 2개 생성 + torch(cu128) + 패키지 + 외부 repo clone/패치.
# 사용: bash install.sh        (FLUX 로그인은 별도: huggingface-cli login)
#       HF_TOKEN=xxx bash install.sh  (FLUX 토큰까지 자동)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "${HERE}"

command -v conda >/dev/null || { echo "ERROR: conda(miniconda/anaconda) 필요"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
TORCH="--index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1"

echo "[1/3] dit360 env (생성/HDR/OmniX/derived)"
conda create -n dit360 python=3.12 -y
conda run -n dit360 pip install ${TORCH}
conda run -n dit360 pip install -r requirements.txt

echo "[2/3] pbrhq env (Marigold)"
conda create -n pbrhq python=3.12 -y
conda run -n pbrhq pip install ${TORCH}
conda run -n pbrhq pip install "diffusers>=0.38" transformers==4.49.0 accelerate safetensors huggingface_hub numpy opencv-python pillow

echo "[3/3] 외부 repo clone + 패치"
bash setup.sh

if [ -n "${HF_TOKEN:-}" ]; then
  conda run -n dit360 huggingface-cli login --token "${HF_TOKEN}"
  echo "HF 로그인 완료"
else
  echo ">> FLUX.1-dev 는 gated. 실행 전: conda activate dit360 && huggingface-cli login"
fi

echo "DONE. 예) ./run_hq_pipeline.sh metal \"A polished steel factory hangar ...\""
