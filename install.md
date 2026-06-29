# 설치

아래 명령을 위에서부터 순서대로 붙여넣으세요. (리포 폴더 `hdripbr/` 안에서 실행)

## 1. 외부 repo clone + 패치 (models/ 안으로)

```bash
git clone https://github.com/Insta360-Research-Team/DiT360.git models/DiT360
git clone https://github.com/dmarnerides/hdr-expandnet.git      models/ExpandNet
git clone https://github.com/HKU-MMLab/OmniX.git                models/OmniX
cp patches/DiT360/inference.py            models/DiT360/inference.py
cp patches/ExpandNet/expand.py            models/ExpandNet/expand.py
cp patches/OmniX/src/systems/omnix.py     models/OmniX/src/systems/omnix.py
cp patches/OmniX/scripts/regen_semantic.py models/OmniX/scripts/regen_semantic.py   # semantic 재생성 유틸(선택)
```

> 코드는 `models/DiT360`, `models/ExpandNet`, `models/OmniX` 경로를 기대한다. 반드시 위처럼 `models/` 안으로 clone.

## 2. conda 환경 — dit360 (생성 / HDR / OmniX / derived)

```bash
conda create -n dit360 python=3.12 -y
conda activate dit360
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
pip install -r requirements.txt
conda deactivate
```

## 3. conda 환경 — pbrhq (Marigold)

```bash
conda create -n pbrhq python=3.12 -y
conda activate pbrhq
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
pip install "diffusers>=0.38" transformers==4.49.0 accelerate safetensors huggingface_hub numpy opencv-python pillow
conda deactivate
```

## 4. Hugging Face 로그인 (FLUX.1-dev gated)

먼저 https://huggingface.co/black-forest-labs/FLUX.1-dev 에서 라이선스 동의 후:

```bash
conda activate dit360
huggingface-cli login
```

설치 끝. 실행은 README.md 참고.
