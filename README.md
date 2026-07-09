# HDRIPBR

## 설치

1. 레포 clone

```bash
git clone https://github.com/MinChoi0129/hdripbr.git
cd hdripbr
```

2. 외부 모델 clone 과 패치 (리포 폴더 안에서 실행)

```bash
git clone https://github.com/Insta360-Research-Team/DiT360.git models/DiT360
git clone https://github.com/dmarnerides/hdr-expandnet.git models/ExpandNet
git clone https://github.com/HKU-MMLab/OmniX.git models/OmniX
cp patches/DiT360/inference.py models/DiT360/inference.py
cp patches/ExpandNet/expand.py models/ExpandNet/expand.py
cp patches/OmniX/src/systems/omnix.py models/OmniX/src/systems/omnix.py
```

3. conda 환경 dit360 (생성, HDR, OmniX, 파생 채널)

```bash
conda create -n dit360 python=3.12 -y
conda activate dit360
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1
pip install -r requirements.txt
conda deactivate
```

4. conda 환경 pbrhq (Marigold)

```bash
conda create -n pbrhq python=3.12 -y
conda activate pbrhq
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1
pip install "diffusers>=0.38" transformers==4.49.0 accelerate \
  safetensors huggingface_hub numpy opencv-python pillow
conda deactivate
```

5. Hugging Face 로그인 (FLUX.1-dev 라이선스 동의 후)

먼저 https://huggingface.co/black-forest-labs/FLUX.1-dev 에서 동의:

```bash
conda activate dit360
huggingface-cli login
```

## 입력

inputs/text 폴더에 영어 한 줄 텍스트 파일을 둔다. 파일명이 출력
폴더 이름이 된다.

예시 (inputs/text/09_mars_colony.txt):

```
A Mars colony at golden hour, domes on rust-red terrain.
```

## 실행

```bash
./run.sh                                 # inputs/text 의 모든 입력 처리
./run.sh inputs/text/09_mars_colony.txt  # 입력 하나만 처리
python scripts/validate.py outputs       # 검증만 다시 실행
```

## 출력 산출물

입력 하나당 outputs 아래에 폴더 하나가 생긴다.

- panorama.png: 360 파노라마 LDR 이미지
- hdri/final/: hdri_final.exr (HDR) 와 hdri_preview.png (미리보기, 노출 3종)
- pbr/final/: basecolor(원본), basecolor_with_semantic(하늘/물 보정),
  normal, roughness, metallic, ao, height, displacement, depth,
  material_id, semantic 의 png
- material/: ue_material_setup.json, 텍스처 import 설정, manifest
- validation/: 검증 결과 json
- contact_sheet.png: HDRI 와 PBR 을 한 장에 모은 비교 이미지
- outputs/Validation_Summary.csv: 전체 입력 검증 요약

## 출력 경로

입력이름 은 inputs/text 의 파일명 (확장자 제외) 이다.

- outputs/입력이름/panorama.png
- outputs/입력이름/hdri/final/hdri_final.exr
- outputs/입력이름/hdri/final/hdri_preview.png
- outputs/입력이름/pbr/final/pbr_basecolor.png
- outputs/입력이름/pbr/final/pbr_basecolor_with_semantic.png
- outputs/입력이름/pbr/final/pbr_normal.png
- outputs/입력이름/pbr/final/pbr_roughness.png
- outputs/입력이름/pbr/final/pbr_metallic.png
- outputs/입력이름/pbr/final/pbr_ao.png
- outputs/입력이름/pbr/final/pbr_height.png
- outputs/입력이름/pbr/final/pbr_displacement.png
- outputs/입력이름/pbr/final/pbr_depth.png
- outputs/입력이름/pbr/final/pbr_material_id.png
- outputs/입력이름/material/ue_material_setup.json
- outputs/입력이름/validation/final_validation_summary.json
- outputs/입력이름/contact_sheet.png
- outputs/Validation_Summary.csv

## 사용 모델

- FLUX.1-dev (Black Forest Labs, 2024): 텍스트에서 이미지 생성 베이스
  https://huggingface.co/black-forest-labs/FLUX.1-dev
- DiT360 (CVPR 2026): FLUX 로 360 파노라마 생성
  https://fenghora.github.io/DiT360-Page/
- ExpandNet (Eurographics 2018, Computer Graphics Forum): HDRI 확장
  https://doi.org/10.1111/cgf.13340
- OmniX (arXiv 2025): 파노라마에서 PBR 재질/노멀/시맨틱 인식
  https://yukun-huang.github.io/OmniX/
- Marigold (CVPR 2024): 확산기반 노멀/메탈릭/뎁스 추정
  https://openaccess.thecvf.com/content/CVPR2024/html/Ke_Repurposing_Diffusion-Based_Image_Generators_for_Monocular_Depth_Estimation_CVPR_2024_paper.html
