# hdripbr 설치 가이드 (깨끗한 서버 기준)

아무것도 없는 리눅스 서버에서 이 파이프라인을 처음부터 실행하기 위한 단계별 안내.

---

## 0. 코드는 어떻게 준비하나? (가장 헷갈리는 부분)

코드는 **3종류**로 나뉩니다.

| 종류 | 무엇 | 어떻게 |
|---|---|---|
| **① 본 리포 (hdripbr)** | 우리가 만든 코드: `pipeline/*.py`, `run_*.sh`, `requirements.txt`, `patches/` | **파일로 제공** (이 폴더 전체 = git clone 또는 zip 전달) |
| **② 외부 3개 repo** | DiT360 / ExpandNet / OmniX | **git clone 한 뒤 패치 적용** — `patches/` 안의 파일로 덮어쓰면 끝 (`setup.sh`가 자동) |
| **③ 모델 가중치** | FLUX, DiT360 LoRA, OmniX, Marigold | **코드 아님. 첫 실행 시 Hugging Face에서 자동 다운로드** (FLUX만 로그인 필요) |

> 정리: **외부 repo는 "clone + 수정(패치)"** 입니다. 수정은 직접 손댈 필요 없이 `patches/`의 3개 파일을 복사하면 됩니다.
> 우리 파이프라인 코드(`pipeline/`, `run_*.sh`)는 이 리포에 들어있으니 **파일째로 받으면** 됩니다.
> 패치 내용은 [§7](#7-패치-상세-3건)에 정리.

수정되는 외부 파일은 **딱 3개**:
- `DiT360/inference.py` (전체 교체 — argv 입력 + seed0 + 4K 업샘플)
- `ExpandNet/expand.py` (`expand_to_hdr()` 추가 → 진짜 HDR)
- `OmniX/src/systems/omnix.py` (큰 입력 shape 1줄 수정)

---

## 1. 하드웨어 / 사전 요구사항

- **GPU**: NVIDIA, VRAM **≥ 24GB** (검증: RTX 5090 32GB, Blackwell sm_120)
- **CUDA 드라이버**: 12.8 호환 (`nvidia-smi` 동작)
- **디스크**: 여유 **≥ 80GB** (가중치 FLUX≈24GB + OmniX≈10GB + Marigold≈수GB + 캐시)
- **소프트웨어**: `git`, `wget`, Miniconda/Anaconda, 인터넷(github + huggingface.co 접근)
- **Hugging Face 계정** (FLUX.1-dev 가 gated 모델 → 라이선스 동의 + 토큰 필요)

---

## 2. 본 리포(hdripbr) 받기

```bash
cd /path/to/work
# (A) git 으로 받는 경우
git clone <hdripbr_repo_url> hdripbr
# (B) zip 으로 받은 경우: 압축 해제
#   unzip hdripbr.zip -d hdripbr
cd hdripbr
```

이 안에 이미 들어있는 것: `pipeline/`, `run_hq_pipeline.sh`, `run_all_scenes.sh`,
`run_compare.sh`, `requirements.txt`, `patches/`, `setup.sh`.

---

## 3. Conda 환경 2개 생성 + 패키지 설치

> **왜 2개?** DiT360 LoRA·OmniX 는 `diffusers==0.32.2`, Marigold 재질은 `diffusers>=0.38` 을
> 요구해 한 환경에 공존 불가. 그래서 **`dit360`(생성·HDR·OmniX·derived)** 와 **`pbrhq`(Marigold)** 로 분리.

PyTorch 는 **반드시 cu128 인덱스에서 먼저** 설치 (Blackwell sm_120 호환):

```bash
TORCH="pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1"

# (1) dit360 env — 생성/HDR/OmniX/derived
conda create -n dit360 python=3.12 -y
conda activate dit360
eval "$TORCH"
pip install -r requirements.txt
conda deactivate

# (2) pbrhq env — Marigold (metallic / normal / depth)
conda create -n pbrhq python=3.12 -y
conda activate pbrhq
eval "$TORCH"
pip install "diffusers>=0.38" transformers==4.49.0 accelerate safetensors huggingface_hub numpy opencv-python pillow
conda deactivate
```

> ⚠️ **OmniX/requirements.txt 는 설치하지 마세요.** 거기엔 `torch==2.4.1`(Blackwell 미지원)과
> open3d/trimesh/pymeshlab/realesrgan(3D 내보내기용, perception엔 불필요)이 들어있습니다.
> perception 경로는 위 `dit360` env 만으로 충분합니다. (equilib 는 OmniX 가 `external/`에 자체 포함)

---

## 4. 외부 repo clone + 패치 (setup.sh)

```bash
bash setup.sh
```

이게 하는 일:
1. `DiT360`, `ExpandNet`(=hdr-expandnet), `OmniX` 를 `git clone`
2. `patches/` 의 3개 파일을 각 repo 위에 복사(전체 교체)

ExpandNet 가중치(`ExpandNet/weights.pth`)는 clone에 포함됩니다.
비교 도구(DeepBump)까지 받으려면: `WITH_COMPARE=1 bash setup.sh`

> 만약 외부 repo 최신 버전이 바뀌어 패치 파일과 어긋나면, [§7](#7-패치-상세-3건)의 변경 3건을 수동 적용하세요.

---

## 5. Hugging Face 로그인 (FLUX gated)

FLUX.1-dev 는 라이선스 동의가 필요한 gated 모델입니다.

1. 브라우저에서 https://huggingface.co/black-forest-labs/FLUX.1-dev 접속 → **Agree/Access** 동의
2. 토큰 발급(https://huggingface.co/settings/tokens) 후:
```bash
conda activate dit360
pip install -U "huggingface_hub[cli]"
huggingface-cli login        # 토큰 입력
```
(DiT360 LoRA·OmniX·Marigold 는 public 이라 별도 동의 불필요.)

경로 일치 확인: `run_hq_pipeline.sh` 상단의
`PY=/opt/conda/envs/dit360/bin/python`, `PY_MAT=/opt/conda/envs/pbrhq/bin/python`
가 실제 conda 경로와 같은지 확인(다르면 수정).

---

## 6. 실행

```bash
# 단일 장면 (text → 전체 산출물). 첫 실행은 가중치 다운로드로 오래 걸림.
./run_hq_pipeline.sh "demo" "A polished steel factory hangar with chrome pipes, silver machinery, bright lights."

# 기존 파노라마 재사용(생성 생략)
./run_hq_pipeline.sh "demo" --reuse /path/to/panorama.png

# 예시 11개 일괄
./run_all_scenes.sh all
```

결과: `outputs_v2/dit360/<name>/` (hdri.exr + 3노출 프리뷰 + pbr_* 채널 + pbr_preview.png).
산출물 목록은 README.md 참고.

---

## 7. 패치 상세 (3건)

upstream이 바뀌어 자동 패치가 안 될 때 수동으로 적용할 변경:

**(1) DiT360/inference.py** — 전체를 `patches/DiT360/inference.py`로 교체.
핵심: 프롬프트/저장경로를 `sys.argv`로 받고, `width=2048,height=1024,seed=0`로 생성 후 `4096×2048` LANCZOS 업샘플.

**(2) ExpandNet/expand.py** — `expand_to_hdr()` 함수 추가 + 호출 1줄.
원본은 최종층 `Sigmoid`로 출력이 [0,1]에 갇혀 진짜 HDR이 아님. sigmoid는 그대로 두고 그 **출력 이후**
입력 LDR의 포화/고휘도 영역을 1.0 위로 지수 확장:
```python
# create_images() 안, prediction = map_range(...) 바로 다음 줄에 추가:
prediction = expand_to_hdr(prediction, ldr_input)
```
(`expand_to_hdr` 정의는 patches 파일 참고.)

**(3) OmniX/src/systems/omnix.py** — `_perceive_panoramic_X`의 입력 리사이즈 블록에 1줄 추가:
```python
if height > max_height or width > max_width:
    panorama = panorama.resize((max_width, max_height), Image.LANCZOS)
    height, width = max_height, max_width        # ← 이 줄 추가 (큰 입력 shape mismatch 수정)
```

---

## 8. 트러블슈팅

- **`torch.cuda.is_available()` False / sm_120 경고**: torch를 반드시 cu128 인덱스로 설치했는지 확인.
- **FLUX 다운로드 401/403**: HF 로그인 + FLUX.1-dev 라이선스 동의 확인.
- **OOM**: VRAM<24GB면 어려움. `run_hq_pipeline.sh`의 DiT360는 `enable_model_cpu_offload`로 절감하나 한계 있음.
- **OmniX import 에러**: `dit360` env에서 실행하는지, `OmniX/external/equilib` 가 있는지 확인.
- **EXR 안 열림**: `OPENCV_IO_ENABLE_OPENEXR=1` 환경변수(스크립트가 자동 설정). opencv-python 설치 확인.
- **conda 경로 다름**: `run_*.sh` 상단 `PY`/`PY_MAT` 를 `which python`(각 env activate 후) 값으로 수정.
