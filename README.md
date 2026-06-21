# hdripbr — 텍스트 → 360° HDRI + PBR

## 설치 (깨끗한 서버, 한 번)

```bash
git clone https://github.com/MinChoi0129/hdripbr.git && cd hdripbr
bash install.sh                              # conda env 2개 + 외부 repo clone + 패치
conda activate dit360 && huggingface-cli login   # FLUX.1-dev 라이선스 동의 후 토큰 입력
```

## 실행 (넣는 것 → 나오는 것)

```bash
# 텍스트 한 줄 → 360 HDRI + PBR 전 채널
./run_hq_pipeline.sh metal "A polished steel factory hangar with chrome pipes, silver machinery, bright lights."

# 이미 있는 파노라마(.png) → 나머지 전부 (생성 생략)
./run_hq_pipeline.sh metal --reuse /path/to/panorama.png

# 예시 프롬프트 11개 한꺼번에
./run_all_scenes.sh all

# 채널별 방법 비교본 (OmniX/Marigold/직접구현/DeepBump/DA-V2)
./run_compare.sh metal
```

## 나오는 것 → `outputs_v2/dit360/<이름>/`

```
hdri.exr                        360 HDR (값>1, IBL)
hdri_preview_p50/p95/p99.png    HDR 3노출 미리보기
pbr_basecolor.png               BaseColor
pbr_roughness.png               Roughness
pbr_normal_omnix.png            Normal (OmniX)
pbr_normal_marigold.png         Normal (Marigold)
pbr_metallic.png                Metallic
pbr_depth.png                   Depth
pbr_ao.png                      AO
pbr_height.png                  Height (16bit)
pbr_displacement.png            Displacement (16bit)
pbr_preview.png                 전 채널 합본
```

상세 설치/구조: [INSTALL.md](INSTALL.md)
