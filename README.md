# hdripbr — 텍스트 → 360° HDRI + PBR

## 설치 (한 번)

```bash
git clone https://github.com/MinChoi0129/hdripbr.git && cd hdripbr
bash install.sh
conda activate dit360 && huggingface-cli login   # FLUX.1-dev 동의 후 토큰
```

## 모드 1 — 문자열 1개

```bash
./run_hq_pipeline.sh metal "A polished steel factory hangar with chrome pipes, silver machinery."
```

## 모드 2 — 문자열 여러개

`run_all_scenes.sh` 안의 `items=( "이름|프롬프트" ... )` 목록을 편집한 뒤:

```bash
./run_all_scenes.sh
```

## 나오는 것 → `outputs_v2/dit360/<이름>/`

```
hdri.exr                        360 HDR (값>1)
hdri_preview_p50/p95/p99.png    HDR 3노출
pbr_basecolor.png               BaseColor
pbr_roughness.png               Roughness
pbr_normal_omnix.png            Normal (OmniX)
pbr_normal_marigold.png         Normal (Marigold)
pbr_metallic.png                Metallic
pbr_depth.png                   Depth
pbr_ao.png                      AO
pbr_height.png                  Height
pbr_displacement.png            Displacement
pbr_preview.png                 전 채널 합본
```
