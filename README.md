# HDRIPBR — 텍스트 → 360° HDRI + PBR (생성·검증 파이프라인)

기준서(입력 처리 · HDRI 생성 · PBR 채널 · 품질 검증 · 납품) 흐름을 따른다.
**학습 없음** — 공개 사전학습 모델 추론 + 결정적 후처리.

## 구조 (루트 4가지)

```
models/    딥러닝 모델   (DiT360 · ExpandNet · OmniX  + Marigold/DA는 HF 자동)
inputs/    입력          (text/ · image/ · mixed/)
outputs/   산출물        (각 입력 → Result 폴더, 기준서 §2.2 구조)
scripts/   실행 스크립트 (run.py · validate.py · refine_input.py · pbr_extra.py · stage_*.py)
run.sh     엔트리
```

## 설치
`git clone` 후 [install.md](install.md) 명령을 순서대로. (conda env 2개 + 모델 clone/패치)

## 실행

```bash
./run.sh                                 # inputs/text/*.txt 전체 → 생성 + 검증(O/X)
./run.sh inputs/text/01_sunny_plaza.txt  # 단일 입력
python scripts/validate.py outputs       # 검증만 (케이스별 O/X + CSV/JSON)
```

`run.sh` 가 하는 일(기준서 §4): 입력 정제 → HDRI 생성(DiT360→ExpandNet) → PBR 생성(OmniX·Marigold·파생) →
Material ID·엔진 파일 → **검증(O/X)** → 리포트.

## 검증 출력 (예)

```
[01_sunny_plaza]
  ── HDRI
    O  해상도/비율(2:1) · 진짜 HDR(값>1) · seam 1px ΔE(0.0) · pole · 노출 · 대비
    X  seam 32px band (mean≤7,95p≤15)      ← 비블러 정책상 미달 허용(아래 참고)
  ── PBR
    O  필수 8채널 · BaseColor(검정<5%) · Normal blue≥150(Marigold) · Roughness/AO/Height/Metallic
  ── 정합성
    X  구조경계 정합 (OmniX albedo registration 오차)
```
→ 케이스별 O/X 를 `outputs/Validation_Summary.csv`, 각 `outputs/<id>/validation/*.json` 에 저장.

### 검증 현황 (샘플 20장 기준, 정직)
29개 검증항목 중 **26개는 전 20장 통과**. 종합(전항목 통과) O 는 2/20(09·20). 미달은 3항목뿐이며 전부 모델/정책 한계:

| 미달 항목 | 장면수 | 원인 (학습 없이 후처리로는 해결 불가) |
|---|:-:|---|
| HDRI **seam 32px band** | 15 | DiT360/FLUX 가 좌우 wrap-일관 파노라마를 생성 못함. 비블러 정책상 미달 허용(블러/band-offset 은 번짐 유발해 폐기) |
| **구조경계 정합** | 9 | OmniX albedo 의 equirect 타일링 registration 오차 |
| **Normal 뒤집힘** | 5 | Marigold normal 자체 한계(일부 장면 z<0 >2%) |

(1px seam·진짜HDR·BaseColor 검정/흰색·Normal blue·전 채널 stddev/seam·필수8채널·MaterialID 등은 전 장면 통과.)

### 해상도
모든 산출물은 **4096×2048(2:1)** 로 저장하나, 실제 native 생성은 — LDR/DiT360 **2048×1024**, OmniX(basecolor·roughness·normal_omnix·semantic) **1440×720**, Marigold(metallic·normal·depth) 처리 2048급. 즉 4K 는 업샘플이며 PDF 의 **8K(기준품질)가 아닌 4K(정량검수 preview) 레벨**이다.

## 산출물 (`outputs/<id>/`, 기준서 §2.2)

```
input/      input_text.txt, input_meta.json
refined/    refined_prompt.json, image_analysis.json, merged_conditions.json
hdri/final/ hdri_final.exr, hdri_preview.png (+ 3노출)
pbr/final/  pbr_basecolor / normal / roughness / metallic / ao / height / displacement / material_id .png
material/   ue_material_setup.json, ue_texture_import_settings.txt, material_manifest.json
validation/ final_validation_summary.json, pbr_validation.json
log.txt
```

채널별 모델: LDR=DiT360 · HDR=ExpandNet · BaseColor/Roughness=OmniX · Metallic/Depth=Marigold ·
**Normal=Marigold tangent**(납품; OmniX normal 은 참고로 동봉) · AO/Height/Displacement=파생 ·
Semantic=OmniX(하늘/물 마스킹) · MaterialID=재질 클러스터링.

### 품질 보정 (결정적 후처리, 학습 없음 — **원본 비블러** 정책)
- **HDRI/PBR seam** — 좌우 끝의 1px 색 차이만 offset-feather 로 분산(`seamfix.py`). **블러·콘텐츠 혼합 없음(원본 디테일 보존).** 그 결과 1px seam 은 통과하나 **32px band 평균(§5.2/§7.2)은 콘텐츠가 실제로 다른 장면에서 미달**할 수 있음 — 비블러 정책상 의도된 트레이드오프.
- **하늘/구름/물** — OmniX 가 albedo 를 검정으로 둔 영역만 파노라마 콘텐츠(약간 탈채도)로 채움(`stage_semantic_fix.py`). 회색 바닥 등 유효 albedo 는 보존.
- **Normal** — Marigold(tangent-space) 납품. wrap seam 은 벡터 feather+재정규화.
- **HDRI preview** — near-white highlight knee 로 천정 클리핑 방지(`stage_exr_preview.py`).
- **contact_sheet.png** — HDRI+전 PBR 채널을 한 장에 모아 한눈에 비교(`make_contact_sheet.py`).

### 스크립트 구분
- **포워드 파이프라인(이식 가능)**: `run.sh`→`run.py` + `stage_*.py` · `refine_input.py` · `pbr_extra.py` · `seamfix.py` · `validate.py` · `make_contact_sheet.py`. `.archive` 불필요 — **새 서버에서 그대로 동작**.
- **로컬 retrofit 전용**: `apply_fixes.py` · `redo_hdri.py`(+ `models/OmniX/scripts/regen_semantic.py`). 이미 만들어진 outputs/.archive 에 후처리 재적용용. 새 서버에선 `./run.sh` 로 새로 생성.
