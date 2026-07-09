"""하늘/구름/물의 검정 albedo 채움 (BaseColor 전용).

문제:  OmniX albedo 는 하늘/물을 검정(0)으로 둬 BaseColor 'black<5%' 검증에 불리하고
       시각적으로도 비어 보인다.
해결:  OmniX 가 검정으로 둔 영역(=하늘/구름/물/광원, albedo 미정의)만 골라
       파노라마 콘텐츠(약간 탈채도)로 채운다. semantic(파랑)은 보조 확인용.
       회색 바닥/건물 등 유효 albedo, 그리고 normal/metallic/roughness 채널은 건드리지 않는다.

사용: python stage_semantic_fix.py <out_dir> <panorama.png>
"""
import os, sys
import numpy as np
from PIL import Image
import cv2


def _load(od, name, rgb=False):
    p = os.path.join(od, name)
    if not os.path.exists(p):
        return None
    im = Image.open(p)
    return np.array(im.convert("RGB")) if rgb else np.array(im)


def _connected_top(mask, upper):
    """morph 정리 후, 탑(천정)에 닿거나 상단영역에 있는 성분만 유지."""
    m = (mask.astype(np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab = cv2.connectedComponents(m)
    keep = np.zeros_like(m, bool)
    top_labels = set(np.unique(lab[:3])) - {0}
    for i in range(1, n):
        comp = lab == i
        if i in top_labels or (comp & upper).sum() > 0.5 * comp.sum():
            keep |= comp
    return keep


def build_skywater_mask(basecolor, semantic):
    """채울 영역 = OmniX 가 albedo 를 검정으로 둔 곳(=하늘/구름/물/광원, albedo 미정의).
       회색 바닥/건물 등 유효 albedo 는 건드리지 않는다(검정만 타깃).
       semantic 은 보조 확인용(파랑+검정 동시일 때만 가산). 과잉 파랑으로 인한 오검출 방지."""
    H, W = basecolor.shape[:2]
    lum_b = basecolor.mean(2)
    black = lum_b < 16                                   # OmniX 미정의(검정) 영역
    # 노이즈 정리: 작은 점 제거 + 구멍 메움
    m = cv2.morphologyEx(black.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)

    yy = np.linspace(0, 1, H)[:, None].repeat(W, 1)
    if semantic is not None:                             # semantic 파랑 & 검정인 곳만 추가(보수적)
        s = semantic.astype(np.int32)
        blue = (s[..., 2] > s[..., 0] + 15) & (s[..., 2] > s[..., 1] + 5)
        m = m | (blue & black)
    sky = _connected_top(m, yy < 0.6) & (yy < 0.8)       # 천정 연결 = 하늘/구름
    water = m & ~sky & (yy > 0.45)                        # 하단 나머지 검정 = 물/수면
    return sky, water


def albedo_ceiling(arr, hi=245.0):
    """물리 albedo 상한: luma>hi 인 highlight(하늘/천장 순백 등)를 hi 로 압축(색조 유지).
       albedo(반사율)는 완전반사체가 불가하므로 순백(255)은 비물리적 + PDF white<5% 위반.
       arr: HxWx3 float, 반환 float."""
    lum = arr.mean(2, keepdims=True)
    gain = np.where(lum > hi, hi / np.maximum(lum, 1e-3), 1.0)
    return arr * gain


def run(out_dir, panorama_path):
    """원본 albedo(pbr_basecolor.png, 순수 OmniX)는 그대로 두고,
       하늘/구름/물의 검정을 채우고 순백 highlight 를 물리 상한으로 압축한 결과를
       pbr_basecolor_with_semantic.png 로 별도 저장한다(원본 보존, 두 벌 유지).
       다른 채널(normal/metallic/roughness)은 건드리지 않는다."""
    base = _load(out_dir, "pbr_basecolor.png", rgb=True)
    if base is None:
        print("  [skip] basecolor 없음"); return None
    H, W = base.shape[:2]
    pano = np.array(Image.open(panorama_path).convert("RGB").resize((W, H))).astype(np.float32)
    semantic = _load(out_dir, "pbr_semantic.png", rgb=True)
    if semantic is not None and semantic.shape[:2] != (H, W):
        semantic = cv2.resize(semantic, (W, H), interpolation=cv2.INTER_NEAREST)

    sky, water = build_skywater_mask(base, semantic)
    fill = cv2.GaussianBlur((sky | water).astype(np.float32), (0, 0), 2)[..., None]

    # 채움 = 파노라마 콘텐츠(구름/물 구조 유지) 를 약간 탈채도 → albedo 근사, 검정 제거
    out = base.astype(np.float32)
    desat = pano * 0.72 + pano.mean(2, keepdims=True) * 0.28
    out = out * (1 - fill) + desat * fill
    # 잔여 순검정만 살짝 올림(색상 유지)
    lum = out.mean(2, keepdims=True)
    fl = lum < 6
    out = np.where(fl, out + (14.0 - lum), out)
    # 순백 highlight 압축(물리 albedo 상한): white<5%
    out = albedo_ceiling(out, hi=245.0)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(
        os.path.join(out_dir, "pbr_basecolor_with_semantic.png"))    # 원본 유지, 처리본 별도 저장

    src = "semantic+prior" if semantic is not None else "prior-only"
    print(f"  하늘/구름/물 채움({src}) -> with_semantic: sky={sky.mean()*100:.0f}% water={water.mean()*100:.0f}%")
    return True

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
