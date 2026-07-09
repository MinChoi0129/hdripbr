"""equirect 좌우 seam 정합 (결정적, 비블러 — 원본 디테일 보존).

DiT360 파노라마는 좌끝열≠우끝열 불연속(seam)이 있다. 콘텐츠를 흐리게(블러) 만들지 않고
좌/우 끝의 1px 색 차이(offset)만 좁은 밴드에 선형 분산해 경계 계단을 없앤다(offset-feather).
  - seam_fix        : 1px offset-feather. HDRI·PBR scalar 채널 공통.
  - fix_normal_seam : normal 벡터 feather 후 재정규화.
주의: 32px band 평균차는 콘텐츠가 실제 다른 경우라 비블러로는 완전히 못 맞춘다(미달 허용).
      band-offset 방식은 seam 부근에 번진 색띠를 만들어 시각 품질을 해쳐 사용하지 않는다.
"""
import os
import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2

PBR_CHANNELS = ["pbr_basecolor.png", "pbr_basecolor_with_semantic.png", "pbr_roughness.png",
                "pbr_metallic.png", "pbr_ao.png", "pbr_height.png", "pbr_displacement.png"]


def _feather(a, Bn):
    gap = a[:, 0] - a[:, -1]                              # 좌우 끝열 1px 차이
    for i in range(Bn):
        w = 0.5 * (1.0 - i / Bn)
        a[:, -1 - i] += gap * w
        a[:, i] -= gap * w
    return a


def _cast(a, dt):
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        a = np.clip(a, info.min, info.max)
    return a.astype(dt)


def seam_fix(arr, band_frac=1.0 / 48):
    """좌우 1px offset-feather(비블러). HDRI·PBR scalar 채널 공통."""
    H, W = arr.shape[:2]
    return _cast(_feather(arr.astype(np.float32), max(4, int(W * band_frac))), arr.dtype)


def fix_image_file(path, band_frac=1.0 / 48):
    from PIL import Image
    im = Image.open(path)
    Image.fromarray(seam_fix(np.array(im), band_frac)).save(path)
    return path


def fix_exr(path):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR).astype(np.float32)
    out = np.maximum(seam_fix(img), 0.0)               # 1px feather, 음수 radiance 방지
    cv2.imwrite(path, out)
    return path


def fix_normal_seam(path, band_frac=1.0 / 48):
    """normal map 의 좌우 wrap 불연속 제거: 벡터 성분 offset-feather 후 재정규화(블러 없음)."""
    from PIL import Image
    a = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255 * 2 - 1
    W = a.shape[1]
    a = _feather(a, max(4, int(W * band_frac)))          # _feather 는 정수 밴드폭(px)을 받음
    a /= (np.linalg.norm(a, axis=2, keepdims=True) + 1e-8)
    Image.fromarray(((a * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)).save(path)
    return path


def fix_pbr_dir(pf, band_frac=1.0 / 48):
    done = []
    for ch in PBR_CHANNELS:
        p = os.path.join(pf, ch)
        if os.path.exists(p):
            fix_image_file(p, band_frac); done.append(ch)
    for nch in ("pbr_normal.png", "pbr_normal_marigold.png", "pbr_normal_omnix.png"):
        p = os.path.join(pf, nch)
        if os.path.exists(p):
            fix_normal_seam(p, band_frac); done.append(nch)
    return done


if __name__ == "__main__":
    import sys
    a = sys.argv[1]
    (fix_exr if a.lower().endswith(".exr") else
     (fix_pbr_dir if os.path.isdir(a) else fix_image_file))(a)
    print("seam fixed:", a)
