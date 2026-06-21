"""직접구현(derived) 채널: AO / Height / Displacement.
입력: pbr_depth.png(Marigold), pbr_normal_marigold.png(경계용), panorama.png(변위 디테일).
전부 결정적(학습 없음).
사용: python stage_pbr_derive.py <out_dir> <panorama.png>
"""
import os, sys
import numpy as np
from PIL import Image
import cv2


def load_gray16(p):
    a = np.array(Image.open(p))
    return a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0)


def norm01(x, lo=2, hi=98):
    a, b = np.percentile(x, [lo, hi]); return np.clip((x - a) / max(b - a, 1e-6), 0, 1)


def main():
    od = sys.argv[1]
    panorama = sys.argv[2]
    depth = load_gray16(os.path.join(od, "pbr_depth.png"))            # near=bright
    nrm_path = os.path.join(od, "pbr_normal_marigold.png")
    if not os.path.exists(nrm_path):
        nrm_path = os.path.join(od, "pbr_normal_omnix.png")
    normal = np.array(Image.open(nrm_path).convert("RGB")).astype(np.float32) / 255
    if normal.shape[:2] != depth.shape[:2]:
        normal = cv2.resize(normal, (depth.shape[1], depth.shape[0]))

    # AO = depth 다중스케일 cavity + normal 경계
    occ = np.zeros_like(depth)
    for sg in (3, 9, 27, 64):
        occ += np.clip(cv2.GaussianBlur(depth, (0, 0), sg) - depth, 0, None)
    occ = norm01(occ, 50, 99)
    n = normal * 2 - 1
    gx = np.abs(cv2.Sobel(n, cv2.CV_32F, 1, 0, ksize=3)).sum(-1)
    gy = np.abs(cv2.Sobel(n, cv2.CV_32F, 0, 1, ksize=3)).sum(-1)
    edge = norm01(gx + gy, 50, 99)
    ao = np.clip(1.0 - (0.85 * occ + 0.35 * edge), 0, 1)
    ao = cv2.GaussianBlur(ao, (0, 0), 1.0)
    Image.fromarray((ao * 255).astype(np.uint8)).save(os.path.join(od, "pbr_ao.png"))

    # Height = depth 대비 정규화 (16-bit)
    height = norm01(depth, 2, 98)
    Image.fromarray((height * 65535).astype(np.uint16)).save(os.path.join(od, "pbr_height.png"))

    # Displacement = panorama 휘도 고주파 (16-bit, 0.5 중심)
    H, W = depth.shape[:2]
    lum = np.array(Image.open(panorama).convert("L").resize((W, H))).astype(np.float32) / 255
    blur = cv2.GaussianBlur(lum, (0, 0), 6)
    s = np.percentile(np.abs(lum - blur), 98) + 1e-6
    disp = np.clip(0.5 + 0.5 * np.clip((lum - blur) / (3 * s), -1, 1), 0, 1)
    Image.fromarray((disp * 65535).astype(np.uint16)).save(os.path.join(od, "pbr_displacement.png"))

    print(f"  derived AO(mean={ao.mean():.3f}) / Height / Displacement -> {od}")


if __name__ == "__main__":
    main()
