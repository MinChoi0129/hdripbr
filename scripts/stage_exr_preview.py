"""기존 HDR EXR에서 톤매핑 미리보기 + 3노출 브래킷(5/50/95%) 생성.

사용: python stage_exr_preview.py <hdri.exr> <out_dir>
출력: hdri_preview.png, hdri_preview_p05_bright/_p50_mid/_p95_dark.png
"""
import os, sys
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np
from PIL import Image
import cv2


def lum(x):
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def _smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def reinhard(hdr, gamma=2.2):
    L = np.maximum(lum(hdr), 1e-6)                      # 음수/0 휘도 클램프 (log 안정)
    log_avg = np.exp(np.log(L).mean())
    s = np.maximum(hdr, 0.0) * (0.18 / log_avg)
    Ld = np.clip(s / (1 + s), 0, 1)
    # near-white highlight knee: 극단 고휘도(천정 밝은 하늘 등)가 순백으로 클리핑되지 않게 압축
    Ld = Ld - 0.12 * _smoothstep(0.82, 1.0, Ld) * Ld
    return (np.clip(Ld, 0, 1) ** (1 / gamma) * 255).astype(np.uint8)


def exposure(hdr, pct, gamma=2.2):
    a = np.percentile(lum(hdr), pct) + 1e-6
    return (np.clip(hdr * (0.5 / a), 0, 1) ** (1 / gamma) * 255).astype(np.uint8)


def main():
    exr_path, out_dir = sys.argv[1], sys.argv[2]
    img = cv2.imread(exr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    hdr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    hdr = np.nan_to_num(np.maximum(hdr, 0.0), nan=0.0, posinf=0.0)   # 음수/NaN/inf 제거
    Image.fromarray(reinhard(hdr)).save(os.path.join(out_dir, "hdri_preview.png"))
    for pct, tag in [(50, "mid"), (95, "high"), (99, "peak")]:
        Image.fromarray(exposure(hdr, pct)).save(
            os.path.join(out_dir, f"hdri_preview_p{pct:02d}_{tag}.png"))
    print(f"  EXR previews (mid + 3 exposure) -> {out_dir}  "
          f"[max={hdr.max():.1f} frac>1={(lum(hdr)>1).mean():.4f}]")


if __name__ == "__main__":
    main()
