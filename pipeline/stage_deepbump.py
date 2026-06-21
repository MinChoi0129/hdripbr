"""Stage DeepBump: 파노라마 -> AO / Height / Displacement (DeepBump, GitHub 사전학습 DL).

DeepBump(HugoTini): color->normals 는 CNN(ONNX 사전학습), 거기서 normals->height,
normals->curvature 를 산출. surface-relief 의미의 Height/Displacement/AO 에 부합.

사용: python stage_deepbump.py <panorama.png> <out_dir> <deepbump_dir>
출력: pbr_ao.png(8bit), pbr_height.png(16bit), pbr_displacement.png(16bit)
"""
import os, sys, subprocess, tempfile
import numpy as np
from PIL import Image
import cv2


def main():
    panorama, out_dir, db_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp()
    nrm, h, c = (os.path.join(tmp, x) for x in ("n.png", "h.png", "c.png"))
    PY = sys.executable

    def run(inp, outp, mod):
        subprocess.run([PY, "cli.py", inp, outp, mod], cwd=db_dir, check=True,
                       stdout=subprocess.DEVNULL)

    run(panorama, nrm, "color_to_normals")              # DL (ONNX)
    run(nrm, h, "normals_to_height")
    run(nrm, c, "normals_to_curvature")

    def norm01(x, lo=1, hi=99):
        a, b = np.percentile(x, [lo, hi]); return np.clip((x - a) / max(b - a, 1e-6), 0, 1)

    # Height (16-bit)
    H = np.array(Image.open(h).convert("L")).astype(np.float32) / 255.0
    H = norm01(H, 2, 98)
    Image.fromarray((H * 65535).astype(np.uint16)).save(os.path.join(out_dir, "pbr_height.png"))

    # Displacement = height 고주파(미세 변위, 0.5 중심, 16-bit)
    blur = cv2.GaussianBlur(H, (0, 0), 8)
    disp = np.clip(0.5 + (H - blur) * 3.0, 0, 1)
    Image.fromarray((disp * 65535).astype(np.uint16)).save(os.path.join(out_dir, "pbr_displacement.png"))

    # AO: curvature(0.5 중심)에서 오목(<0.5)을 가림으로 -> 평면=흰색
    C = np.array(Image.open(c).convert("L")).astype(np.float32) / 255.0
    occ = np.clip(0.5 - C, 0, 0.5) * 2.0
    occ = cv2.GaussianBlur(occ, (0, 0), 1.0)
    ao = 1.0 - np.clip(occ, 0, 1)
    Image.fromarray((ao * 255).astype(np.uint8)).save(os.path.join(out_dir, "pbr_ao.png"))

    print(f"  DeepBump AO/Height/Displacement -> {out_dir}  (AO mean={ao.mean():.3f})")


if __name__ == "__main__":
    main()
