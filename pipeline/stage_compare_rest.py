"""비교용 나머지 변형 생성 (dit360 env) + 채널별 비교 시트.
- OmniX(legacy 재사용): albedo/normal/roughness/metallic/depth _omnix
- DA-V2: depth_dav2
- DeepBump: normal/ao/height/displacement _deepbump
- 직접구현(derived): normal/ao/height/displacement _derived
사용: python stage_compare_rest.py <panorama> <compare_dir> <legacy_omnix_dir> <deepbump_dir>
"""
import os, sys, subprocess, tempfile
import numpy as np
from PIL import Image
import cv2

panorama, cdir, omnix_dir, db_dir = sys.argv[1:5]
os.makedirs(cdir, exist_ok=True)
W, H = 4096, 2048
PY = sys.executable


def save(arr, name, bits=8):
    if bits == 16:
        Image.fromarray((np.clip(arr, 0, 1) * 65535).astype(np.uint16)).save(os.path.join(cdir, name))
    elif arr.ndim == 3:
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(os.path.join(cdir, name))
    else:
        Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(os.path.join(cdir, name))


def rs(im):
    return im.resize((W, H), Image.LANCZOS) if im.size != (W, H) else im


def norm01(x, lo=2, hi=98):
    a, b = np.percentile(x, [lo, hi]); return np.clip((x - a) / max(b - a, 1e-6), 0, 1)


# ---------- OmniX (legacy 재사용) ----------
for src, dst in [("output_albedo.png", "albedo_omnix.png"), ("output_normal.png", "normal_omnix.png"),
                 ("output_roughness.png", "roughness_omnix.png"), ("output_metallic.png", "metallic_omnix.png"),
                 ("output_depth.png", "depth_omnix.png")]:
    p = os.path.join(omnix_dir, src)
    if os.path.exists(p):
        rs(Image.open(p).convert("RGB")).save(os.path.join(cdir, dst))
print("  omnix variants copied")

# ---------- Depth Anything V2 ----------
from transformers import pipeline
import torch
dav2 = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf",
                device=0 if torch.cuda.is_available() else -1)
pano = Image.open(panorama).convert("RGB")
dd = np.array(dav2(pano.resize((1036, 518)))["depth"]).astype(np.float32)
dd = norm01(dd, 1, 99)
depth_dav2 = np.array(Image.fromarray((dd * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)).astype(np.float32) / 255
save(depth_dav2, "depth_dav2.png")
print("  dav2 depth done")

# ---------- DeepBump ----------
tmp = tempfile.mkdtemp()
nrm, h, c = (os.path.join(tmp, x) for x in ("n.png", "h.png", "c.png"))
def db(inp, outp, mod):
    subprocess.run([PY, "cli.py", inp, outp, mod], cwd=db_dir, check=True, stdout=subprocess.DEVNULL)
db(panorama, nrm, "color_to_normals")
db(nrm, h, "normals_to_height")
db(nrm, c, "normals_to_curvature")
rs(Image.open(nrm).convert("RGB")).save(os.path.join(cdir, "normal_deepbump.png"))
Hd = norm01(np.array(Image.open(h).convert("L")).astype(np.float32) / 255)
save(Hd, "height_deepbump.png", 16)
blur = cv2.GaussianBlur(Hd, (0, 0), 8); save(np.clip(0.5 + (Hd - blur) * 3, 0, 1), "displacement_deepbump.png", 16)
C = np.array(Image.open(c).convert("L")).astype(np.float32) / 255
occ = cv2.GaussianBlur(np.clip(0.5 - C, 0, 0.5) * 2, (0, 0), 1)
save(1 - np.clip(occ, 0, 1), "ao_deepbump.png")
print("  deepbump variants done")

# ---------- 직접구현 (derived: DA-V2 depth 기반) ----------
d = depth_dav2  # near=bright [0,1]
# normal from depth heightfield
zx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=5)
zy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=5)
strength = 8.0
nrm_d = np.dstack([-zx * strength, -zy * strength, np.ones_like(d)])
nrm_d /= np.maximum(np.linalg.norm(nrm_d, axis=-1, keepdims=True), 1e-8)
save((nrm_d * 0.5 + 0.5), "normal_derived.png".replace(".png", "_rgb.png")) if False else \
    Image.fromarray(((nrm_d * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(cdir, "normal_derived.png"))
# height derived = depth
save(norm01(d, 2, 98), "height_derived.png", 16)
# displacement derived = panorama luminance 고주파
lum = np.array(pano.resize((W, H)).convert("L")).astype(np.float32) / 255
blur = cv2.GaussianBlur(lum, (0, 0), 6); s = np.percentile(np.abs(lum - blur), 98) + 1e-6
save(np.clip(0.5 + 0.5 * np.clip((lum - blur) / (3 * s), -1, 1), 0, 1), "displacement_derived.png", 16)
# AO derived = depth cavity + normal edge
occ = np.zeros_like(d)
for sg in (3, 9, 27, 64):
    occ += np.clip(cv2.GaussianBlur(d, (0, 0), sg) - d, 0, None)
occ = norm01(occ, 50, 99)
gx = np.abs(cv2.Sobel(nrm_d, cv2.CV_32F, 1, 0, ksize=3)).sum(-1)
gy = np.abs(cv2.Sobel(nrm_d, cv2.CV_32F, 0, 1, ksize=3)).sum(-1)
edge = norm01(gx + gy, 50, 99)
save(np.clip(1 - (0.85 * occ + 0.35 * edge), 0, 1), "ao_derived.png")
print("  derived variants done")
print(f"  ALL compare variants -> {cdir}")
