"""OmniX 출력에서 선택된 채널만 4K 업샘플+정리: albedo->basecolor, roughness, normal->normal_omnix.
(metallic/depth는 Marigold 사용 → 제외)
사용: python stage_omnix_post.py <out_dir> [W H]
"""
import os, sys
from PIL import Image

od = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
H = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
# (src, dst, mode) — roughness 는 grayscale(L), 나머지는 RGB. semantic 은 NEAREST 업샘플.
mapping = [
    ("output_albedo.png",   "pbr_basecolor.png",    "RGB"),
    ("output_roughness.png", "pbr_roughness.png",   "L"),     # 단일채널(Linear 값)
    ("output_normal.png",   "pbr_normal_omnix.png", "RGB"),
    ("output_semantic.png", "pbr_semantic.png",     "RGB"),   # 하늘/물 마스크용 (Fix1)
]
for src, dst, mode in mapping:
    p = os.path.join(od, src)
    if not os.path.exists(p):
        print(f"  [warn] missing {src}"); continue
    im = Image.open(p).convert(mode)
    if im.size != (W, H):
        im = im.resize((W, H), Image.NEAREST if "semantic" in dst else Image.LANCZOS)
    im.save(os.path.join(od, dst))
    print(f"  {src} -> {dst} ({W}x{H}, {mode})")
# OmniX 보조파일 제거 (semantic 은 보존)
for f in ("output_albedo.png", "output_normal.png", "output_roughness.png",
          "output_metallic.png", "output_depth.png", "output_semantic.png",
          "output_stitched.png", "input_panorama.png"):
    fp = os.path.join(od, f)
    if os.path.exists(fp):
        os.remove(fp)
