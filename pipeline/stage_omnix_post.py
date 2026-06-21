"""OmniX 출력에서 선택된 채널만 4K 업샘플+정리: albedo->basecolor, roughness, normal->normal_omnix.
(metallic/depth는 Marigold 사용 → 제외)
사용: python stage_omnix_post.py <out_dir> [W H]
"""
import os, sys
from PIL import Image

od = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
H = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
mapping = {
    "output_albedo.png": "pbr_basecolor.png",
    "output_roughness.png": "pbr_roughness.png",
    "output_normal.png": "pbr_normal_omnix.png",
}
for src, dst in mapping.items():
    p = os.path.join(od, src)
    if not os.path.exists(p):
        print(f"  [warn] missing {src}"); continue
    im = Image.open(p)
    if im.size != (W, H):
        im = im.resize((W, H), Image.LANCZOS)
    im.save(os.path.join(od, dst))
    print(f"  {src} -> {dst} ({W}x{H})")
# OmniX 보조파일 제거
for f in ("output_albedo.png", "output_normal.png", "output_roughness.png",
          "output_metallic.png", "output_depth.png", "output_semantic.png",
          "output_stitched.png", "input_panorama.png"):
    fp = os.path.join(od, f)
    if os.path.exists(fp):
        os.remove(fp)
