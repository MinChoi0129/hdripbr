"""Stage Assemble: 산출물 합본 프리뷰(stitched) + 패노라마 복사.

사용: python stage_assemble.py <out_dir> --panorama <ldr.png>
출력: panorama.png, preview_stitched.png
"""
import os, argparse
import numpy as np
from PIL import Image, ImageDraw


def load_disp(p, tile):
    a = np.array(Image.open(p))
    if a.dtype == np.uint16:
        a = (a / 256).astype(np.uint8)
    im = Image.fromarray(a).convert("RGB")
    return im.resize(tile, Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--panorama", required=True)
    args = ap.parse_args()
    od = args.out_dir
    Image.open(args.panorama).convert("RGB").save(os.path.join(od, "panorama.png"))

    tile = (640, 320)
    items = [
        ("LDR panorama", args.panorama),
        ("HDRI mid(p50)", os.path.join(od, "hdri_preview_p50_mid.png")),
        ("HDRI high(p95)", os.path.join(od, "hdri_preview_p95_high.png")),
        ("HDRI peak(p99)", os.path.join(od, "hdri_preview_p99_peak.png")),
        ("BaseColor (OmniX)", os.path.join(od, "pbr_basecolor.png")),
        ("Roughness (OmniX)", os.path.join(od, "pbr_roughness.png")),
        ("Normal (OmniX)", os.path.join(od, "pbr_normal_omnix.png")),
        ("Normal (Marigold)", os.path.join(od, "pbr_normal_marigold.png")),
        ("Metallic (Marigold)", os.path.join(od, "pbr_metallic.png")),
        ("Depth (Marigold)", os.path.join(od, "pbr_depth.png")),
        ("AO (derived)", os.path.join(od, "pbr_ao.png")),
        ("Height (derived)", os.path.join(od, "pbr_height.png")),
        ("Displacement (derived)", os.path.join(od, "pbr_displacement.png")),
    ]
    items = [(t, p) for t, p in items if os.path.exists(p)]
    cols = 3
    rows = (len(items) + cols - 1) // cols
    pad, lab = 8, 22
    cw, ch = tile[0] + pad, tile[1] + lab + pad
    sheet = Image.new("RGB", (cols * cw + pad, rows * ch + pad), (24, 24, 28))
    dr = ImageDraw.Draw(sheet)
    for i, (title, p) in enumerate(items):
        r, c = divmod(i, cols)
        x, y = pad + c * cw, pad + r * ch
        dr.text((x + 2, y), title, fill=(235, 235, 235))
        sheet.paste(load_disp(p, tile), (x, y + lab))
    sheet.save(os.path.join(od, "preview_stitched.png"))
    print(f"  stitched -> {os.path.join(od,'preview_stitched.png')} ({len(items)} maps)")


if __name__ == "__main__":
    main()
