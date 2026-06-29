"""HDRI + 전 PBR 채널을 한 장에 이어붙인 contact sheet(한눈에 보기) 생성.
각 타일 상단에 라벨. 16-bit/단일채널은 표시용으로 변환.
사용:
  python scripts/make_contact_sheet.py outputs                 # 전체
  python scripts/make_contact_sheet.py outputs/01_sunny_plaza  # 단일
출력: <id>/contact_sheet.png
"""
import os, sys, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TW, TH = 640, 320            # 타일 크기(2:1)
LAB = 22                     # 라벨바 높이
COLS = 3
PAD = 6

# (상대경로, 라벨)  — 순서대로 배치
TILES = [
    ("panorama.png", "Panorama (LDR)"),
    ("hdri/final/hdri_preview.png", "HDRI (preview)"),
    ("hdri/final/hdri_preview_p99_peak.png", "HDRI p99 (광원)"),
    ("pbr/final/pbr_basecolor.png", "BaseColor"),
    ("pbr/final/pbr_normal.png", "Normal (납품=Marigold)"),
    ("pbr/final/pbr_normal_omnix.png", "Normal (OmniX 참고)"),
    ("pbr/final/pbr_roughness.png", "Roughness"),
    ("pbr/final/pbr_metallic.png", "Metallic"),
    ("pbr/final/pbr_ao.png", "AO"),
    ("pbr/final/pbr_height.png", "Height"),
    ("pbr/final/pbr_displacement.png", "Displacement"),
    ("pbr/final/pbr_depth.png", "Depth"),
    ("pbr/final/pbr_material_id.png", "Material ID"),
    ("pbr/final/pbr_semantic.png", "Semantic (하늘/물)"),
]


def _load_disp(path):
    im = Image.open(path)
    a = np.array(im)
    if a.dtype == np.uint16:                       # 16-bit -> 8-bit
        a = (a.astype(np.float32) / 257.0).astype(np.uint8)
        im = Image.fromarray(a)
    return im.convert("RGB").resize((TW, TH), Image.LANCZOS)


def _font():
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, 15)
    return ImageFont.load_default()


def make(scene_dir):
    iid = os.path.basename(scene_dir.rstrip("/"))
    present = [(p, lab) for p, lab in TILES if os.path.exists(os.path.join(scene_dir, p))]
    if not present:
        print(f"[{iid}] 타일 없음 -> skip"); return None
    rows = (len(present) + COLS - 1) // COLS
    cw, ch = TW + PAD, TH + LAB + PAD
    sheet = Image.new("RGB", (COLS * cw + PAD, rows * ch + PAD + 26), (24, 24, 28))
    d = ImageDraw.Draw(sheet); font = _font()
    d.text((PAD + 2, 5), f"{iid}  —  HDRI + PBR contact sheet", fill=(240, 240, 245), font=font)
    for i, (p, lab) in enumerate(present):
        r, c = divmod(i, COLS)
        x = PAD + c * cw; y = 26 + PAD + r * ch
        d.rectangle([x, y, x + TW, y + LAB - 1], fill=(45, 45, 55))
        d.text((x + 5, y + 3), lab, fill=(235, 235, 240), font=font)
        try:
            sheet.paste(_load_disp(os.path.join(scene_dir, p)), (x, y + LAB))
        except Exception as e:
            d.text((x + 5, y + LAB + 10), f"(load err)", fill=(220, 120, 120), font=font)
    out = os.path.join(scene_dir, "contact_sheet.png")
    sheet.save(out)
    print(f"[{iid}] -> contact_sheet.png ({sheet.size[0]}x{sheet.size[1]}, {len(present)} tiles)")
    return out


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    if os.path.isdir(os.path.join(base, "pbr")):       # 단일 장면
        make(base)
    else:
        for d in sorted(glob.glob(os.path.join(base, "*"))):
            if os.path.isdir(os.path.join(d, "pbr")):
                make(d)


if __name__ == "__main__":
    main()
