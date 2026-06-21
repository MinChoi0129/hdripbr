"""compare_dir의 <channel>_<method>.png 들을 채널별 비교 시트로 합본.
사용: python stage_compare_sheets.py <compare_dir>
"""
import os, sys
from PIL import Image, ImageDraw

cdir = sys.argv[1]
channels = {
    "albedo": ["omnix", "marigold"],
    "normal": ["omnix", "marigold", "deepbump", "derived"],
    "roughness": ["omnix", "marigold"],
    "metallic": ["omnix", "marigold"],
    "depth": ["omnix", "marigold", "dav2"],
    "ao": ["deepbump", "derived"],
    "height": ["deepbump", "derived"],
    "displacement": ["deepbump", "derived"],
}
tw, th, lab, pad = 760, 380, 24, 8
for ch, methods in channels.items():
    avail = [(m, os.path.join(cdir, f"{ch}_{m}.png")) for m in methods
             if os.path.exists(os.path.join(cdir, f"{ch}_{m}.png"))]
    if not avail:
        continue
    n = len(avail)
    sheet = Image.new("RGB", (tw + 2 * pad, n * (th + lab) + pad), (18, 18, 22))
    dr = ImageDraw.Draw(sheet)
    for i, (m, p) in enumerate(avail):
        y = pad + i * (th + lab)
        dr.text((pad + 2, y), f"{ch}  —  {m}", fill=(240, 240, 120))
        sheet.paste(Image.open(p).convert("RGB").resize((tw, th)), (pad, y + lab))
    sheet.save(os.path.join(cdir, f"_compare_{ch}.png"))
    print(f"  sheet: _compare_{ch}.png ({n} methods)")
