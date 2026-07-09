"""Material ID(+manifest, §6.9) + 엔진 적용 파일(§10) 생성. 학습 없음(클러스터링/템플릿)."""
import os, sys, json
import numpy as np
from PIL import Image
import cv2

_PAL = np.array([[70, 130, 180], [60, 179, 113], [205, 133, 63], [178, 34, 34],
                 [138, 43, 226], [255, 215, 0], [128, 128, 128], [0, 139, 139]], np.uint8)
_FIELDS = ["sky", "ground_floor", "wall_building", "object", "metal", "glass_water", "vegetation", "light_source"]


def _name(beh):
    met, rgh, ao = beh["metallic"], beh["roughness"], beh["ao"]
    if met > 0.5: return "metal"
    if rgh < 0.25: return "glass_or_polished"
    if ao < 0.5: return "crevice_object"
    if rgh > 0.7: return "rough_surface"
    return "generic_material"


def make_material_id(od, k=6):
    g = lambda p: np.array(Image.open(os.path.join(od, p)).convert("L")).astype(np.float32)  # 단일채널 보장
    _bc = "pbr_basecolor_with_semantic.png"                 # 물리보정본으로 세그먼트(원본 없으면 fallback)
    if not os.path.exists(os.path.join(od, _bc)):
        _bc = "pbr_basecolor.png"
    albedo = np.array(Image.open(os.path.join(od, _bc)).convert("RGB")).astype(np.float32) / 255
    rough = g("pbr_roughness.png") / 255.0
    metal = g("pbr_metallic.png") / 255.0
    ao = g("pbr_ao.png") / 255.0
    height = (np.array(Image.open(os.path.join(od, "pbr_height.png"))).astype(np.float32)
              / (65535.0 if np.array(Image.open(os.path.join(od, "pbr_height.png"))).dtype == np.uint16 else 255.0))
    H, W = rough.shape
    feat = np.concatenate([cv2.GaussianBlur(albedo, (0, 0), 2), rough[..., None], metal[..., None]], -1)
    ds = cv2.resize(feat, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    Z = ds.reshape(-1, 5).astype(np.float32); Z[:, 3] *= 1.5; Z[:, 4] *= 2.0
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, _ = cv2.kmeans(Z, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    ids = cv2.medianBlur(cv2.resize(lab.reshape(ds.shape[:2]).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST), 5)
    manifest = []
    for i in range(k):
        m = ids == i
        ratio = float(m.mean()) * 100
        if ratio < 0.3: continue
        beh = {n: round(float((mp.mean(-1) if mp.ndim == 3 else mp)[m].mean()), 4)
               for n, mp in [("metallic", metal), ("roughness", rough), ("ao", ao), ("height", height)]}
        manifest.append({"id": int(i), "name": _name(beh), "color_rgb": [int(c) for c in _PAL[i % 8]],
                         "region_ratio_percent": round(ratio, 3), "expected_behavior": beh})
    Image.fromarray(_PAL[ids % 8]).save(os.path.join(od, "pbr_material_id.png"))
    json.dump({"materials": manifest, "unknown_ratio_percent": 0.0},
              open(os.path.join(od, "material_manifest.json"), "w"), indent=2, ensure_ascii=False)
    return len(manifest)


def make_engine(material_dir, iid, res="4096x2048"):
    os.makedirs(material_dir, exist_ok=True)
    setup = {
        "material_name": f"M_HDRI_PBR_{iid}", "uv_type": "equirectangular_2_to_1",
        "texture_resolution": res, "normal_space": "OpenGL",
        "textures": {"hdri": "../hdri/final/hdri_final.exr",
                     "basecolor": "../pbr/final/pbr_basecolor_with_semantic.png", "normal": "../pbr/final/pbr_normal.png",
                     "roughness": "../pbr/final/pbr_roughness.png", "metallic": "../pbr/final/pbr_metallic.png",
                     "ao": "../pbr/final/pbr_ao.png", "height": "../pbr/final/pbr_height.png",
                     "displacement": "../pbr/final/pbr_displacement.png", "material_id": "../pbr/final/pbr_material_id.png"},
        "all_textures_same_resolution": True, "all_textures_same_uv": True}
    json.dump(setup, open(os.path.join(material_dir, "ue_material_setup.json"), "w"), indent=2, ensure_ascii=False)
    txt = ("# Unreal Engine texture import settings\n"
           "BaseColor : sRGB=ON,  Compression=Default\n"
           "Normal    : sRGB=OFF, Compression=Normalmap, NormalSpace=OpenGL\n"
           "Roughness : sRGB=OFF, Compression=Masks (Linear)\n"
           "Metallic  : sRGB=OFF, Compression=Masks (Linear)\n"
           "AO        : sRGB=OFF, Compression=Masks (Linear)\n"
           "Height    : sRGB=OFF, Compression=Masks (Linear), 16-bit\n"
           "Displacement: sRGB=OFF, Compression=Masks (Linear), 16-bit\n"
           "MaterialID: sRGB=OFF, Compression=VectorDisplacementmap (no filter)\n"
           "HDRI      : HDR, used as Environment/SkyLight (equirectangular UV)\n"
           "UV        : all textures share the same equirectangular 2:1 UV\n")
    open(os.path.join(material_dir, "ue_texture_import_settings.txt"), "w").write(txt)


if __name__ == "__main__":
    od = sys.argv[1]
    n = make_material_id(od)
    print("material regions:", n)
