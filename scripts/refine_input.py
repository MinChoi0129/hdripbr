"""입력 처리 (기준서 §3): 텍스트 -> 구조화 JSON. 규칙기반(학습 없음).
필수 키 누락 0건(§3.5) 보장. image/mixed 입력도 동일 스키마로 처리.
"""
import os, sys, json, re

SCENE = [("industrial", ["factory", "hangar", "industrial", "warehouse", "machinery"]),
         ("city", ["city", "cyberpunk", "street", "urban", "neon"]),
         ("interior", ["library", "room", "indoor", "interior", "hall"]),
         ("nature", ["forest", "jungle", "reef", "desert", "beach", "bamboo", "coral", "mountain", "ocean"]),
         ("fantasy", ["mars", "floating", "sky island", "fantasy", "magic", "alien"]),
         ("exterior", ["plaza", "village", "temple", "cathedral", "ruins", "street"])]
TIME = [("night", ["night", "starry", "milky way", "stars"]), ("sunset", ["sunset", "twilight", "dusk"]),
        ("dawn", ["dawn", "sunrise", "golden hour"]), ("morning", ["morning"]), ("day", ["noon", "day", "midday", "afternoon"])]
WEATHER = [("snowy", ["snow", "snowy"]), ("foggy", ["mist", "misty", "fog", "foggy"]),
           ("rainy", ["rain", "wet", "rainy"]), ("cloudy", ["cloud", "overcast", "pink clouds"]),
           ("clear", ["clear", "sunny", "blue sky", "starry"])]
MOOD = [("high_contrast", ["neon", "high contrast"]), ("dark", ["night", "dark", "dim"]),
        ("warm", ["warm", "golden", "sunset", "cozy"]), ("bright", ["bright", "sunny", "noon"]),
        ("cold", ["snow", "cold", "icy"]), ("soft", ["misty", "soft", "dawn"])]
MATERIALS = {"wood": ["wood", "wooden", "bamboo", "palm"], "metal": ["metal", "steel", "chrome", "aluminum", "brass", "silver"],
             "concrete": ["concrete", "asphalt"], "stone": ["stone", "cobblestone", "rock", "cathedral", "temple", "ruins"],
             "soil": ["soil", "dirt", "terrain", "dune", "sand", "rust-red"], "glass": ["glass", "window"],
             "fabric": ["fabric", "leather", "cloth", "curtain"], "vegetation": ["vine", "tree", "forest", "grass", "coral", "moss"],
             "water": ["water", "ocean", "sea", "reef", "waterfall", "beach", "river"]}
COLORHINT = {"blue sky": "#87CEEB", "snow": "#EAF2FA", "neon": "#FF2D95", "desert": "#C2895B",
             "golden": "#E0A33E", "forest": "#3E6B3A", "rust": "#9C4A2E", "warm": "#C08A4A",
             "twilight": "#E59FB0", "night": "#1B2440", "steel": "#A8AEB5"}


def pick(table, text, default="unspecified"):
    for label, kws in table:
        if any(k in text for k in kws):
            return label
    return default


def refine(text, input_id, input_type="text", image_path=None):
    t = text.lower()
    mats = [m for m, kws in MATERIALS.items() if any(k in t for k in kws)]
    colors = [c for k, c in COLORHINT.items() if k in t]
    if not colors:
        colors = ["#808080"]
    return {
        "input_id": input_id,
        "input_type": input_type,
        "text_raw": text if input_type != "image" else None,
        "image_path": image_path,
        "scene_type": pick(SCENE, t),
        "location": "unspecified",
        "time_of_day": pick(TIME, t),
        "weather": pick(WEATHER, t),
        "lighting_mood": pick(MOOD, t),
        "material_candidates": mats or ["unspecified"],
        "dominant_colors": colors,
        "output_target": "HDRI_PBR",
        "target_hdri_resolution": "4096x2048",
        "target_pbr_resolution": "4096x2048",
        "negative_conditions": [],
        "conflict_flags": [],
        "uncertainty_score": 0.0 if mats else 0.3,
    }


def process(text, input_id, out_dir, input_type="text", image_path=None, created=""):
    """input/ 와 refined/ 폴더 생성."""
    os.makedirs(os.path.join(out_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "refined"), exist_ok=True)
    if input_type != "image":
        open(os.path.join(out_dir, "input", "input_text.txt"), "w").write(text)
    meta = {"input_id": input_id, "input_type": input_type, "created": created,
            "text_path": "input/input_text.txt" if input_type != "image" else None,
            "image_path": image_path}
    json.dump(meta, open(os.path.join(out_dir, "input", "input_meta.json"), "w"), indent=2, ensure_ascii=False)
    refined = refine(text, input_id, input_type, image_path)
    json.dump(refined, open(os.path.join(out_dir, "refined", "refined_prompt.json"), "w"), indent=2, ensure_ascii=False)
    json.dump(refined, open(os.path.join(out_dir, "refined", "merged_conditions.json"), "w"), indent=2, ensure_ascii=False)
    json.dump({"analyzed": input_type in ("image", "text_image_mixed"), "image_path": image_path,
               "scene_type": refined["scene_type"], "dominant_colors": refined["dominant_colors"]},
              open(os.path.join(out_dir, "refined", "image_analysis.json"), "w"), indent=2, ensure_ascii=False)
    # 필수 키 누락 체크
    required = ["input_id", "input_type", "scene_type", "time_of_day", "weather",
                "lighting_mood", "material_candidates", "dominant_colors", "output_target"]
    missing = [k for k in required if k not in refined or refined[k] in (None, [], "")]
    return refined, missing


if __name__ == "__main__":
    text = open(sys.argv[1]).read().strip() if os.path.isfile(sys.argv[1]) else sys.argv[1]
    iid = sys.argv[2] if len(sys.argv) > 2 else "Result_001"
    out = sys.argv[3] if len(sys.argv) > 3 else f"outputs/{iid}"
    refined, missing = process(text, iid, out)
    print(json.dumps(refined, indent=2, ensure_ascii=False))
    print("필수키 누락:", missing or "없음")
