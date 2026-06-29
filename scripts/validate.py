"""검증기 (기준서 §5 HDRI / §6 PBR / §7 정합성 / §14 자동불합격).
결과 폴더를 받아 케이스별 O/X 를 터미널에 출력하고 JSON/CSV 리포트를 저장.
사용:
  python validate.py outputs/01_sunny_plaza      # 단일
  python validate.py outputs                      # 전체 (하위 Result 폴더 일괄)
"""
import os, sys, json, csv, glob
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np
from PIL import Image
from skimage import color as skcolor
from scipy import ndimage as ndi
import cv2

PW, PH = 4096, 2048
POLE = 0.08
BAND = 32
OK, NO = "O", "X"

# 결과폴더 내 역할별 파일 후보경로 (spec 레이아웃 / flat 모두 허용)
PATHS = {
    "exr": ["hdri/final/hdri_final.exr", "hdri.exr"],
    "hdri_preview": ["hdri/final/hdri_preview.png", "hdri_preview.png"],
    "basecolor": ["pbr/final/pbr_basecolor.png", "pbr_basecolor.png"],
    "normal": ["pbr/final/pbr_normal.png", "pbr_normal.png", "pbr_normal_omnix.png"],
    "roughness": ["pbr/final/pbr_roughness.png", "pbr_roughness.png"],
    "metallic": ["pbr/final/pbr_metallic.png", "pbr_metallic.png"],
    "ao": ["pbr/final/pbr_ao.png", "pbr_ao.png"],
    "height": ["pbr/final/pbr_height.png", "pbr_height.png"],
    "displacement": ["pbr/final/pbr_displacement.png", "pbr_displacement.png"],
    "material_id": ["pbr/final/pbr_material_id.png", "pbr_material_id.png"],
}


def find(d, role):
    for p in PATHS[role]:
        fp = os.path.join(d, p)
        if os.path.isfile(fp):
            return fp
    return None


def lab(rgb8):
    return skcolor.rgb2lab(rgb8 / 255.0)


def seam_rgb(img):
    L = lab(img[int(PH * POLE):PH - int(PH * POLE)])
    de = skcolor.deltaE_ciede2000(L[:, :1], L[:, -1:]).ravel()
    le, ri = L[:, :BAND].mean(1, keepdims=True), L[:, -BAND:].mean(1, keepdims=True)
    de32 = skcolor.deltaE_ciede2000(le, ri).ravel()
    return de.mean(), np.percentile(de, 95), de32.mean(), np.percentile(de32, 95)


def seam_scalar(g, lo=0.05, hi=0.15):
    diff = np.abs(g[:, 0] - g[:, -1])
    return float(diff.mean()), float(np.percentile(diff, 95))


def seam_normal(n8):
    n = n8.astype(np.float32) / 255 * 2 - 1
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-8
    dot = (n[:, 0] * n[:, -1]).sum(1)
    ang = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    return float(ang.mean()), float(np.percentile(ang, 95))


def edges(gray):
    s = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3))
    return s >= np.percentile(s, 85)


def alignment(base, ref):
    if base.shape[:2] != (PH, PW): base = cv2.resize(base, (PW, PH))
    if ref.shape[:2] != (PH, PW): ref = cv2.resize(ref, (PW, PH))
    eb = edges(cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)); er = edges(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY))
    dist = ndi.distance_transform_edt(~er); vals = dist[eb]
    return float(vals.mean()), float(np.percentile(vals, 95))


def load_rgb(p):
    return np.array(Image.open(p).convert("RGB"))


def load_gray01(p):
    a = np.array(Image.open(p))
    return a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0)


def validate_result(d, verbose=True):
    name = os.path.basename(d.rstrip("/"))
    checks = []  # (group, label, passed, detail)

    def chk(group, label, passed, detail=""):
        checks.append((group, label, bool(passed), detail))

    # ---------- §5 HDRI ----------
    exr_p = find(d, "exr"); prev_p = find(d, "hdri_preview")
    exr = cv2.imread(exr_p, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR) if exr_p else None
    chk("HDRI", "파일 존재/열람", exr is not None, exr_p or "없음")
    if exr is not None:
        exr = cv2.cvtColor(exr, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w = exr.shape[:2]
        chk("HDRI", "해상도/비율(2:1)", abs(w / h - 2) <= 0.005, f"{w}x{h} aspect={w/h:.3f}")
        chk("HDRI", "진짜 HDR(값>1)", exr.max() > 1.0, f"max={exr.max():.1f}")
    prev = load_rgb(prev_p) if prev_p else None
    if prev is not None:
        if prev.shape[:2] != (PH, PW):
            prev = cv2.resize(prev, (PW, PH))
        ylo, yhi = int(PH * POLE), PH - int(PH * POLE)
        m1, p1, m32, p32 = seam_rgb(prev)
        chk("HDRI", "seam 1px ΔE (mean≤5,95p≤12)", m1 <= 5 and p1 <= 12, f"mean={m1:.2f} 95p={p1:.2f}")
        chk("HDRI", "seam 32px band (mean≤7,95p≤15)", m32 <= 7 and p32 <= 15, f"mean={m32:.2f} 95p={p32:.2f}")
        lum = prev.mean(2)
        top, bot = lum[:ylo], lum[-ylo:]
        pole = max(((top < 5) | (top > 250)).mean(), ((bot < 5) | (bot > 250)).mean())
        chk("HDRI", "pole 클리핑 ≤5%", pole <= 0.05, f"{pole*100:.1f}%")
        black = (lum < 2).mean(); white = (lum > 253).mean()
        rgb = prev.reshape(-1, 3).mean(0); bias = rgb.max() / max(rgb.min(), 1e-3); std = prev.std()
        chk("HDRI", "노출(black≤3%,white≤6%,bias≤2.5)", black <= 0.03 and white <= 0.06 and bias <= 2.5,
            f"black={black*100:.1f}% white={white*100:.1f}% bias={bias:.2f}")
        chk("HDRI", "대비 stddev>8", std > 8, f"std={std:.1f}")
        # horizon (불명확 장면은 참고용)
        gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        sy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)); prof = sy.mean(1)
        hy = int(np.argmax(prof[ylo:yhi]) + ylo); pos = hy / PH
        chk("HDRI", "horizon 위치 35~65%(참고)", 0.35 <= pos <= 0.65, f"pos={pos:.2f}")

    # ---------- §6 PBR 필수 채널 (§14 자동불합격) ----------
    req = ["basecolor", "normal", "roughness", "metallic", "ao", "height", "displacement", "material_id"]
    present = {r: find(d, r) for r in req}
    chk("PBR", "필수 8채널 존재", all(present.values()),
        "누락:" + ",".join(r for r in req if not present[r]) or "없음")

    base = load_rgb(present["basecolor"]) if present["basecolor"] else None
    if base is not None:
        chk("PBR", "BaseColor 해상도/비율", base.shape[1] == PW and base.shape[0] == PH, f"{base.shape[1]}x{base.shape[0]}")
        std = base.std(); l = base.mean(2)
        chk("PBR", "BaseColor stddev>5", std > 5, f"std={std:.1f}")
        chk("PBR", "BaseColor black<5% & white<5%", (l < 5).mean() < 0.05 and (l > 250).mean() < 0.05,
            f"black={(l<5).mean()*100:.1f}% white={(l>250).mean()*100:.1f}%")
        m1, p1, _, _ = seam_rgb(base)
        chk("PBR", "BaseColor seam ΔE(1px≤5,95p≤15)", m1 <= 5 and p1 <= 15, f"mean={m1:.2f} 95p={p1:.2f}")
        if prev is not None:
            am, ap = alignment(base, prev)
            chk("정합성", "구조경계 정합(mean≤6,95p≤16)", am <= 6 and ap <= 16, f"mean={am:.1f} 95p={ap:.1f}")

    nrm = load_rgb(present["normal"]) if present["normal"] else None
    if nrm is not None:
        v = nrm.astype(np.float32) / 255 * 2 - 1
        vn = v / (np.linalg.norm(v, axis=2, keepdims=True) + 1e-8)
        ang = np.degrees(np.arccos(np.clip((vn[:, :-1] * vn[:, 1:]).sum(2), -1, 1)))
        chk("PBR", "Normal Blue평균≥150", nrm[..., 2].mean() >= 150, f"blue={nrm[...,2].mean():.0f}")
        chk("PBR", "Normal 평면비율≤85%", (vn[..., 2] > 0.9).mean() <= 0.85, f"flat={(vn[...,2]>0.9).mean()*100:.0f}%")
        chk("PBR", "Normal 인접각도≤30°", ang.mean() <= 30, f"ang={ang.mean():.1f}")
        chk("PBR", "Normal 뒤집힘≤2%", (vn[..., 2] < 0).mean() <= 0.02, f"inv={(vn[...,2]<0).mean()*100:.0f}%")
        sm, s95 = seam_normal(nrm)
        chk("PBR", "Normal seam 각도(mean≤10,95p≤25)", sm <= 10 and s95 <= 25, f"mean={sm:.1f} 95p={s95:.1f}")

    for role, thr, dispname in [("roughness", 3, "Roughness"), ("ao", 3, "AO")]:
        if present[role]:
            g = load_gray01(present[role]); std = g.std() * 255
            chk("PBR", f"{dispname} stddev>3", std > 3, f"std={std:.1f}")
            sm, s95 = seam_scalar(g)
            chk("PBR", f"{dispname} seam(mean≤0.05,95p≤0.15)", sm <= 0.05 and s95 <= 0.15, f"mean={sm:.3f}")
    for role, dispname in [("height", "Height"), ("displacement", "Displacement")]:
        if present[role]:
            g = load_gray01(present[role]); std = g.std() * 255
            chk("PBR", f"{dispname} stddev>3", std > 3, f"std={std:.1f}")
            sm, s95 = seam_scalar(g)
            chk("PBR", f"{dispname} seam(mean≤0.06,95p≤0.18)", sm <= 0.06 and s95 <= 0.18, f"mean={sm:.3f}")
    if present["metallic"]:
        g = load_gray01(present["metallic"]); mean = g.mean() * 255
        chk("PBR", "Metallic 단색아님(전부검정/흰색 불가)", 1 < mean < 254 and g.std() * 255 > 1, f"mean={mean:.1f}")
        sm, s95 = seam_scalar(g)
        chk("PBR", "Metallic seam(mean≤0.05)", sm <= 0.05 and s95 <= 0.15, f"mean={sm:.3f}")

    # ---------- 종합 (참고 항목은 판정 제외) ----------
    hard = [c for c in checks if "참고" not in c[1]]
    npass = sum(1 for c in hard if c[2])
    total = len(hard)
    overall = all(c[2] for c in hard)
    fails = [c[1] for c in hard if not c[2]]

    if verbose:
        print(f"\n{'='*64}\n[{name}]  ({npass}/{total} 통과)")
        grp = None
        for g, label, p, detail in checks:
            if g != grp:
                print(f"  ── {g}"); grp = g
            print(f"    {OK if p else NO}  {label}" + (f"   ({detail})" if detail else ""))
        print(f"  ==> 종합: {OK if overall else NO}" + ("" if overall else f"   실패: {', '.join(fails)}"))

    rep = {"scene": name, "pass": overall, "pass_count": npass, "total": total,
           "checks": [{"group": g, "label": l, "pass": p, "detail": dt} for g, l, p, dt in checks],
           "failure_reasons": fails}
    # validation/ 리포트 저장
    vdir = os.path.join(d, "validation"); os.makedirs(vdir, exist_ok=True)
    json.dump(rep, open(os.path.join(vdir, "final_validation_summary.json"), "w"), indent=2, ensure_ascii=False)
    json.dump({k: v for k, v in rep.items() if v}, open(os.path.join(vdir, "pbr_validation.json"), "w"),
              indent=2, ensure_ascii=False)
    return rep


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    if os.path.isdir(os.path.join(target, "validation")) or find(target, "basecolor") or find(target, "exr"):
        dirs = [target]
    else:
        dirs = sorted(d for d in glob.glob(os.path.join(target, "*")) if os.path.isdir(d))
    reps = [validate_result(d) for d in dirs]
    # 전체 요약 + CSV
    npass = sum(r["pass"] for r in reps)
    print(f"\n{'#'*64}\n총괄: {npass}/{len(reps)} 케이스 통과")
    print("케이스      " + "  ".join(f"{r['scene'][:14]:14s} {OK if r['pass'] else NO}" for r in reps[:0]))
    with open(os.path.join(target, "Validation_Summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "pass", "pass_count", "total", "failure_reasons"])
        for r in reps:
            w.writerow([r["scene"], r["pass"], r["pass_count"], r["total"], "; ".join(r["failure_reasons"])])
    print(f"CSV: {os.path.join(target,'Validation_Summary.csv')}")


if __name__ == "__main__":
    main()
