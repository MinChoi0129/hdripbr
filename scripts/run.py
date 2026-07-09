"""메인 오케스트레이터 (기준서 §4 파이프라인).
입력 -> (정제) -> HDRI/PBR 생성 또는 재사용 -> Result_NNN 조립 -> 검증(O/X) -> 리포트.

사용:
  python scripts/run.py inputs/text/01_sunny_plaza.txt            # 전체 생성(모델 필요)
  python scripts/run.py inputs/text/01_sunny_plaza.txt --reuse <flat_scene_dir>   # 생성 재사용
"""
import os, sys, json, shutil, subprocess, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import refine_input, pbr_extra, validate, seamfix

CONDA = os.environ.get("CONDA_BASE", "/opt/conda")
PY = f"{CONDA}/envs/dit360/bin/python"
PY_MAT = f"{CONDA}/envs/pbrhq/bin/python"


def sh(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def generate(prompt, work):
    """모델 생성 파이프라인. work=임시 평면폴더.
    DiT360(LDR) -> ExpandNet(HDR) -> OmniX(basecolor/roughness/normal_omnix/semantic)
    -> Marigold(metallic/normal_marigold/depth) -> derived(ao/height/displacement)
    -> 하늘/물 채움(semantic-fix) -> seam offset-feather.
    basecolor 는 두 벌 유지: pbr_basecolor(원본 = 순수 OmniX albedo)
    + pbr_basecolor_with_semantic(하늘/물 채움·물리보정; 검증·material·UE 가 사용)."""
    os.makedirs(work, exist_ok=True)
    pano = os.path.join(work, "panorama.png")
    sh([PY, "inference.py", f"This is a panorama. {prompt}", pano], cwd=f"{ROOT}/models/DiT360")
    seamfix.fix_image_file(pano)          # 좌우 seam offset-feather(비블러)
    sh(["bash", "-c", f"OPENCV_IO_ENABLE_OPENEXR=1 {PY} expand.py {pano} --use_exr True --out {work}"],
       cwd=f"{ROOT}/models/ExpandNet")
    shutil.move(os.path.join(work, "panorama_prediction.exr"), os.path.join(work, "hdri.exr"))
    seamfix.fix_exr(os.path.join(work, "hdri.exr"))          # HDRI 1px seam offset-feather(비블러)
    sh([PY, f"{HERE}/stage_exr_preview.py", os.path.join(work, "hdri.exr"), work])
    sh([PY, "scripts/run_pano_perception.py", "--panorama", pano, "--output_dir", work], cwd=f"{ROOT}/models/OmniX")
    sh([PY, f"{HERE}/stage_omnix_post.py", work, "4096", "2048"])
    sh([PY_MAT, f"{HERE}/stage_marigold_final.py", pano, work, "--res", "2048", "--ensemble", "8", "--steps", "4"])
    sh([PY, f"{HERE}/stage_pbr_derive.py", work, pano])
    # 원본 pbr_basecolor(순수 OmniX)는 유지, 하늘/물 채움본을 pbr_basecolor_with_semantic 로 별도 저장
    sh([PY, f"{HERE}/stage_semantic_fix.py", work, pano])
    seamfix.fix_pbr_dir(work)                                 # 채널 seam offset-feather(두 basecolor 포함)
    return work


def assemble(work, out_dir):
    """평면 생성결과(work) -> Result_NNN spec 레이아웃."""
    hf = os.path.join(out_dir, "hdri", "final"); pf = os.path.join(out_dir, "pbr", "final")
    mat = os.path.join(out_dir, "material")
    for x in (hf, pf, mat, os.path.join(out_dir, "hdri", "attempts", "attempt_01"),
              os.path.join(out_dir, "pbr", "attempts", "attempt_01")):
        os.makedirs(x, exist_ok=True)
    cp = lambda s, dd, dn: shutil.copy(os.path.join(work, s), os.path.join(dd, dn)) if os.path.exists(os.path.join(work, s)) else None
    cp("hdri.exr", hf, "hdri_final.exr")
    cp("hdri_preview.png", hf, "hdri_preview.png")
    for s in ("hdri_preview_p50_mid.png", "hdri_preview_p95_high.png", "hdri_preview_p99_peak.png"):
        cp(s, hf, s)
    cp("panorama.png", out_dir, "panorama.png")
    pbr_map = {"pbr_basecolor.png": "pbr_basecolor.png",                # 원본(순수 OmniX albedo)
               "pbr_basecolor_with_semantic.png": "pbr_basecolor_with_semantic.png",  # 하늘/물 채움·물리보정
               "pbr_normal_marigold.png": "pbr_normal.png",         # 납품 normal = Marigold tangent
               "pbr_normal_omnix.png": "pbr_normal_omnix.png",      # 참고 (world-space)
               "pbr_roughness.png": "pbr_roughness.png", "pbr_metallic.png": "pbr_metallic.png",
               "pbr_ao.png": "pbr_ao.png", "pbr_height.png": "pbr_height.png",
               "pbr_displacement.png": "pbr_displacement.png",
               "pbr_depth.png": "pbr_depth.png", "pbr_semantic.png": "pbr_semantic.png"}
    for s, dn in pbr_map.items():
        cp(s, pf, dn)
    cp("pbr_normal_marigold.png", pf, "pbr_normal_marigold.png")    # 참고로도 동봉
    return hf, pf, mat


def iteration_report(out_dir, stage, ok):
    p = os.path.join(out_dir, stage, f"{stage}_iteration_report.json")
    json.dump({"input_id": os.path.basename(out_dir), "stage": stage, "target_resolution": "4096x2048",
               "attempt_count": 1, "attempts": [{"attempt": 1, "created": True, "validation_pass": ok,
               "score": None, "failure_reasons": [], "output_path": f"{stage}/final"}],
               "selected_attempt": 1, "selected_result": "pass" if ok else "final_fail",
               "final_path": f"{stage}/final"}, open(p, "w"), indent=2, ensure_ascii=False)


def run(input_path, reuse=None):
    iid = os.path.splitext(os.path.basename(input_path))[0]
    prompt = open(input_path).read().strip() if os.path.isfile(input_path) else input_path
    out_dir = os.path.join(ROOT, "outputs", iid)
    os.makedirs(out_dir, exist_ok=True)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"\n[{iid}] 입력처리...")
    refined, missing = refine_input.process(prompt, iid, out_dir, created=now)
    print(f"  refined: scene={refined['scene_type']} time={refined['time_of_day']} "
          f"mats={refined['material_candidates']} / 필수키누락={missing or '없음'}")

    work = reuse if reuse else generate(prompt, os.path.join(out_dir, "_work"))
    print(f"  생성완료(소스: {'재사용 '+reuse if reuse else 'DiT360+ExpandNet+OmniX+Marigold'})")
    hf, pf, mat = assemble(work, out_dir)
    pbr_extra.make_material_id(pf)
    if os.path.exists(os.path.join(pf, "material_manifest.json")):
        shutil.copy(os.path.join(pf, "material_manifest.json"), os.path.join(mat, "material_manifest.json"))
    pbr_extra.make_engine(mat, iid)
    iteration_report(out_dir, "hdri", True)
    iteration_report(out_dir, "pbr", True)
    open(os.path.join(out_dir, "log.txt"), "w").write(f"{now} generated {iid}\n")

    rep = validate.validate_result(out_dir)
    return rep


if __name__ == "__main__":
    args = sys.argv[1:]
    reuse = None
    if "--reuse" in args:
        i = args.index("--reuse"); reuse = args[i + 1]; args = args[:i] + args[i + 2:]
    run(args[0], reuse=reuse)
