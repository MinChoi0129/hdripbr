"""[로컬 retrofit 전용] 기존 outputs/<id> 에 후처리 재적용 (무-GPU).

⚠️ 이 도구는 .archive/outputs_flat (이 서버에 아카이브된 pristine 모델 출력) 에 의존한다.
   새 서버에는 .archive 가 없으므로 동작하지 않는다 — 새 서버에서는 `./run.sh` (포워드 파이프라인)로 새로 생성할 것.
   (HDRI 재생성은 redo_hdri.py, semantic 재생성은 regen_semantic.py 가 별도 담당.)

archive flat(pristine 모델 출력)에서 채널을 복원한 뒤 후처리만 다시 적용하므로 idempotent 하다.
  1. archive 원본 채널 복원 (basecolor/metallic/roughness/depth/normal_omnix/normal_marigold/ao/height/displacement)
  2. 납품 normal = Marigold (pbr_normal.png); 폐기된 pbr_normal_depth.png 정리
  3. 하늘/구름/물 검정 albedo 채움 (BaseColor 만)
  4. 채널 seam offset-feather (비블러)
  5. material_id + manifest 재생성
  6. 재검증 (O/X)

사용: python scripts/apply_fixes.py [outputs] [--only 01_sunny_plaza,...]
"""
import os, sys, re, shutil, glob
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import stage_semantic_fix, pbr_extra, validate, seamfix

ARCHIVE = os.path.join(ROOT, ".archive", "outputs_flat", "dit360")


def archive_dir(iid):
    bare = re.sub(r"^\d+_", "", iid)
    d = os.path.join(ARCHIVE, bare)
    return d if os.path.isdir(d) else None


def fix_one(out_dir):
    iid = os.path.basename(out_dir.rstrip("/"))
    pf = os.path.join(out_dir, "pbr", "final")
    if not os.path.isdir(pf):
        print(f"[{iid}] pbr/final 없음 -> skip"); return None
    arch = archive_dir(iid)

    # 1. archive 원본(pristine OmniX/Marigold)에서 채널 복원 -> 재실행 가능(idempotent).
    #    pbr_semantic.png(재생성본)은 pf 에 유지.
    PRISTINE = ["pbr_basecolor.png", "pbr_metallic.png", "pbr_roughness.png", "pbr_depth.png",
                "pbr_normal_omnix.png", "pbr_normal_marigold.png",
                "pbr_ao.png", "pbr_height.png", "pbr_displacement.png"]
    if arch:
        for f in PRISTINE:
            src = os.path.join(arch, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(pf, f))

    # 2. 납품 normal = Marigold (tangent-space, 시각/검증 모두 양호). omnix 는 참고로 동봉.
    mg = os.path.join(pf, "pbr_normal_marigold.png")
    cur_n = os.path.join(pf, "pbr_normal.png")
    if os.path.exists(mg):
        shutil.copy(mg, cur_n)
    dn = os.path.join(pf, "pbr_normal_depth.png")        # 폐기된 depth-normal 정리
    if os.path.exists(dn):
        os.remove(dn)

    # 3. 하늘/구름/물 검정 albedo 채움 (basecolor 만)
    pano = os.path.join(out_dir, "panorama.png")
    if not os.path.exists(pano) and arch:
        pano = os.path.join(arch, "panorama.png")
    if os.path.exists(pano):
        stage_semantic_fix.run(pf, pano)

    # 4. seam 정합(offset-feather, 비블러)
    ch = seamfix.fix_pbr_dir(pf)
    print(f"  seam feather -> {len(ch)} channels")

    # 5. material_id 재생성
    try:
        n = pbr_extra.make_material_id(pf)
        mm = os.path.join(pf, "material_manifest.json")
        if os.path.exists(mm):
            shutil.copy(mm, os.path.join(out_dir, "material", "material_manifest.json"))
    except Exception as e:
        print(f"  [warn] material_id: {e}")

    # 6. 재검증
    return validate.validate_result(out_dir, verbose=True)


def main():
    if not os.path.isdir(ARCHIVE):
        print(f"[중단] {ARCHIVE} 없음 — 이 도구는 로컬 retrofit 전용입니다.\n"
              f"       새 서버에서는 ./run.sh (포워드 파이프라인)로 생성하세요.")
        return
    base = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(ROOT, "outputs")
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    dirs = sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)
                  and os.path.isdir(os.path.join(d, "pbr")))
    if only:
        dirs = [d for d in dirs if os.path.basename(d) in only]
    for d in dirs:
        print(f"\n################ {os.path.basename(d)} ################")
        fix_one(d)


if __name__ == "__main__":
    main()
