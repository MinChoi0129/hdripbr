"""[로컬 retrofit 전용] 기존 outputs/<id> 의 HDRI 만 ExpandNet 으로 재생성 (seam offset-feather).
panorama.png(없으면 .archive fallback) -> hdri_final.exr(+previews) 를 hdri/final 에 덮어쓴다.
새 생성은 ./run.sh (포워드 파이프라인) 사용. 이 도구는 이미 만들어진 outputs 가 있을 때만 의미.
사용: python scripts/redo_hdri.py [outputs] [--only 01_sunny_plaza,...]
"""
import os, sys, re, glob, shutil, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import seamfix
CONDA = os.environ.get("CONDA_BASE", "/opt/conda")
PY = f"{CONDA}/envs/dit360/bin/python"
EXPAND = os.path.join(ROOT, "models", "ExpandNet")
ARCHIVE = os.path.join(ROOT, ".archive", "outputs_flat", "dit360")


def redo_one(out_dir, scratch):
    out_dir = os.path.abspath(out_dir)
    iid = os.path.basename(out_dir.rstrip("/"))
    pano = os.path.join(out_dir, "panorama.png")
    if not os.path.exists(pano):
        bare = re.sub(r"^\d+_", "", iid)
        pano = os.path.join(ARCHIVE, bare, "panorama.png")
    if not os.path.exists(pano):
        print(f"[{iid}] panorama 없음 -> skip"); return
    pano = os.path.abspath(pano)
    hf = os.path.join(out_dir, "hdri", "final"); os.makedirs(hf, exist_ok=True)
    work = os.path.abspath(os.path.join(scratch, iid)); os.makedirs(work, exist_ok=True)
    # seam 제거된 파노라마로 ExpandNet 실행 (입력 자체의 좌우 불연속 제거)
    seamless = os.path.join(work, "panorama.png")
    shutil.copy(pano, seamless)
    seamfix.fix_image_file(seamless)
    print(f"[{iid}] ExpandNet wrap-pad 재생성(seamless 입력)...")
    subprocess.run(["bash", "-c",
                    f"OPENCV_IO_ENABLE_OPENEXR=1 {PY} expand.py {seamless} --use_exr True --out {work}"],
                   cwd=EXPAND, check=True)
    exr = os.path.join(work, "panorama_prediction.exr")
    dst = os.path.join(hf, "hdri_final.exr")
    shutil.copy(exr, dst); os.remove(exr)
    seamfix.fix_exr(dst)                                  # EXR 1px seam offset-feather(비블러)
    subprocess.run([PY, os.path.join(HERE, "stage_exr_preview.py"), dst, hf], check=True)


def main():
    base = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(ROOT, "outputs")
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    dirs = sorted(d for d in glob.glob(os.path.join(base, "*"))
                  if os.path.isdir(os.path.join(d, "hdri")))
    if only:
        dirs = [d for d in dirs if os.path.basename(d) in only]
    scratch = tempfile.mkdtemp(prefix="redo_hdri_")
    for d in dirs:
        try:
            redo_one(d, scratch)
        except subprocess.CalledProcessError as e:
            print(f"  [fail] {os.path.basename(d)}: {e}")
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
