"""OmniX semantic 만 배치로 재생성 (FLUX 1회 로드 -> 여러 파노라마 루프).
하늘/식생/물 마스킹(Fix1)용 pbr_semantic.png 를 각 출력 폴더에 저장한다.

사용:
  python scripts/regen_semantic.py --jobs <pano1>:<outdir1> <pano2>:<outdir2> ...
  python scripts/regen_semantic.py --list jobs.txt          # 한 줄에 "pano::outdir"
각 outdir 에 pbr_semantic.png (4096x2048 NEAREST) 저장.
"""
import os, os.path as osp, sys, argparse
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
from PIL import Image
import torch
from src.systems.omnix import OmniX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="*", default=[], help="pano::outdir 쌍들")
    ap.add_argument("--list", default=None, help="pano::outdir 줄 목록 파일")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--W", type=int, default=4096)
    ap.add_argument("--H", type=int, default=2048)
    a = ap.parse_args()

    jobs = list(a.jobs)
    if a.list:
        jobs += [ln.strip() for ln in open(a.list) if ln.strip() and "::" in ln]
    pairs = [j.split("::", 1) for j in jobs]
    pairs = [(p, o) for p, o in pairs if osp.exists(p)]
    if not pairs:
        print("no valid jobs"); return

    print(f"OmniX 로드 (semantic x{len(pairs)})...")
    omnix = OmniX(hf_repo="KevinHuang/OmniX", device=None, dtype=torch.bfloat16)
    for i, (pano, outdir) in enumerate(pairs):
        os.makedirs(outdir, exist_ok=True)
        print(f"[{i+1}/{len(pairs)}] semantic <- {pano}")
        panorama = Image.open(pano).convert("RGB")
        sem = omnix.perceive_panoramic_semantic(panorama, num_inference_steps=a.steps)
        sem = sem.convert("RGB").resize((a.W, a.H), Image.NEAREST)
        sem.save(osp.join(outdir, "pbr_semantic.png"))
        print(f"   -> {osp.join(outdir, 'pbr_semantic.png')}")


if __name__ == "__main__":
    main()
