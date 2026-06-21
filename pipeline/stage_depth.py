"""Stage Depth: 파노라마 PNG -> 고품질 depth (Depth Anything V2, 2024 사전학습).

OmniX depth가 흐릿한 문제를 대체. 학습 없음 — HF 공식 가중치 추론만.
equirectangular 왜곡이 있으나 OmniX 대비 구조/경계가 훨씬 선명.

사용:
  python stage_depth.py <panorama.png> <out_dir> [--size 2048] [--encoder vitl]
출력:
  <out_dir>/pbr_depth.png   (16-bit, near=밝음, 0~65535 상대깊이)
  <out_dir>/pbr_depth8.png  (8-bit 미리보기)
"""
import os, sys, argparse
import numpy as np
from PIL import Image
import torch


def run_depth_anything(img: Image.Image, size: int, encoder: str) -> np.ndarray:
    """Depth Anything V2. transformers 파이프라인 우선, 실패 시 안내."""
    from transformers import pipeline
    repo = {
        "vits": "depth-anything/Depth-Anything-V2-Small-hf",
        "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
        "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
    }[encoder]
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("depth-estimation", model=repo, device=device)
    # 메모리/속도 위해 다운스케일 후 추론, 이후 원본 비율로 업샘플
    w, h = img.size
    if w > size:
        small = img.resize((size, size // 2), Image.LANCZOS)
    else:
        small = img
    out = pipe(small)
    depth = np.array(out["depth"]).astype(np.float32)   # 상대 inverse depth (큰값=가까움)
    return depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panorama")
    ap.add_argument("out_dir")
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    img = Image.open(args.panorama).convert("RGB")
    W, H = img.size

    try:
        depth = run_depth_anything(img, args.size, args.encoder)
    except torch.cuda.OutOfMemoryError:
        print("  [warn] GPU OOM -> vits/CPU 재시도")
        torch.cuda.empty_cache()
        depth = run_depth_anything(img, 1024, "vits")

    # 정규화 [0,1], near=1
    d = depth - depth.min()
    d = d / max(d.max(), 1e-6)
    # 원본 해상도로 업샘플
    d_img = Image.fromarray((d * 65535).astype(np.uint16)).resize((W, H), Image.LANCZOS)
    d_img.save(os.path.join(args.out_dir, "pbr_depth.png"))
    Image.fromarray((np.array(d_img) / 256).astype(np.uint8)).save(
        os.path.join(args.out_dir, "pbr_depth8.png"))
    print(f"  depth   : {os.path.join(args.out_dir,'pbr_depth.png')} ({W}x{H}, 16-bit)")


if __name__ == "__main__":
    main()
