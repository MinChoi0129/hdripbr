"""Marigold 최종 채널: metallic + normal + depth (풀 equirect 단일추론, seam 없음).
pbrhq env. 출력: pbr_metallic.png, pbr_normal_marigold.png, pbr_depth.png
사용: python stage_marigold_final.py <panorama.png> <out_dir> [--res 2048 --ensemble 8 --steps 4]
"""
import os, sys, argparse
import numpy as np
from PIL import Image
import torch


def wrap_pad(img, pad):
    return np.concatenate([img[:, -pad:], img, img[:, :pad]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panorama"); ap.add_argument("out_dir")
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--ensemble", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    from diffusers import (MarigoldIntrinsicsPipeline, MarigoldNormalsPipeline,
                           MarigoldDepthPipeline)
    dt = torch.float16
    equi = np.array(Image.open(a.panorama).convert("RGB"))
    H, W = equi.shape[:2]; pad = W // 16
    padded = Image.fromarray(wrap_pad(equi, pad))

    def run(p):
        return p(padded, num_inference_steps=a.steps, ensemble_size=a.ensemble,
                 processing_resolution=a.res, match_input_resolution=True).prediction

    def unpad(x):
        return x[:, pad:pad + W]

    def save(arr, name):
        Image.fromarray(arr).save(os.path.join(a.out_dir, name))

    print("  [marigold] metallic (intrinsics)...")
    iid = MarigoldIntrinsicsPipeline.from_pretrained(
        "prs-eth/marigold-iid-appearance-v1-1", torch_dtype=dt).to("cuda")
    iid.set_progress_bar_config(disable=True)
    pred = np.asarray(run(iid))
    save((np.clip(unpad(pred[1][..., 1]), 0, 1) * 255).astype(np.uint8), "pbr_metallic.png")
    del iid; torch.cuda.empty_cache()

    print("  [marigold] normal...")
    nrm = MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-v1-1", torch_dtype=dt).to("cuda")
    nrm.set_progress_bar_config(disable=True)
    n = np.asarray(run(nrm)); n = n[0] if n.ndim == 4 else n
    n = unpad(n); n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    save(((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8), "pbr_normal_marigold.png")
    del nrm; torch.cuda.empty_cache()

    print("  [marigold] depth...")
    dep = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-v1-1", torch_dtype=dt).to("cuda")
    dep.set_progress_bar_config(disable=True)
    d = np.asarray(run(dep)); d = d[0] if d.ndim == 4 else d
    d = unpad(np.squeeze(d)).astype(np.float32)
    d = (d - d.min()) / max(d.max() - d.min(), 1e-6)
    save((d * 65535).astype(np.uint16), "pbr_depth.png")
    print(f"  [marigold] metallic/normal/depth -> {a.out_dir}")


if __name__ == "__main__":
    main()
