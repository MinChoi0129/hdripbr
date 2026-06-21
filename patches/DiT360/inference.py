import sys
from PIL import Image
from src.pipeline import DiT360Pipeline
import torch

prompt = sys.argv[1] if len(sys.argv) > 1 else "This is a panorama."
out_path = sys.argv[2] if len(sys.argv) > 2 else "result.png"

device = torch.device("cuda:0")
pipe = DiT360Pipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
pipe.load_lora_weights("Insta360-Research/DiT360-Panorama-Image-Generation")

# VRAM 에 맞춰 오프로딩 선택:
#   >= 30GB (예: RTX 5090 32GB)  -> model offload (빠름)
#   < 30GB  (예: RTX 4090 24GB)  -> sequential offload (저VRAM, 느림) + VAE 타일링
total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
if total_gb >= 30:
    pipe.enable_model_cpu_offload(gpu_id=0)
else:
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    pipe.enable_sequential_cpu_offload(gpu_id=0)
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
print(f"[inference] VRAM={total_gb:.1f}GB -> {'model' if total_gb >= 30 else 'sequential'} offload")

image = pipe(
    prompt,
    width=2048,
    height=1024,
    num_inference_steps=28,
    guidance_scale=2.8,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]

image = image.resize((4096, 2048), Image.LANCZOS)
image.save(out_path)
print(f"saved {out_path} {image.size}")
