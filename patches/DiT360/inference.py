import sys
from PIL import Image
from src.pipeline import DiT360Pipeline
import torch

prompt = sys.argv[1] if len(sys.argv) > 1 else "This is a panorama."
out_path = sys.argv[2] if len(sys.argv) > 2 else "result.png"

device = torch.device("cuda:0")
pipe = DiT360Pipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
pipe.load_lora_weights("Insta360-Research/DiT360-Panorama-Image-Generation")
pipe.enable_model_cpu_offload(gpu_id=0)

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
