from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import img2img
import time


pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                 use_safetensors=True,
                                                 add_watermarker=False,
                                                 torch_dtype=torch.float16,
                                                 variant="fp16").to("cuda")

def generate(prompt: str, img: str, img_mask: str, strength: float=0.99, seed: int=0) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    neg_prompt = "3d, illustration, painting, stylized, 2d, vector, overlap, rendering, render, watermark, text"

    original_img = load_image(img)
    mask_img = load_image(img_mask)
    image = load_image(original_img).resize((1024, 1024))
    mask_image = load_image(mask_img).resize((1024, 1024))

    image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        negative_prompt=neg_prompt,
        guidance_scale = 10.0, # 8 has been working
        num_inference_steps = 25,  # steps between 15 and 30 work well
        strength=strength,  # make sure to use `strength` below 1.0 << 1 Changes area outside mask the most
        generator=generator,
        ).images[0]

    image.save(f"./generated/{round(time.time())}.png")



original_image = "./sample_13.png"
masked_image = "./sample_13_mask.png"
prompt = "mint leaf next to bottle on counter top"

prompt = prompt + ", bokeh, 8k, 85mm, detailed, uhd, film"
iters = 1

start_time = time.time()
i = 0
while i < iters:
#for prompt in prompts:
    generate(prompt, original_image, masked_image, strength=0.99, seed=i)
    i += 1

print("\n[  TIME  STATS  ]")
print(f"| Total: {round(time.time() - start_time)} sec |")
print(f"| s/Img: {round(round(time.time() - start_time) / iters)} sec |\n")

