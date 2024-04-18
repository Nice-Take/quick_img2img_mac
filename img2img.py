import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import time


img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
          "stabilityai/stable-diffusion-xl-refiner-1.0",
          torch_dtype=torch.float16, 
          variant="fp16",
          use_safetensors=True,
          add_watermarker=False
          ).to("cuda")


def generate(inspiration_img: str, prompt: str, strength: float = .85, seed: int = 0, steps: int = 50):
    generator = torch.Generator("cuda").manual_seed(seed)

    orig_name = inspiration_img
    neg_prompt = "3d, illustration, painting, stylized, 2d, vector, overlap, rendering, render, watermark, text"
    inf_steps = steps

    inspiration_img = load_image(inspiration_img)

    refine_prompt = prompt + "bloom, flare, 8k, dslr, depth of field, high detail, detailed"
    image = img2img(prompt=refine_prompt,
                    prompt2=prompt,
                    strength=strength,
                    generator=generator,
                    image=inspiration_img,
                    negative_prompt=neg_prompt,
                    num_inference_steps=inf_steps).images[0]

    save_name = f"{orig_name}_{round(time.time())}.png"
    image.save(save_name)