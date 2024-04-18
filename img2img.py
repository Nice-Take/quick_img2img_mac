import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoPipelineForText2Image, StableDiffusionUpscalePipeline
from PIL import Image
from diffusers.utils import load_image, make_image_grid
import time
import upscale


img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
          "stabilityai/stable-diffusion-xl-refiner-1.0",
          torch_dtype=torch.float16, 
          variant="fp16",
          use_safetensors=True,
          add_watermarker=False
          ).to("cuda")


def generate(inspiration_img: str, prompt: str, seed: int = 0, steps: int = 50):
    generator = torch.Generator("cuda").manual_seed(seed)

    orig_name = inspiration_img
    neg_prompt = "3d, illustration, painting, stylized, 2d, vector, overlap, rendering, render, watermark, text"
    inf_steps = steps

    inspiration_img = load_image(inspiration_img)

    refine_prompt = prompt + "bloom, flare, 8k, dslr, depth of field, high detail, detailed"
    image = img2img(prompt=refine_prompt,
                    prompt2=prompt,
                    strength=.88,
                    generator=generator,
                    image=inspiration_img,
                    negative_prompt=neg_prompt,
                    num_inference_steps=inf_steps).images[0]

    save_name = f"{orig_name}_{round(time.time())}.png"
    image.save(save_name)

    #upscale.create(prompt=refine_prompt, image_name=save_name)

generate("./sample_1024x1024.png", "amber glass bottle with label dark wood foreground in a night time scene", seed=99)