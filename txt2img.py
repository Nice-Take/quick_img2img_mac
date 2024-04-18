import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoPipelineForText2Image, StableDiffusionUpscalePipeline
from PIL import Image
from diffusers.utils import load_image, make_image_grid
import time
import upscale

# ------------------------------------------------------------------------------#
#init_image = load_image("./sample2_1024.png") # loading in reference image
#init_mask = load_image("./sample2_1024_mask.png")

pipeline = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", 
torch_dtype=torch.float16,
variant="fp16",
use_safetensors=True, 
add_watermarker=False
).to("cuda") # variant="fp16"

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-refiner-1.0",
torch_dtype=torch.float16, 
variant="fp16",
use_safetensors=True,
add_watermarker=False
).to("cuda")


def generate(prompt: str, seed: int, steps: int):

    neg_prompt = "3d, illustration, painting, stylized, 2d, vector, overlap, rendering, render, watermark, text"
    inf_steps = steps

    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipeline(prompt=prompt, 
                    negative_prompt=neg_prompt,
                    generator=generator, 
                    strength=.65, 
                    guidance_scale=10.5, # keep under 15
                    num_inference_steps=inf_steps,
                    ).images[0] # refining noise
    """
    image=init_image, 
    mask_image=init_mask,
    """

    refine_prompt = prompt + "bloom, flare, 8k, dslr, depth of field, high detail, detailed"
    image = refiner(prompt=refine_prompt,
                    image=image,
                    negative_prompt=neg_prompt,
                    num_inference_steps=inf_steps).images[0]

    save_name = f"./generated/{round(time.time())}.png"
    image.save(save_name)

    #upscale.create(prompt=refine_prompt, image_name=save_name)