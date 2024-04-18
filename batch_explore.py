import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoPipelineForText2Image, StableDiffusionUpscalePipeline
from PIL import Image
from diffusers.utils import load_image, make_image_grid
import time


def batch(mainPrompt: str, seed: int) -> None:

    generator = torch.Generator("cuda").manual_seed(seed) # LOCK seed to 0 if iterating on single concept
    #denoise_handoff = 0.85

    neg_prompt = "3d, illustration, painting, stylized, 2d, vector, overlap, alcohol, whiskey"

    prompt = mainPrompt
    image = pipeline(prompt=prompt, 
                    negative_prompt=neg_prompt,
                    generator=generator, 
                    strength=.65, 
                    guidance_scale=8.5, # keep under 15
                    num_inference_steps=50,
                    ).images[0] # refining noise

    refine_prompt = "detailed, hdr, bloom, flare, 8k, dslr, 135mm imax, 85mm"
    image = refiner(prompt=refine_prompt,
                    image=image,
                    num_inference_steps=200,
                    negative_prompt=neg_prompt).images[0]

    image.save(f'./generated/{round(time.time())}.png')


#------------------- Setup Models ---------------------------------
pipeline = StableDiffusionXLPipeline.from_pretrained(
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True, 
    add_watermarker=False
    ).to("cuda") # variant="fp16"

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16, 
    variant="fp16",
    use_safetensors=True,
    add_watermarker=False
    ).to("cuda")

# high_res = StableDiffusionUpscalePipeline.from_pretrained(
#    "stabilityai/stable-diffusion-x4-upscaler",
#    variant="fp16",
#    torch_dtype=torch.float16,
#    use_safetensors=True,
#    add_watermarker=False
# ).to("cuda")

# inpaint model
#inpaint = AutoPipelineForText2Image.from_pipe(pipeline).to("cuda")


#--------------------- Create Loop -----------------------------
# prompts = [
#     "glass of water on a kitchen island countertop, 8k, 85mm, highly detailed"
#     "empty wooden table surface, 3/4 view, tabletop",
#     "desk with an empty space for a drink on a coaster, macro, counter",
#     "office with an empty space near a computer, close up",
#     "beach retaining wall in foreground with a stack of rocks framing the image, close up",
#     "nightstand with an empty spot to place a book, 3/4 view",
#     "soft pillowcase, lookdown, empty place for a product",
# ]
prompts = [
    "a drink coaster on an empty table, 8k, 85mm, highly detailed",
    "a drink coaster on an empty table, 8k, 85mm, highly detailed",
    "a drink coaster on an empty table, 8k, 85mm, highly detailed",
    "a drink coaster on an empty table, 8k, 85mm, highly detailed",
]

i = 0
num_images = len(prompts)
while i < num_images:
    batch(prompts[i], i) 
    i += 1