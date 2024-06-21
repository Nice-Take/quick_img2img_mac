from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import time
import upscale


pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                 use_safetensors=True,
                                                 add_watermarker=False,
                                                 torch_dtype=torch.float16,
                                                 variant="fp16").to("cuda")

def generate_single(prompt: str, original_img: str, strength: float=0.99, seed: int=0) -> str:
    """
    Creates an image from a reference image and a mask image.
    The original image is the prompt and the mask area in black
    is the area that remains unchanged.

    Returns the string filepath that the image was saved to incase
    further operations are desired.

    Adjust the guidance scale to affect the 'creativity' gen level.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    neg_prompt = "3d, illustration, painting, stylized, 2d, vector, overlap, rendering, render, watermark, text"

    # Deriving mask image filename from original file name, mask files should exist as original name + "_mask"
    parsed_name = original_img.split(".")
    mask_img = "./" + parsed_name[1] + "_mask." + parsed_name[2]

    try:
        image = load_image(original_img).resize((1024, 1024))
        mask_image = load_image(mask_img).resize((1024, 1024))
    except:
        raise ValueError("Unable to load image or mask image, check filename(s).")

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

    save_name = f"./generated/{round(time.time())}.png" 
    image.save(save_name)

    return save_name


###########################################################################################


def generate_batch(txt_prompt: str, original_img: str, quantity: int, seed: int=0, upscale: bool=False):
    """
    Creates a batch of images from a single prompt & original image.
    Default seed = 0
    Default upscale = False
    """

    txt_prompt +=  ", bright, bokeh, 8k, 85mm, detailed, uhd, film"
    iters = quantity
    start_time = time.time()

    for i in range(iters):
        generated_name = generate_single(txt_prompt, original_img, strength=0.99, seed=seed)
        if upscale == True:
            upscale.create(prompt=txt_prompt, image_name=generated_name)

    print("\n[  TIME  STATS  ]")
    print(f"| Total: {round(time.time() - start_time)} sec |")
    print(f"| s/Img: {round(round(time.time() - start_time) / iters)} sec |\n")

