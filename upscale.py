import requests
from io import BytesIO

import os
import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image

# # Set the GPU ID (e.g., "0" for the first GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# upscale_model = StableDiffusionUpscalePipeline.from_pretrained(
#     "stabilityai/stable-diffusion-x4-upscaler",
#     torch_dtype=torch.float16
#     ).to("cuda")

# def create(image_path_in: str) -> None:
#     image = Image.open(image_path_in)  # Load the generated image
#     #image.show()

#     # Convert image to tensor
#     image_tensor = ToTensor()(image).unsqueeze(0).to("cuda")  # Add batch dimension and send to GPU

#     # Upscale the image
#     with torch.no_grad():
#         upscaled_image = upscale_model(prompt="empty wooden table surface, 3/4 view, tabletop",image=image_tensor)["sample"][0]

#     # Convert back to PIL for saving or displaying
#     upscaled_pil_image = to_pil_image(upscaled_image.squeeze(0))  # Remove batch dimension
#     upscaled_pil_image.save(f"{image_path_in}_HR.png")
#     upscaled_pil_image.show()

# create("./generated/1713310047.png") 

# # load model and scheduler
# model_id = "stabilityai/stable-diffusion-x4-upscaler"
# pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipeline = pipeline.to("cuda")

# image_name = "./generated/1713393883.png"
# low_res_img = Image.open(image_name).convert("RGB")
# low_res_img = low_res_img.resize((512, 512))

# prompt = "marble coutnertop in a farmhouse style kitchen, tabletop, 8k, 85mm, depth of field, detailed"

# upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
# upscaled_image.save(f"{image_name}_HR.png")


def create(prompt: str, image_name: str):
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")

    img = Image.open(image_name).convert("RGB")
    low_res_img = img.resize((512, 512))

    upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
    upscaled_image.save(f"{image_name}_HR_2.png")


#create("vitamin bottle in kitchen", "T:/MG NT Dropbox/Working Jobs/TMP/generated/1713825937.png")