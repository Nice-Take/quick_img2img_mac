import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoPipelineForText2Image, StableDiffusionUpscalePipeline
from PIL import Image
from diffusers.utils import load_image, make_image_grid
import time
import upscale


def image_grid(imgs, rows=2, cols=2):                                                                                                                                                                                                         
    w, h = imgs[0].size                                                                                                                                                                                                                       
    grid = Image.new('RGB', size=(cols*w, rows*h))                                                                                                                                                                                            
                                                                                                                                                                                                                                              
    for i, img in enumerate(imgs):                                                                                                                                                                                                            
        grid.paste(img, box=(i%cols*w, i//cols*h))                                                                                                                                                                                            
    return grid                     


def get_inputs(batch_size=1):                                                                                                                                                                                                                 
  generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]                                                                                                                                                             
  prompts = batch_size * [prompt]                                                                                                                                                                                                             
  num_inference_steps = 20                                                                                                                                                                                                                    
  return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}                                                                                                                                              


# ------------------------------------------------------------------------------#
init_image = load_image("./sample2_1024.png") # loading in reference image
init_mask = load_image("./sample2_1024_mask.png")

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

# high_res = StableDiffusionUpscalePipeline.from_pretrained(
#    "stabilityai/stable-diffusion-x4-upscaler",
#    variant="fp16",
#    torch_dtype=torch.float16,
#    use_safetensors=True,
#    add_watermarker=False
# ).to("cuda")

# inpaint model
inpaint = AutoPipelineForText2Image.from_pipe(pipeline).to("cuda")

generator = torch.Generator("cuda").manual_seed(1)

###---------- Typical prompt workflow
#prompt = "vitamin, amber glass, macro, water glass, lamp, night, warm, detailed, 85mm, 8k"#detailed, 85mm, 50mm, dslr, 8
neg_prompt = "3d, illustration, painting, stylized, 2d, vector, overlap"

### --------- Strictly starting from prompt
#image = pipeline(prompt=prompt, negative_prompt=neg_prompt).images[0] # base
#image = refiner(prompt=prompt, image=image).images[0] # refining noise

### --------- Starting from image and adding details ontop
#prompt = "drink table at the beach, umbrella, overhead, hdr, 135mm imax, 8k, detailed"
prompt = "marble coutnertop in a farmhouse style kitchen, tabletop, 8k, 85mm, depth of field, detailed"
inf_steps = 100

image = pipeline(prompt=prompt, 
                 image=init_image, 
                 mask_image=init_mask,
                 negative_prompt=neg_prompt,
                 generator=generator, 
                 strength=.65, 
                 guidance_scale=10.5, # keep under 15
                 num_inference_steps=inf_steps,
                 ).images[0] # refining noise

refine_prompt = prompt + "bloom, flare, 8k, dslr, depth of field, high detail, detailed"
image = refiner(prompt=refine_prompt,
                image=image,
                negative_prompt=neg_prompt,
                num_inference_steps=inf_steps).images[0]

save_name = f"./generated/{round(time.time())}.png"
image.save(save_name)

#upscale it
upscale.create(prompt=refine_prompt, image_name=save_name)