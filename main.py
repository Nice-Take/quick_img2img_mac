#import txt2img
#import time
import mask2img

##  --- Below is a sample list of prompts ---
# prompts = [
#     "sleek granite countertop in a modern kitchen",
#     "rustic wooden countertop in a cottage style kitchen",
#     "polished concrete countertop in an industrial style kitchen",
#     "quartz countertop in a minimalist kitchen, bright natural lighting",
#     "butcher block countertop in a Scandinavian style kitchen, soft lighting",
#     "recycled glass countertop in an eco-friendly kitchen, bright daylight",
#     "stainless steel countertop in a professional chef’s kitchen, spotlighting",
#     "terrazzo countertop in a mid-century modern kitchen, gentle morning light",
#     "soapstone countertop in a traditional kitchen with soft under-cabinet lighting",
#     "laminated countertop in a retro-style kitchen, vibrant colors",
#     "ceramic tile countertop in a Mediterranean style kitchen, sunset lighting",
#     "black marble countertop in an ultra-modern kitchen, LED strip lights",
#     "reclaimed wood countertop in a rustic barn kitchen, natural sunlight",
#     "bamboo countertop in a contemporary Asian inspired kitchen, dim overhead lighting",
#     "onyx countertop in a luxury kitchen, evening mood lighting",
#     "solid surface countertop in a sleek, high-tech kitchen, bright white lighting",
#     "limestone countertop in a French provincial kitchen, warm yellow lighting",
#     "blue pearl granite countertop in a sea-themed kitchen, natural coastal light",
#     "engineered stone countertop in a transitional style kitchen, adjustable track lighting",
#     "tiled mosaic countertop in a vibrant Spanish-inspired kitchen, golden hour lighting"
#     ]

print("\n------------------------")
print("[Begin Batch Generation]")
print("------------------------\n")

txt_prompt = input("Text Prompt: ")
original_img = input("Original Image Path: ")
quantity = int(input("Number of Images: "))
seed = int(input("Seed: "))

mask2img.generate_batch(
    txt_prompt,
    original_img,
    quantity,
    seed=seed)

print("\n---------------------")
print("[Generation Complete]")
print("---------------------\n")