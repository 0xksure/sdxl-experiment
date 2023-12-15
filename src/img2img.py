import torch
from diffusers import StableDiffusionXLImg2ImgPipeline,AutoPipelineForImage2Image
from diffusers.utils import load_image

url = "./dfod.jpg"
image = load_image(url).convert("RGB").resize((512, 512))

# pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float8, variant="fp16", use_safetensors=True
# )
# pipeline = pipe.to("cuda")
# pipeline.enable_vae_slicing()

# prompt = "Dog in a trippy void, same drawing style, high quality, 8k"
# negative_prompt = "Low quality, blurry image."
# image = pipe(prompt, negative_prompt, strength=0.8,guidance_scale=8.0, image=init_image).images[0]
# image.save("dfod_1.png")

# kadinsky 
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
)
pipeline.to("cuda")

image2image = pipeline("Trippy version of dog, same drawing style, on black background", image).images[0]
rnd = torch.randint(0, 100000, (1,)).item()
image2image.save("dfod_"+str(rnd)+".png")

# comic
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "ogkalu/Comic-Diffusion", torch_dtype=torch.float16
)
pipeline.to("cuda")

# need to include the token "charliebo artstyle" in the prompt to use this checkpoint
image = pipeline("Trippy version of dog, same drawing style, cartoon drawing as an NFT", image=image).images[0]
rnd = torch.randint(0, 100000, (1,)).item()
image.save("dfod_"+str(rnd)+".png")