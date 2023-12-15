import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
url = "./dfod.jpg"

init_image = load_image(url).convert("RGB")
prompt = "An NFT collection with more colors but in the same style."
image = pipe(prompt, image=init_image).images