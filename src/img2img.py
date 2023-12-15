import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
url = "./dfod.jpg"

init_image = load_image(url).convert("RGB")
prompt = "Dog in a trippy void, same drawing style, high quality, 8k"
negative_prompt = "Low quality, blurry image."
image = pipe(prompt, negative_prompt, strength=0.8, image=init_image).images[0]
image.save("dfod_1.png")