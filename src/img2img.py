import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
url = "./dfod.jpg"

init_image = load_image(url).convert("RGB")
prompt = "Dog with blue skin on a pink background. The dog should be looking at the camera. The image should be trippy"
negative_prompt = "Low quality, blurry image."
image = pipe(prompt, negative_prompt, image=init_image).images[0]
image.save("dfod_1.png")