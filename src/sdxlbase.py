import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler,StableDiffusionUpscalePipeline,StableDiffusionSAGPipeline
from diffusers.utils import load_image
model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionSAGPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

n_steps = 50 
high_noise_frac = 0.8
guidance_scale = 7.5
num_images_per_prompt = 1

sag_scale = 1.0

init_image = load_image("./pp.jpeg").resize((512, 512))


prompt = "Paw patrol on a mission to rescue chicaletta"
negative_prompt = "Low quality, blurry image."
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt, 
    image=init_image,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    num_images_per_prompt=num_images_per_prompt, guidance_scale=guidance_scale, sag_scale=sag_scale
).images[0]

model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "Paw Patrol on a mission "
negative_prompt="bad, deformed, ugly, bad anotomy"
image = pipe(prompt=prompt, image=image, negative_prompt=negative_prompt, strength=0.7).images[0]
    
image.save("paw_patrol.png")