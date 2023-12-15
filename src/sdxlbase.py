import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler,StableDiffusionUpscalePipeline,EulerDiscreteScheduler
from diffusers.utils import load_image
model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

n_steps = 50 
high_noise_frac = 0.8

init_image = load_image("./pp.jpeg").resize((512, 512))


prompt = "Paw patrol on a mission to rescue chicaletta"
negative_prompt = "Low quality, blurry image."
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt, 
    image=init_image,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
).images[0]

model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

prompt = "Paw Patrol on a mission "
negative_prompt="bad, deformed, ugly, bad anotomy"
image = pipe(prompt=prompt, image=image, negative_prompt=n_propmt, strength=0.7).images[0]   
image.save("paw_patrol.png")