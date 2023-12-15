import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image
model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

init_image = load_image("./pp.jpeg").resize((512, 512))


prompt = "Cinematic shot of paw patrollers on their way to save chicaletta once more. Make sure that the eyes are photo realistic."
image = pipe(prompt=prompt, image=init_image).images[0]
    
image.save("astronaut_rides_horse.png")