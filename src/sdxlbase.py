import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image
model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

refiner = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

n_steps = 40
high_noise_frac = 0.8

init_image = load_image("./pp.jpeg").resize((512, 512))


prompt = "Cinematic shot of paw patrollers on their way to save chicaletta once more. Make sure that the eyes are photo realistic."
image = pipe(
    prompt=prompt, 
    image=init_image,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images[0]

image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]
    
image.save("astronaut_rides_horse.png")