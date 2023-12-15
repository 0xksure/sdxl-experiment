from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

prompt = "Cinematic shot of paw patrollers on their way to save chicaletta once more"

image = pipe(prompt=prompt, num_inference_steps=100, guidance_scale=0.0).images[0]
image.save("test.png")
