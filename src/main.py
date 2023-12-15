from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

init_image = load_image("./pp.jpeg").resize((512, 512))

prompt = "Cinematic shot of paw patrollers on their way to save chicaletta once more"

image = pipe(prompt=prompt, image=init_image,num_inference_steps=100, guidance_scale=0.0).images[0]
image.save("test.png")
