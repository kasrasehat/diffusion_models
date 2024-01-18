from diffusers import VQDiffusionPipeline
import torch

pipe = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq", torch_dtype=torch.float16)
pipe.to("cuda:0")
prompt = "two people real selfie photo in stadium "

image = pipe(prompt).images[0]
image.save("/home/kasra/AI_projects_/simple_diffusion/data/generated_image1.png")