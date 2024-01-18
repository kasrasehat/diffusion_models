from diffusers import DiffusionPipeline
import torch

# CompVis/stable-diffusion-v1-4
# stabilityai/stable-diffusion-xl-base-1.0
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda:0")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "selfie real photo of two people in new camp stadium"

images = pipe(prompt=prompt).images[0]
images.save("/home/kasra/AI_projects_/simple_diffusion/data/generated_image1.png")