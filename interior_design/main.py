import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline

# Define the device (GPU recommended for faster processing)
device = "cuda:0"

# Load the Stable Diffusion image generation pipeline
# "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to(device)
# URL of the initial image
url = "https://media.discordapp.net/attachments/1003744761804501124/1150912907622756433/pexels-max-rahubovskiy-7031616.jpg?width=1456&height=972"

# Download the image and convert it to RGB format
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image.save("/home/kasra/AI_projects_/simple_diffusion/data/init_image_design.png")
print('image saved')
# Resize the image to a maximum of 768x768 pixels
init_image.thumbnail((768, 768))

# Display the initial image (optional)
prompt = "minimalist interior design (((living room))) with full furnitures: TV, living room Couch, table, lamp, Wall art, pillow, Natural Lighting, Incandescent lamps, Optical Fiber, Capricious Lighting, Ray Tracing Reflections, (((blue and yellow color))) --ar 16:9 --v 5.2"
negative_prompt = ("(((Ugly))), low-resolution, morbid, blurry, cropped, deformed, dehydrated, text, disfigured, duplicate, error, extra arms, extra fingers,"
                   " extra legs, extra limbs, fused fingers, gross proportions, jpeg artifacts, long neck, low resolution, tiling, poorly drawn feet, extra limbs,"
                   " disfigured, body out of frame, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, low quality, lowres,"
                   " low saturation, deformed body features, watermark.")

image_output = pipe(prompt, init_image, strength=0.7, guidance_scale=10, negative_prompt=negative_prompt).images[0]
image_output.save("/home/kasra/AI_projects_/simple_diffusion/data/generated_image_design.png")
print('image saved')