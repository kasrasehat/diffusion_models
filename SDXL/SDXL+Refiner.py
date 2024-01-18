from diffusers import DiffusionPipeline, PNDMScheduler, DPMSolverMultistepScheduler
import torch

# Load the base and refiner models
# /home/kasra/AI_projects_/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors
# "stabilityai/stable-diffusion-xl-base-1.0"
base = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
base.to("cuda:0")
# "stabilityai/stable-diffusion-xl-refiner-1.0"
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda:0")

# Define the scheduler and its parameters
scheduler = PNDMScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000) # Example parameters

# Assign the scheduler to the pipelines
# base.set_scheduler(scheduler)
# refiner.set_scheduler(scheduler)

# Define how many steps and what % of steps to be run on each expert (80/20) here
n_steps = 27
high_noise_frac = 0.8

prompt = "two models, man standing right side, woman standing left side, selfie photo, touching heads in layered street style, standing against a vibrant graffiti wall, Vivid colors, Mirrorless, 28mm lens, f/2.5 aperture, ISO 400, natural daylight."
generator = torch.manual_seed(0)
# Run both experts
image = base(
    prompt=prompt,
    negative_prompt='out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature, malformed eyes.',
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
    generator=generator
).images
image = refiner(
    prompt=prompt,
    negative_prompt=" out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature, malformed eyes.",
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
    generator=generator
).images[0]

image.save("/home/kasra/AI_projects_/simple_diffusion/data/generated_image_sdxl_refiner.png")
print('image saved')

