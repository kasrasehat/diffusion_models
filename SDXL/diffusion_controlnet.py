from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from diffusers.utils import load_image
import cv2
from PIL import Image
import numpy as np

def canny(image):
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

device = 'cuda:0'
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16).to(device)

image = load_image("/home/kasra/AI_projects_/simple_diffusion/data/generated_image_sdxl_refiner.png")
canny_image = canny(image)
canny_image.save("/home/kasra/AI_projects_/simple_diffusion/data/canny_filter.png")
print('canny filter saved')
canny_image = canny_image
prompt = ["two models, man standing right side, woman standing left side, selfie photo, touching heads in layered street style, standing against a vibrant graffiti wall, Vivid colors, Mirrorless, 28mm lens, f/2.5 aperture, ISO 400, natural daylight."]
generator = [torch.Generator(device="cuda:0").manual_seed(2) for i in range(len(prompt))]

output = pipe(
    prompt,
    canny_image,
    negative_prompt=["out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature, malformed eyes."] * len(prompt),
    generator=generator,
    num_inference_steps=27,
)
#image_grid(output.images, 2, 2)
output.images[0].save("/home/kasra/AI_projects_/simple_diffusion/data/controlnet_generated_canny.png")

# from controlnet_aux import OpenposeDetector,DWposeDetector
# model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
# poses = model(image)
# # model1 = DWposeDetector.from_pretrained("lllyasviel/ControlNet")
# # poses1 = model1(image)
# #image_grid(poses, 2, 2)
# controlnet = ControlNetModel.from_pretrained(
#     "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16
# )
# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     model_id,
#     controlnet=controlnet,
#     torch_dtype=torch.float16,
# )
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()
# pipe.enable_xformers_memory_efficient_attention()






from diffusers import AutoencoderKL, StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image


# Compute openpose conditioning image.
# openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

# image = load_image(
#     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"
# )
# openpose_image = openpose(image)


# specify configs, ckpts and device, or it will be downloaded automatically and use cpu by default
# det_config: ./src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py
# det_ckpt: https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth
# pose_config: ./src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py
# pose_ckpt: https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth
import torch
from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector, DWposeDetector
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dwpose = DWposeDetector(det_config='/home/kasra/AI_projects_/simple_diffusion/DWpose/yolox_l_8xb8-300e_coco.py',
                        det_ckpt=None, pose_config='/home/kasra/AI_projects_/simple_diffusion/DWpose/dwpose-l_384x288.py', pose_ckpt=None, device=device)
processed_image_dwpose = dwpose(image)

# Initialize ControlNet pipeline.
controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# Infer.
prompt = "two models, man standing right side, woman standing left side, selfie photo, touching heads in layered street style, standing against a vibrant graffiti wall, Vivid colors, Mirrorless, 28mm lens, f/2.5 aperture, ISO 400, natural daylight."
negative_prompt = "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature, malformed eyes."
images = pipe(
    prompt, 
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    num_images_per_prompt=1,
    image=processed_image_dwpose.resize((1024, 1024)),
    generator=torch.manual_seed(97),
).images
images[0].save("/home/kasra/AI_projects_/simple_diffusion/data/controlnet_generated_dwpose.png")
print('controlnet_generated_dwpose')