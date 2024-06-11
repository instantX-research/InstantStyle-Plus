import torch
from PIL import Image

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from inversion import run as invert

import cv2
import numpy as np

import diffusers
from controlnet_aux import AnylineDetector

from transformers import CLIPVisionModelWithProjection
from diffusers import DDIMScheduler, StableDiffusionXLControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler

from pipeline_controlnet_sd_xl_img2img import StableDiffusionXLControlNetImg2ImgPipeline

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision

from CSD_Score.model import CSD_CLIP, convert_state_dict

from PIL import Image

# init clip model
clip_model = CSD_CLIP("vit_large", "default")
model_path = "CSD_Score/models/checkpoint.pth"
checkpoint = torch.load(model_path, map_location="cpu")
state_dict = convert_state_dict(checkpoint['model_state_dict'])
clip_model.load_state_dict(state_dict, strict=False)
clip_model = clip_model.cuda()
# clip_model.half()

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
preprocess = transforms.Compose([
                transforms.Resize(size=224, interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

# style image
image = preprocess(Image.open("4.jpg")).unsqueeze(0).to("cuda") # torch.Size([1, 3, 224, 224])
_, content_output, style_output = clip_model(image)

# clip_model.half()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_type = Model_Type.SDXL
scheduler_type = Scheduler_Type.DDIM
# pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

config = RunConfig(model_type = model_type,
                   num_inference_steps = 50,
                   num_inversion_steps = 50,
                   num_renoise_steps = 1,
                   scheduler_type = scheduler_type,
                   perform_noise_correction = False,
                   seed = 7865
                  )

# line detector
# controlnet_path = "/share2/wanghaofan/workspace/checkpoints/MistoLine/Anyline"
# anyline = AnylineDetector.from_pretrained(
#     controlnet_path, filename="MTEED.pth", subfolder="Anyline"
# ).to(device)


content_image_dir = "bradpitt.png"
prompt = "a man"

# content
input_image = Image.open(content_image_dir).convert("RGB").resize((1024, 1024))

# inv_latent = torch.load("latent_cs.pt")
# inv_latent = torch.load("inv_latent.pt")
# inv_latent = torch.load("inv_latent_style.pt")

# _, inv_latent, _, all_latents = invert(input_image,
#                                        prompt,
#                                        config,
#                                        pipe_inversion=pipe_inversion,
#                                        pipe_inference=pipe_inference,
#                                        do_reconstruction=False) # torch.Size([1, 4, 128, 128])
# # print(inv_latent.shape)
# torch.save(inv_latent, "inv_latent.pt")

inv_latent = torch.load("inv_latent.pt")

# condition image
input_image_cv2 = cv2.imread(content_image_dir)
input_image_cv2 = np.array(input_image_cv2)
input_image_cv2 = cv2.Canny(input_image_cv2, 100, 200)
input_image_cv2 = input_image_cv2[:, :, None]
input_image_cv2 = np.concatenate([input_image_cv2, input_image_cv2, input_image_cv2], axis=2)
anyline_image = Image.fromarray(input_image_cv2).resize((1024, 1024))

# anyline_image.save("anyline_image.jpg")

# load
controlnet_path = "./checkpoints/MistoLine"
controlnet = ControlNetModel.from_pretrained(
    controlnet_path,
    torch_dtype=torch.float16,
    variant="fp16",
).to(device)

# load pipeline
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "./checkpoints/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16
).to(device)

pipe_inference = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                "./checkpoints/sdxlUnstableDiffusers_v8HeavensWrathVAE",
                controlnet=controlnet,
                clip_model=clip_model,
                image_encoder=image_encoder,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to(device)
pipe_inference.scheduler = DDIMScheduler.from_config(pipe_inference.scheduler.config) # works the best
# pipe_inference.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_inference.scheduler.config)
pipe_inference.unet.enable_gradient_checkpointing()

# load multiple IPA
pipe_inference.load_ip_adapter(
    ["checkpoints/IP-Adapter", 
     "checkpoints/IP-Adapter", 
    ],
    subfolder=["sdxl_models", "sdxl_models"],
    weight_name=[
        "ip-adapter_sdxl_vit-h.safetensors",
        "ip-adapter_sdxl_vit-h.safetensors",
    ],
    image_encoder_folder=None,
)

scale_global = 0.3

scale_style = {
    "up": {"block_0": [0.0, 1.0, 0.0]},
}
pipe_inference.set_ip_adapter_scale([scale_global, scale_style])

# style image
# image = "../InstantStyle-Plus/assets/3.jpg"
# image = "../InstantStyle-Plus/assets/4.jpg"
image = "4.jpg"
image = Image.open(image)

images = pipe_inference(
    prompt=prompt, # prompt used for inversion
    negative_prompt="lowres, low quality, worst quality, deformed, noisy, blurry",
    ip_adapter_image=[input_image, image],
    guidance_scale=5, # high cfg increase style
    num_inference_steps=config.num_inference_steps, # config.num_inference_steps achieves the best
    image=inv_latent,
    control_image=anyline_image,
    controlnet_conditioning_scale=0.6, # high control cond decrease style
    denoising_start=0.0001,
    style_embeddings_clip=style_output,
).images

images[0].save("result.jpg")