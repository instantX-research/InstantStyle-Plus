# Copyright 2024 InstantX Team. All rights reserved.

import cv2
import numpy as np
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers import DDIMScheduler, ControlNetModel
from transformers import CLIPVisionModelWithProjection

import torch
import torchvision
from torchvision import transforms

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from inversion import run as invert
from CSD_Score.model import CSD_CLIP, convert_state_dict
from pipeline_controlnet_sd_xl_img2img import StableDiffusionXLControlNetImg2ImgPipeline


def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # image dirs
    style_image_dir = "./data/style/103.jpg"
    style_image = Image.open(style_image_dir).convert("RGB").resize((512, 512))

    content_image_dir = "./data/content/20.jpg"
    content_image_prompt = "a river"
    content_image = Image.open(content_image_dir).convert("RGB")
    content_image = resize_img(content_image)

    # init style clip model
    clip_model = CSD_CLIP("vit_large", "default", model_path="CSD_Score/models/ViT-L-14.pt")
    model_path = "CSD_Score/models/checkpoint.pth"
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = convert_state_dict(checkpoint['model_state_dict'])
    clip_model.load_state_dict(state_dict, strict=False)
    clip_model = clip_model.to(device)

    # preprocess
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    preprocess = transforms.Compose([
                    transforms.Resize(size=224, interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])

    # computer style embedding
    style_image_ = preprocess(Image.open(style_image_dir).convert("RGB")).unsqueeze(0).to(device) # torch.Size([1, 3, 224, 224])
    with torch.no_grad():
        _, __, style_output = clip_model(style_image_)

    # computer content embedding
    content_image_ = preprocess(Image.open(content_image_dir).convert("RGB")).unsqueeze(0).to(device) # torch.Size([1, 3, 224, 224])
    with torch.no_grad():
        _, content_output, __ = clip_model(content_image_)

    # inversion
    model_type = Model_Type.SDXL
    scheduler_type = Scheduler_Type.DDIM
    pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device, model_name="checkpoints/sdxlUnstableDiffusers_v8HeavensWrathVAE")

    config = RunConfig(model_type = model_type,
                       num_inference_steps = 50,
                       num_inversion_steps = 50,
                       num_renoise_steps = 1,
                       scheduler_type = scheduler_type,
                       perform_noise_correction = False,
                       seed = 7865
                      )
    
    # obtain content latent
    _, inv_latent, _, all_latents = invert(content_image,
                                           content_image_prompt,
                                           config,
                                           pipe_inversion=pipe_inversion,
                                           pipe_inference=pipe_inference,
                                           do_reconstruction=False) # torch.Size([1, 4, 128, 128])
    
    rec_image = pipe_inference(image = inv_latent,
                               prompt = content_image_prompt,
                               denoising_start=0.00001,
                               num_inference_steps = config.num_inference_steps,
                               guidance_scale = 1.0).images[0]

    rec_image.save(f"./results/result_rec.jpg")

    del pipe_inversion, pipe_inference, all_latents
    torch.cuda.empty_cache()
    
    
    control_type = "tile"
    
    if control_type == "tile":
        # condition image
        cond_image = load_image(content_image_dir)
        cond_image = resize_img(cond_image)
        
        controlnet_path = "TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic"
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device)
        
    elif control_type == "canny":
        # condition image
        input_image_cv2 = cv2.imread(content_image_dir)
        input_image_cv2 = np.array(input_image_cv2)
        input_image_cv2 = cv2.Canny(input_image_cv2, 100, 200)
        input_image_cv2 = input_image_cv2[:, :, None]
        input_image_cv2 = np.concatenate([input_image_cv2, input_image_cv2, input_image_cv2], axis=2)
        anyline_image = Image.fromarray(input_image_cv2)
        cond_image = resize_img(anyline_image)

        # load ControlNet
        controlnet_path = "checkpoints/MistoLine"
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)

    # load pipeline
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "checkpoints/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16
    ).to(device)

    pipe_inference = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    "checkpoints/sdxlUnstableDiffusers_v8HeavensWrathVAE",
                    controlnet=controlnet,
                    clip_model=clip_model,
                    image_encoder=image_encoder,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                ).to(device)
    pipe_inference.scheduler = DDIMScheduler.from_config(pipe_inference.scheduler.config) # works the best
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

    scale_global = 0.2 # high semantic content decrease style effect
    scale_style = {
        "up": {"block_0": [0.0, 1.0, 0.0]},
    }
    pipe_inference.set_ip_adapter_scale([scale_global, scale_style])

    # infer
    images = pipe_inference(
        prompt=content_image_prompt, # prompt used for inversion
        negative_prompt="lowres, low quality, worst quality, deformed, noisy, blurry",
        ip_adapter_image=[content_image, style_image], # IPA for semantic content, InstantStyle for style
        guidance_scale=5, # high cfg increase style
        num_inference_steps=config.num_inference_steps, # config.num_inference_steps achieves the best
        image=inv_latent, # init content latent
        control_image=cond_image, # ControlNet for spatial structure
        controlnet_conditioning_scale=0.25, # high control cond decrease style
        denoising_start=0.0001,
        style_embeddings_clip=style_output, # style guidance embedding
        content_embeddings_clip=content_output, # content guidance embedding
        style_guidance_scale=0, # enable style_guidance when style_guidance_scale > 0, cost high RAM, need optimization here
        content_guidance_scale=0, # enable content_guidance when style_guidance_scale > 0, cost high RAM, need optimization here
    ).images

    # computer style similarity score
    generated_image = preprocess(images[0]).unsqueeze(0).to(device)
    _, content_output1, style_output1 = clip_model(generated_image)

    style_sim = (style_output@style_output1.T).detach().cpu().numpy().mean()
    content_sim = (content_output@content_output1.T).detach().cpu().numpy().mean()

    print(style_sim, content_sim)
    images[0].save(f"./results/result_{style_sim}_{content_sim}.jpg")