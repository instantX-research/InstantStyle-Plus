<div align="center">
<h1>InstantStyle-Plus: Style Transfer with Content-Preserving in Text-to-Image Generation</h1>

[**Haofan Wang**](https://haofanwang.github.io/)<sup>*</sup> · Peng Xing · Hao Ai · Renyuan Huang · Qixun Wang · Xu Bai

InstantX Team 

<sup>*</sup>corresponding authors

<a href='https://instantstyle-plus.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2404.02733'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-red)](https://huggingface.co/spaces/InstantX/InstantStyle)
[![ModelScope](https://img.shields.io/badge/ModelScope-Studios-blue)](https://modelscope.cn/studios/instantx/InstantStyle/summary)
[![GitHub](https://img.shields.io/github/stars/InstantStyle/InstantStyle?style=social)](https://github.com/InstantStyle/InstantStyle)

</div>

InstantStyle-Plus is the successor to [InstantStyle](https://github.com/InstantStyle/InstantStyle) and is designed to improve content preservation capabilities in style transfer. We decompose this task into three subparts: (1) Spatial Structure Content; (2) Semantic Content; (3) Style Content.

<div align="center">
<img src='assets/teaser.png' width = 900 >
</div>

### Key Features:
- Init Content Latent + ControlNet for Spatial Structure Content;
- Global IP-Adapter for Semantic Content;
- InstantStyle + Style Guidance for Style Content.

## Release
- [2024/06/12] 🔥 Code released.

## Download
Our work requires pre-trained checkpoints from [InstantStyle](https://github.com/InstantStyle/InstantStyle), [MistoLine](https://huggingface.co/TheMistoAI/MistoLine) and [CSD](https://github.com/learn2phoenix/CSD).

```
# download adapters
huggingface-cli download --resume-download h94/IP-Adapter --local-dir checkpoints/IP-Adapter

# download ControlNets
huggingface-cli download --resume-download TheMistoAI/MistoLine --local-dir checkpoints/MistoLine

# follow https://github.com/haofanwang/CSD-Score?tab=readme-ov-file#download to download CSD models
git clone https://github.com/haofanwang/CSD-Score
```

Set [HF_ENDPOINT](https://hf-mirror.com/) in case you cannot access HuggingFace.

## Usage

```python
from infer_style import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# image dirs
style_image_dir = "./data/style/103.jpg"
style_image = Image.open(style_image_dir).resize((512, 512))

content_image_dir = "./data/content/11.jpg"
content_image_prompt = "paris"
content_image = Image.open(content_image_dir).convert("RGB").resize((1024, 1024))

# init style clip model
clip_model = CSD_CLIP("vit_large", "default")
model_path = "CSD_Score/models/checkpoint.pth"
checkpoint = torch.load(model_path, map_location="cpu")
state_dict = convert_state_dict(checkpoint['model_state_dict'])
clip_model.load_state_dict(state_dict, strict=False)
clip_model = clip_model.to(device)

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
preprocess = transforms.Compose([
                transforms.Resize(size=224, interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

# computer style embedding
style_image = preprocess(Image.open(style_image_dir)).unsqueeze(0).to(device) # torch.Size([1, 3, 224, 224])
with torch.no_grad():
    _, content_output, style_output = clip_model(style_image)

# load ControlNet
controlnet_path = "TheMistoAI/MistoLine"
controlnet = ControlNetModel.from_pretrained(
    controlnet_path,
    torch_dtype=torch.float16,
    variant="fp16",
).to(device)

# load pipeline
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16
).to(device)

pipe_inference = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
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
    ["h94/IP-Adapter", 
     "h94/IP-Adapter", 
    ],
    subfolder=["sdxl_models", "sdxl_models"],
    weight_name=[
        "ip-adapter_sdxl_vit-h.safetensors",
        "ip-adapter_sdxl_vit-h.safetensors",
    ],
    image_encoder_folder=None,
)
```

### Step1: apply content inversion for spatial structure

```python
model_type = Model_Type.SDXL
scheduler_type = Scheduler_Type.DDIM
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device, model_name="stabilityai/stable-diffusion-xl-base-1.0")

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

# delete to save memory
del pipe_inversion, pipe_inference, all_latents, content_output
torch.cuda.empty_cache()
```

### Step2: apply ControlNet for spatial structure

```python
# condition image
input_image_cv2 = cv2.imread(content_image_dir)
input_image_cv2 = np.array(input_image_cv2)
input_image_cv2 = cv2.Canny(input_image_cv2, 100, 200)
input_image_cv2 = input_image_cv2[:, :, None]
input_image_cv2 = np.concatenate([input_image_cv2, input_image_cv2, input_image_cv2], axis=2)
anyline_image = Image.fromarray(input_image_cv2).resize((1024, 1024))
```

### Step3: apply IP-Adapter and InstantStyle for semantic content and style

```python
scale_global = 0.3 # high semantic content decrease style effect
scale_style = {
    "up": {"block_0": [0.0, 1.0, 0.0]},
}
pipe_inference.set_ip_adapter_scale([scale_global, scale_style])
```

### Step4: apply style guidance to preserve style

```python
# infer
images = pipe_inference(
    prompt=content_image_prompt, # prompt used for inversion
    negative_prompt="lowres, low quality, worst quality, deformed, noisy, blurry",
    ip_adapter_image=[content_image, style_image], # IPA for semantic content, InstantStyle for style
    guidance_scale=5, # high cfg increase style
    num_inference_steps=config.num_inference_steps, # config.num_inference_steps achieves the best
    image=inv_latent, # init content latent
    control_image=anyline_image, # ControlNet for spatial structure
    controlnet_conditioning_scale=0.6, # high control cond decrease style
    denoising_start=0.0001,
    style_embeddings_clip=style_output, # style guidance embedding
    style_guidance_scale=0, # enable style_guidance when style_guidance_scale > 0, cost high RAM because VAE has to be in float32
).images

# computer style similarity score
generated_image = preprocess(images[0]).unsqueeze(0).to(device)
_, content_output1, style_output1 = clip_model(generated_image)

sim = (style_output@style_output1.T).detach().cpu().numpy().mean()
print(sim)
images[0].save(f"result_{sim}.jpg")
```

## Resources
- [InstantStyle](https://github.com/InstantStyle/InstantStyle)
- [InstantID](https://github.com/InstantID/InstantID)

## Disclaimer
Users are granted the freedom to create images using this tool, but they are obligated to comply with local laws and utilize it responsibly. The developers will not assume any responsibility for potential misuse by users.

## Acknowledgements
InstantStyle-Plus is developed by InstantX Team. Our work is built on [ReNoise-Inversion](https://github.com/garibida/ReNoise-Inversion) and [CSD](https://github.com/learn2phoenix/CSD).

## Cite
If you find InstantStyle-Plus useful for your research and applications, please cite us using this BibTeX:

```bibtex
@article{wang2024instantstyle,
  title={Instantstyle: Free lunch towards style-preserving in text-to-image generation},
  author={Wang, Haofan and Wang, Qixun and Bai, Xu and Qin, Zekui and Chen, Anthony},
  journal={arXiv preprint arXiv:2404.02733},
  year={2024}
}
```

For any question, feel free to contact us via haofanwang.ai@gmail.com.
