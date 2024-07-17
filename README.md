<div align="center">
<h1>InstantStyle-Plus: Style Transfer with Content-Preserving in Text-to-Image Generation</h1>

[**Haofan Wang**](https://haofanwang.github.io/)<sup>*</sup> · Peng Xing · Hao Ai · Renyuan Huang · Qixun Wang · Xu Bai

InstantX Team 

<sup>*</sup>corresponding authors

<a href='https://instantstyle-plus.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='[https://arxiv.org/abs/2404.02733](https://arxiv.org/abs/2407.00788)'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
[![GitHub](https://img.shields.io/github/stars/InstantStyle/InstantStyle?style=social)](https://github.com/instantX-research/InstantStyle-Plus)

</div>

InstantStyle-Plus is a pre-experimental project on the top of [InstantStyle](https://github.com/InstantStyle/InstantStyle) and is designed to improve content preservation capabilities in style transfer. We decompose this task into three subtasks: style injection, spatial structure preservation, and semantic content preservation.

<div align="center">
<img src='assets/teaser.png' width = 900 >
</div>

## Release
- [2024/07/01] 🔥 Code and Techincal Report released.

## Download
Our work requires pre-trained checkpoints from [InstantStyle](https://github.com/InstantStyle/InstantStyle), [Tile-ControlNet](https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0), [MistoLine](https://huggingface.co/TheMistoAI/MistoLine) and [CSD](https://github.com/learn2phoenix/CSD).

```
# download adapters
huggingface-cli download --resume-download h94/IP-Adapter --local-dir checkpoints/IP-Adapter

# download ControlNets
huggingface-cli download --resume-download TheMistoAI/MistoLine --local-dir checkpoints/MistoLine
huggingface-cli download --resume-download xinsir/controlnet-tile-sdxl-1.0 --local-dir checkpoints/controlnet-tile-sdxl-1.0

# follow https://github.com/haofanwang/CSD_Score?tab=readme-ov-file#download to download CSD models
git clone https://github.com/haofanwang/CSD_Score
```

Set [HF_ENDPOINT](https://hf-mirror.com/) in case you cannot access HuggingFace.

## Usage
```
python infer_style.py
```

## Resources
- [InstantStyle](https://github.com/InstantStyle/InstantStyle)
- [InstantID](https://github.com/InstantID/InstantID)

## Disclaimer
Users are granted the freedom to create images using this tool, but they are obligated to comply with local laws and utilize it responsibly. The developers will not assume any responsibility for potential misuse by users.

## Acknowledgements
InstantStyle-Plus is developed by InstantX Team. Our work is built on [ReNoise-Inversion](https://github.com/garibida/ReNoise-Inversion) and [CSD](https://github.com/learn2phoenix/CSD).

## Cite
If you find InstantStyle-Plus useful for your research and applications, please cite us using following BibTeX:

```bibtex
@article{wang2024instantstyle,
  title={InstantStyle-Plus: Style Transfer with Content-Preserving in Text-to-Image Generation},
  author={Wang, Haofan and Xing, Peng and Huang, Renyuan and Ai, Hao and Wang, Qixun and Bai, Xu},
  journal={arXiv preprint arXiv:2407.00788},
  year={2024}
}

@article{wang2024instantstyle,
  title={Instantstyle: Free lunch towards style-preserving in text-to-image generation},
  author={Wang, Haofan and Wang, Qixun and Bai, Xu and Qin, Zekui and Chen, Anthony},
  journal={arXiv preprint arXiv:2404.02733},
  year={2024}
}
```

For any question, feel free to contact us via haofanwang.ai@gmail.com.
