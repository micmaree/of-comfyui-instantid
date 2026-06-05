# ai-toolkit Z-Image LoRA training — minimal deps baked at build time.
# Bypass ai-toolkit's requirements.txt (bleeding-edge transformers 5.x conflicts
# with our torch 2.6 on driver 12.7) — install only what training needs.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends     git wget curl unzip aria2 ca-certificates  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /ai-toolkit
WORKDIR /ai-toolkit

# Pinned versions known to work with torch 2.6 + ai-toolkit Z-Image training.
# diffusers >=0.36 needed for ZImagePipeline. transformers <5 / hub <1 to avoid
# the conflict from the 5.x upstream pin.
RUN pip install --no-cache-dir     "diffusers>=0.36,<0.40"     "transformers>=4.46,<5"     "huggingface_hub>=0.25,<1.0"     "accelerate>=0.34"     "safetensors>=0.4"     "peft>=0.13,<0.16"     "bitsandbytes>=0.43"     "torchao>=0.6,<0.9"     "prodigyopt"     "hf_transfer"     "pyyaml" "oyaml" "omegaconf" "toml" "python-dotenv" "python-slugify"     "pillow" "opencv-python" "albumentations>=1.4" "kornia" "einops"     "lycoris-lora" "open_clip_torch" "timm" "k-diffusion"     "sentencepiece" "tensorboard" "flatten-json" "invisible-watermark"     "controlnet-aux" "matplotlib" "av"

# Build-time sanity check.
RUN python -c "import torch, diffusers, transformers, accelerate, peft, bitsandbytes, safetensors; print('torch', torch.__version__, 'transformers', transformers.__version__, 'diffusers', diffusers.__version__, 'peft', peft.__version__)"
