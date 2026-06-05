# ai-toolkit Z-Image LoRA training — CUDA 12.4 base because RunPod EU-RO-1
# 4090 hosts ship NVIDIA driver 12.7 which cannot run cu128 wheels. cu124 +
# torch 2.6.0 is the highest combo that works on driver 12.7+.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends     software-properties-common ca-certificates curl git wget unzip aria2  && add-apt-repository -y ppa:deadsnakes/ppa  && apt-get update && apt-get install -y --no-install-recommends     python3.11 python3.11-dev python3.11-venv python3.11-distutils  && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11  && ln -sf /usr/bin/python3.11 /usr/bin/python  && ln -sf /usr/bin/python3.11 /usr/bin/python3  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# torch first (~2.4GB) so subsequent cache busts don't re-download it.
RUN pip install --no-cache-dir     torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0     --index-url https://download.pytorch.org/whl/cu124

# numpy<2 pinned BEFORE everything else because sklearn 1.5+ + numpy 2.x had
# Cython build failures (build 4 crash). Then full training stack — explicit
# versions known to coexist with torch 2.6 (no requirements.txt to avoid the
# bleeding-edge transformers 5.x / hub 1.10.1 / diffusers-from-git conflicts
# that killed builds 1-5).
RUN pip install --no-cache-dir "numpy<2"
RUN pip install --no-cache-dir     "diffusers>=0.36,<0.40"     "transformers>=4.50,<4.55"     "huggingface_hub>=0.25,<1.0"     "accelerate>=0.34"     "safetensors>=0.4"     "peft>=0.18"     "bitsandbytes>=0.43"     "torchao>=0.6,<0.9"     prodigyopt hf_transfer sentencepiece tensorboard     pyyaml oyaml omegaconf toml python-dotenv python-slugify     pillow opencv-python-headless einops kornia     lycoris-lora "open_clip_torch>=2.20,<3" "timm<2" "k-diffusion>=0.1"     flatten-json controlnet-aux

RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /ai-toolkit

# Sanity check: build fails here if anything is missing.
RUN python -c "import torch, diffusers, transformers, accelerate, peft, bitsandbytes, safetensors, torchao; print('torch', torch.__version__, 'transformers', transformers.__version__, 'diffusers', diffusers.__version__, 'peft', peft.__version__, 'bitsandbytes', bitsandbytes.__version__)"

WORKDIR /ai-toolkit
