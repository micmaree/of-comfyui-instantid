# ai-toolkit Z-Image LoRA training — OnlyFans-AI Kohya pattern.
# Base image ships Python 3.11 + torch 2.4.0+cu124 + ML deps. Works on any
# RunPod GPU host with driver >= 12.4.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends     git wget curl unzip aria2 ca-certificates  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /ai-toolkit
WORKDIR /ai-toolkit

# Pre-pin numpy<2 BEFORE any pip install — sklearn 1.5+ builds from source on
# numpy 2.x and triggers a Cython build failure on the GHA runner.
RUN pip install --no-cache-dir "numpy<2"

# Strip from requirements:
#   - torch/torchvision/torchaudio/xformers (keep image's)
#   - bleeding-edge pins (transformers, hub, torchao, peft) — they target
#     torch 2.9+cu128, incompatible with base torch 2.4.
#   - pytorch_fid (FID metric) — pulls sklearn → Cython build hell.
#   - gradio (UI) — 400MB+ and useless for headless training.
#   - invisible-watermark, pytorch-wavelets, lpips — unused at training time.
# Replace the git diffusers pin with a stable PyPI release.
RUN sed -i '/^transformers==/d; /^huggingface_hub==/d; /^torchao==/d' requirements_base.txt  && sed -i '/^peft==/d; /^pytorch_fid/d; /^gradio/d' requirements_base.txt  && sed -i '/^invisible-watermark/d; /^pytorch-wavelets/d; /^lpips/d' requirements_base.txt  && sed -i 's|^git+https://github.com/huggingface/diffusers.*|diffusers>=0.36,<0.40|' requirements_base.txt  && grep -vE '^(torch|torchvision|torchaudio|xformers)([=<>!~]|$)' requirements_base.txt     | grep -vE '^\.$|^-e' > /tmp/req.txt  && pip install --no-cache-dir -r /tmp/req.txt  && pip install --no-cache-dir "huggingface_hub<1.0" "peft<0.16" hf_transfer

# Lightweight sanity check — same as of-kohya-train.
RUN python -c "import accelerate, transformers, diffusers, safetensors; print('OK')"
