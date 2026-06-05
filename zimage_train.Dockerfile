# ai-toolkit Z-Image LoRA training — mirrors of-kohya-train pattern.
# Base image ships Python 3.11 + torch 2.4.0+cu124 + base ML deps. Works on
# any RunPod GPU host with NVIDIA driver >= 12.4 (covers all current hosts).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends     git wget curl unzip aria2 ca-certificates  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /ai-toolkit
WORKDIR /ai-toolkit

# Same approach as of-kohya-train: strip torch/torchvision/torchaudio/xformers
# (keep image's). Also strip the bleeding-edge pins (transformers 5.x, hub
# 1.10, torchao 0.10) that target torch 2.9+cu128 — incompatible with base.
# Replace the git diffusers pin with a stable PyPI version.
RUN sed -i '/^transformers==/d' requirements_base.txt  && sed -i '/^huggingface_hub==/d' requirements_base.txt  && sed -i '/^torchao==/d' requirements_base.txt  && sed -i '/^peft==/d' requirements_base.txt  && sed -i 's|^git+https://github.com/huggingface/diffusers.*|diffusers>=0.36,<0.40|' requirements_base.txt  && grep -vE '^(torch|torchvision|torchaudio|xformers)([=<>!~]|$)' requirements_base.txt     | grep -vE '^\.$|^-e' > /tmp/req.txt  && pip install --no-cache-dir -r /tmp/req.txt  && pip install --no-cache-dir "huggingface_hub<1.0" "peft<0.16" hf_transfer

# Same sanity check style as of-kohya-train — just verify core imports work.
RUN python -c "import accelerate, transformers, diffusers, safetensors; print('OK')"
