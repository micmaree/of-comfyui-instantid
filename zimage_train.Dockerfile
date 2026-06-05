# ai-toolkit Z-Image LoRA training — ALL deps baked at build time so pods do
# ZERO runtime install. Headless dispatch via dockerArgs bash script (no SSH).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends     git wget curl unzip aria2 ca-certificates  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /ai-toolkit
WORKDIR /ai-toolkit

# Keep base image's torch (2.4.0+cu124) — strip torch/torchvision/xformers from
# requirements (only those lines, not substring matches like open-clip-torch).
RUN grep -vE '^(torch|torchvision|torchaudio|xformers)([=<>!~]|$)' requirements_base.txt > /tmp/req.txt  && pip install --no-cache-dir -r /tmp/req.txt  && pip install --no-cache-dir peft prodigyopt bitsandbytes hf_transfer

# Sanity check imports — build fails here if anything is missing.
RUN python -c "import torch, diffusers, transformers, accelerate, peft, bitsandbytes, safetensors; print('torch', torch.__version__, 'ai-toolkit deps OK')"
