# ai-toolkit Z-Image LoRA training — ALL deps baked at build time so pods do
# ZERO runtime install. Headless dispatch via dockerArgs (see runpod_train.py).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends     git wget curl unzip aria2 ca-certificates  && rm -rf /var/lib/apt/lists/*

# ai-toolkit official requires torch 2.9.1 (per README). The base image ships
# 2.4 — upgrade to match, cu124 wheels run on driver 12.4+ (every RunPod GPU).
RUN pip install --no-cache-dir --upgrade     torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1     --index-url https://download.pytorch.org/whl/cu124

RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /ai-toolkit
WORKDIR /ai-toolkit

# Strip torch/torchvision/torchaudio/xformers from ai-toolkit's requirements
# so it doesn't fight our pinned versions. Then install the rest + helpers
# (peft for LoRA load, prodigyopt for Prodigy optimizer, bitsandbytes for
# adamw8bit, hf_transfer for parallel HF downloads).
RUN grep -vE '^(torch|torchvision|torchaudio|xformers)([=<>!~]|$)' requirements_base.txt > /tmp/req.txt  && pip install --no-cache-dir -r /tmp/req.txt  && pip install --no-cache-dir peft prodigyopt bitsandbytes hf_transfer

# Build-time sanity check — fail FAST here on missing deps, not at first pod spawn.
RUN python -c "import torch, diffusers, transformers, accelerate, peft, bitsandbytes, safetensors; print('torch', torch.__version__, 'ai-toolkit deps OK')"
