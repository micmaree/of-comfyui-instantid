# ai-toolkit Z-Image LoRA training — ALL deps baked at build time.
# Official PyTorch image because RunPod 4090/A6000 pods have NVIDIA driver
# 12.7 — cu128 wheels need driver 12.8+, so we stay on cu124 + torch 2.6.0
# (highest cu124 wheel available on pytorch.org).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends     git wget curl unzip aria2 ca-certificates  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /ai-toolkit
WORKDIR /ai-toolkit

# Strip torch/torchvision/torchaudio/xformers from ai-toolkit's requirements.
# Also pin transformers + torchao to versions compatible with torch 2.6
# (newer ones in upstream requirements need torch 2.9+).
RUN sed -i 's/^transformers==.*/transformers>=4.46,<5/' requirements_base.txt  && sed -i 's/^torchao==.*/torchao>=0.6,<0.9/'        requirements_base.txt  && grep -vE '^(torch|torchvision|torchaudio|xformers)([=<>!~]|$)' requirements_base.txt > /tmp/req.txt  && pip install --no-cache-dir -r /tmp/req.txt  && pip install --no-cache-dir peft prodigyopt bitsandbytes hf_transfer

# Sanity check imports — fails BUILD here on missing deps, not first pod spawn.
RUN python -c "import torch, diffusers, transformers, accelerate, peft, bitsandbytes, safetensors; print('torch', torch.__version__, 'transformers', transformers.__version__)"
