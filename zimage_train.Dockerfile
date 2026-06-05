# ai-toolkit Z-Image LoRA training — ALL deps baked at build time.
# Official PyTorch image because RunPod 4090/A6000 pods have NVIDIA driver
# 12.7 — cu128 wheels need driver 12.8+, so we stay on cu124 + torch 2.6.0
# (highest cu124 wheel available on pytorch.org).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends     git wget curl unzip aria2 ca-certificates  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /ai-toolkit
WORKDIR /ai-toolkit

# Cap upstream pins that target torch 2.9 (we have 2.6). Also relax the
# huggingface_hub pin (==1.10.1) which fights transformers<5 (hub<1.0). And
# replace the diffusers git URL with the stable PyPI release that matches.
RUN sed -i 's|^transformers==.*|transformers>=4.46,<5|'        requirements_base.txt  && sed -i 's|^torchao==.*|torchao>=0.6,<0.9|'                 requirements_base.txt  && sed -i 's|^huggingface_hub==.*|huggingface_hub>=0.34,<1.0|' requirements_base.txt  && sed -i 's|^peft==.*|peft>=0.13,<0.16|'                     requirements_base.txt  && sed -i 's|^git+https://github.com/huggingface/diffusers.*|diffusers>=0.36,<0.40|' requirements_base.txt  && grep -vE '^(torch|torchvision|torchaudio|xformers)([=<>!~]|$)' requirements_base.txt > /tmp/req.txt  && cat /tmp/req.txt  && pip install --no-cache-dir -r /tmp/req.txt  && pip install --no-cache-dir prodigyopt bitsandbytes hf_transfer

RUN python -c "import torch, diffusers, transformers, accelerate, peft, bitsandbytes, safetensors; print('torch', torch.__version__, 'transformers', transformers.__version__, 'diffusers', diffusers.__version__)"
