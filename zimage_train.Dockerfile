# ai-toolkit Z-Image LoRA training — OnlyFans-AI Kohya pattern.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends     git wget curl unzip aria2 ca-certificates  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /ai-toolkit
WORKDIR /ai-toolkit

# Pre-install numpy<2 + scikit-learn pinned to a version with cp311 wheels.
# Otherwise transient deps (clean-fid, controlnet-aux, etc.) pull bleeding-edge
# scikit-learn that has no wheel for our combo and Cython-builds from source.
RUN pip install --no-cache-dir "numpy<2" "scikit-learn==1.5.2" "scipy<1.14"

# Strip from ai-toolkit's requirements:
#   - torch/torchvision/torchaudio/xformers (keep image's torch 2.4.0+cu124)
#   - transformers, huggingface_hub, torchao, peft pins (target torch 2.9)
#   - pytorch_fid, clean-fid (FID metrics, pull sklearn build-from-source)
#   - gradio (UI, 400MB+ useless for headless)
#   - invisible-watermark, pytorch-wavelets, lpips (unused at training time)
#   - controlnet_aux (pulls mediapipe + sklearn — not needed for LoRA training)
# Replace the git diffusers pin with a stable PyPI release.
RUN sed -i '/^transformers==/d; /^huggingface_hub==/d; /^torchao==/d' requirements_base.txt  && sed -i '/^peft==/d; /^pytorch_fid/d; /^clean[-_]fid/d; /^gradio/d' requirements_base.txt  && sed -i '/^invisible-watermark/d; /^pytorch-wavelets/d; /^lpips/d; /^controlnet[-_]aux/d' requirements_base.txt  && sed -i 's|^git+https://github.com/huggingface/diffusers.*|diffusers>=0.36,<0.40|' requirements_base.txt  && grep -vE '^(torch|torchvision|torchaudio|xformers)([=<>!~]|$)' requirements_base.txt     | grep -vE '^\.$|^-e' > /tmp/req.txt  && pip install --no-cache-dir -r /tmp/req.txt  && pip install --no-cache-dir "huggingface_hub<1.0" "peft<0.16" hf_transfer

RUN python -c "import accelerate, transformers, diffusers, safetensors; print('OK')"
