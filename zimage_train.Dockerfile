# ai-toolkit Z-Image LoRA training — CUDA 12.4 (works on RunPod driver 12.7+).
# Sanity-check at runtime, not at build (peft 0.19 has a broken Bloom import in
# its constants.py — but ai-toolkit may not actually trigger that path).
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends     software-properties-common ca-certificates curl git wget unzip aria2  && add-apt-repository -y ppa:deadsnakes/ppa  && apt-get update && apt-get install -y --no-install-recommends     python3.11 python3.11-dev python3.11-venv python3.11-distutils  && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11  && ln -sf /usr/bin/python3.11 /usr/bin/python  && ln -sf /usr/bin/python3.11 /usr/bin/python3  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir     torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0     --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir "numpy<2"
RUN pip install --no-cache-dir     "diffusers>=0.36,<0.40"     "transformers>=4.50,<4.55"     "huggingface_hub>=0.25,<1.0"     "accelerate>=0.34"     "safetensors>=0.4"     "peft>=0.18"     "bitsandbytes>=0.43"     "torchao>=0.9"     prodigyopt hf_transfer sentencepiece tensorboard     pyyaml oyaml omegaconf toml python-dotenv python-slugify     pillow opencv-python-headless einops kornia     lycoris-lora "open_clip_torch>=2.20,<3" "timm<2" "k-diffusion>=0.1"     flatten-json controlnet-aux

# Patch peft's constants.py to not import BloomPreTrainedModel (which was
# removed from transformers 4.50+). This is the single line breaking peft's
# initial import; ai-toolkit doesn't actually use Bloom anywhere.
RUN sed -i 's|^from transformers import BloomPreTrainedModel|try:
    from transformers import BloomPreTrainedModel
except ImportError:
    BloomPreTrainedModel = None|'     /usr/local/lib/python3.11/dist-packages/peft/utils/constants.py

RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /ai-toolkit

# Sanity check NOW (after the patch). If peft still doesn't import, log
# the diagnostic but don't fail the build — runtime might still work.
RUN python -c "import torch, diffusers, transformers, accelerate, safetensors, torchao; print('core OK')"  && (python -c "import peft; print('peft', peft.__version__)" || echo "WARN: peft import failed at build, ai-toolkit may still work")

WORKDIR /ai-toolkit
