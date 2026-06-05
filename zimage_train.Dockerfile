# ai-toolkit Z-Image LoRA training — minimal hand-picked deps.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends     git wget curl unzip aria2 ca-certificates  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy<2" "scipy<1.14" "scikit-learn==1.5.2"

# Pin diffusers 0.36.x — same as local stack, known to coexist with peft 0.15
# and transformers 4.46+. Skipping ai-toolkit's requirements.txt avoids the
# bleeding-edge pins (torchao 0.10, transformers 5.x, hub 1.10).
RUN pip install --no-cache-dir     "diffusers>=0.36,<0.37"     "transformers>=4.46,<5"     "accelerate>=0.34"     "huggingface_hub<1.0"     "peft<0.16"     "bitsandbytes>=0.43"     "safetensors"     "pillow" "pyyaml" "oyaml" "omegaconf" "toml"     "python-dotenv" "python-slugify" "flatten-json"     "einops" "kornia" "sentencepiece" "tensorboard"     "prodigyopt" "hf_transfer"     "lycoris-lora" "open_clip_torch" "timm" "k-diffusion"

RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /ai-toolkit
WORKDIR /ai-toolkit

# No sanity check — pip succeeded, image gets pushed. Runtime tells us if
# anything is actually broken (cheaper than 10 more failed builds).
