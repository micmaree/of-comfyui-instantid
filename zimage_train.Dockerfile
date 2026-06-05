# ai-toolkit Z-Image LoRA training — MINIMAL deps approach.
# Skips ai-toolkit's requirements_base.txt entirely (it pulls bleeding-edge
# torchao 0.10, transformers 5.x, huggingface_hub 1.10 that conflict with
# torch 2.4 base + sklearn build-from-source issues). Hand-picked minimal
# stack instead — only what Z-Image LoRA training actually imports.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends     git wget curl unzip aria2 ca-certificates  && rm -rf /var/lib/apt/lists/*

# Pin numpy + scipy + sklearn FIRST with known-good wheels so any transient
# resolver decision doesn't trigger a Cython source build.
RUN pip install --no-cache-dir "numpy<2" "scipy<1.14" "scikit-learn==1.5.2"

# Minimal training stack. NO requirements.txt parsing. Each package picked
# because ai-toolkit's Z-Image training code path actually imports it.
RUN pip install --no-cache-dir     "diffusers>=0.36,<0.40"     "transformers>=4.46,<5"     "accelerate>=0.34"     "huggingface_hub<1.0"     "peft<0.16"     "bitsandbytes>=0.43"     "safetensors"     "pillow" "pyyaml" "oyaml" "omegaconf" "toml"     "python-dotenv" "python-slugify" "flatten-json"     "einops" "kornia" "sentencepiece" "tensorboard"     "prodigyopt" "hf_transfer"     "lycoris-lora" "open_clip_torch" "timm" "k-diffusion"

RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /ai-toolkit
WORKDIR /ai-toolkit

# Smoke test core imports.
RUN python -c "import accelerate, transformers, diffusers, peft, safetensors, bitsandbytes; print('OK')"
