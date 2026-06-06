# ai-toolkit Z-Image LoRA training — version combo that ACTUALLY co-exists.
# Resolves a 4-way conflict: diffusers 0.36 requires peft>=0.17, peft 0.17+
# imports BloomPreTrainedModel removed in transformers 4.57, transformers
# 4.50+ imports Int4WeightOnlyConfig from torchao 0.9+.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends     git wget curl unzip aria2 ca-certificates  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy<2" "scipy<1.14" "scikit-learn==1.5.2"

# THE 4-way compatible combo:
#   diffusers 0.36.x (has ZImagePipeline, requires peft>=0.17)
#   peft 0.17.0      (minimum diffusers 0.36 will accept, uses Bloom import)
#   transformers <4.55 (still has BloomPreTrainedModel, needs torchao>=0.9)
#   torchao >=0.9    (has Int4WeightOnlyConfig)
RUN pip install --no-cache-dir     "diffusers>=0.36,<0.37"     "transformers>=4.50,<4.55"     "huggingface_hub>=0.25,<1.0"     "accelerate>=0.34"     "peft==0.17.0"     "bitsandbytes>=0.43"     "torchao>=0.9"     "safetensors"     "pillow" "pyyaml" "oyaml" "omegaconf" "toml"     "python-dotenv" "python-slugify" "flatten-json"     "einops" "kornia" "sentencepiece" "tensorboard"     "prodigyopt" "hf_transfer"     "lycoris-lora" "open_clip_torch" "timm" "k-diffusion"

RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /ai-toolkit
WORKDIR /ai-toolkit

# Sanity check NOW — fail BUILD if imports break.
RUN python -c "import accelerate, transformers, diffusers, peft, safetensors, bitsandbytes, torchao; print(f'torch {__import__(\"torch\").__version__} | diffusers {diffusers.__version__} | transformers {transformers.__version__} | peft {peft.__version__}')"
