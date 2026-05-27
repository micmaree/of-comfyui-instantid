# Kohya sd-scripts with ALL deps baked in (tested at build time) so training
# pods need ZERO runtime install — just download dataset + accelerate launch.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends git wget curl unzip ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git /sd-scripts
WORKDIR /sd-scripts

# Install Kohya's full requirements EXCEPT torch/xformers (keep the image's torch),
# plus imagesize which sd-scripts imports. Done at build time = reproducible.
# Strip ONLY the torch/torchvision/torchaudio/xformers package lines (keep the
# image's torch) — must NOT strip lines that merely contain 'torch' as a
# substring like diffusers[torch] or pytorch-lightning or open-clip-torch.
RUN grep -vE '^(torch|xformers)' requirements.txt | grep -vE '^\.$|^-e' > /tmp/req.txt \
 && pip install --no-cache-dir -r /tmp/req.txt imagesize

# Import sanity check — fail the BUILD if anything is missing (no `|| true`).
RUN python -c "import imagesize, accelerate, transformers, diffusers, safetensors; import library.train_util"
