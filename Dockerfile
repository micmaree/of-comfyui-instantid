# worker-comfyui (RunPod serverless ComfyUI) + InstantID for SDXL face-lock.
# Models are baked into the image so the 15GB network volume (LUSTIFY) stays untouched.
FROM runpod/worker-comfyui:5.1.0-base

# --- InstantID custom node (cubiq) + its python deps ---
RUN git clone --depth 1 https://github.com/cubiq/ComfyUI_InstantID.git /comfyui/custom_nodes/ComfyUI_InstantID \
 && pip install --no-cache-dir insightface onnxruntime-gpu

# --- InstantID models ---
RUN comfy model download \
      --url "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin" \
      --relative-path models/instantid \
      --filename ip-adapter.bin \
 && comfy model download \
      --url "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors" \
      --relative-path models/controlnet \
      --filename instantid_control.safetensors

# --- InsightFace antelopev2 (face detector/embedder InstantID needs) ---
RUN for f in 1k3d68.onnx 2d106det.onnx genderage.onnx glintr100.onnx scrfd_10g_bnkps.onnx ; do \
      comfy model download \
        --url "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/$f" \
        --relative-path models/insightface/models/antelopev2 \
        --filename "$f" ; \
    done
