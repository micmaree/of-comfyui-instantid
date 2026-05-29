# Flux worker — models BAKED in the image so the endpoint needs NO network volume →
# region-agnostic (all 31 RunPod datacenters) = max GPU availability (RunPod's own
# recommendation). FlashBoot keeps popular endpoints warm. PuLID-Flux = face-lock.
FROM runpod/worker-comfyui:5.8.5-flux1-dev-fp8

# --- PuLID-Flux custom node (lldacing) + deps ---
RUN git clone --depth 1 https://github.com/lldacing/ComfyUI_PuLID_Flux_ll.git /comfyui/custom_nodes/ComfyUI_PuLID_Flux_ll \
 && pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI_PuLID_Flux_ll/requirements.txt \
 && pip install --no-cache-dir facenet-pytorch --no-deps insightface onnxruntime-gpu

# --- PuLID-Flux model ---
RUN comfy model download \
      --url "https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors" \
      --relative-path models/pulid --filename pulid_flux_v0.9.1.safetensors

# --- EVA02-CLIP (PuLID face encoder) ---
RUN comfy model download \
      --url "https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_L_336_psz14_s6B.pt" \
      --relative-path models/clip --filename EVA02_CLIP_L_336_psz14_s6B.pt

# --- InsightFace antelopev2 ---
RUN for f in 1k3d68.onnx 2d106det.onnx genderage.onnx glintr100.onnx scrfd_10g_bnkps.onnx ; do \
      comfy model download \
        --url "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/$f" \
        --relative-path models/insightface/models/antelopev2 --filename "$f" ; \
    done

# --- facexlib parsing/detection models ---
RUN comfy model download --url "https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth" --relative-path models/facexlib --filename parsing_bisenet.pth \
 && comfy model download --url "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" --relative-path models/facexlib --filename parsing_parsenet.pth \
 && comfy model download --url "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" --relative-path models/facexlib --filename detection_Resnet50_Final.pth

# --- Flux skin/realism LoRA (XLabs) baked → naturalness "patch" ---
RUN comfy model download \
      --url "https://huggingface.co/XLabs-AI/flux-RealismLora/resolve/main/lora.safetensors" \
      --relative-path models/loras --filename flux_realism.safetensors
