# SLIM Flux worker: ComfyUI + PuLID-Flux node ONLY. The big models (flux1-dev-fp8,
# pulid, EVA-CLIP, antelopev2, facexlib, skin LoRA, trained LoRAs) live on the network
# VOLUME → tiny image = fast cold start. Models loaded from /runpod-volume/models.
FROM runpod/worker-comfyui:5.8.5-base

# --- PuLID-Flux custom node (lldacing) + deps ---
RUN git clone --depth 1 https://github.com/lldacing/ComfyUI_PuLID_Flux_ll.git /comfyui/custom_nodes/ComfyUI_PuLID_Flux_ll \
 && pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI_PuLID_Flux_ll/requirements.txt \
 && pip install --no-cache-dir facenet-pytorch --no-deps insightface onnxruntime-gpu
