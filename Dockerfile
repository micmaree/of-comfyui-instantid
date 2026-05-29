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

# --- ADetailer: Impact Pack + Subpack (FaceDetailer + UltralyticsDetectorProvider) ---
# Used for separate FACE and HAND repair passes after the main render.
RUN git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /comfyui/custom_nodes/ComfyUI-Impact-Pack \
 && git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git /comfyui/custom_nodes/ComfyUI-Impact-Subpack \
 && pip install --no-cache-dir ultralytics dill segment-anything \
 && pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-Impact-Pack/requirements.txt \
 && pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-Impact-Subpack/requirements.txt

# --- Detection models for ADetailer (separate face + hand detectors) ---
RUN comfy model download \
      --url "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt" \
      --relative-path models/ultralytics/bbox \
      --filename face_yolov8n.pt \
 && comfy model download \
      --url "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt" \
      --relative-path models/ultralytics/bbox \
      --filename hand_yolov8n.pt
