# Worker de refino de forma (Hunyuan3D-2.1) do Vertexing (RunPod Serverless, sem volume).
# Só a parte de forma (hy3dshape): é Python puro, sem extensões CUDA pra compilar.
# Sem pesos na imagem: DiT e VAE vêm do cache de modelos do RunPod
# (tencent/Hunyuan3D-2.1).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# libopengl0/libglu: o pymeshlab (importado pelo pacote hy3dshape) não carrega sem elas
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libopengl0 libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# mesmas versões do worker de textura (transformers < 5, diffusers < 0.36, testadas no pod)
COPY paint/constraints.txt /tmp/constraints.txt
RUN pip install --upgrade pip wheel setuptools \
    && pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install -c /tmp/constraints.txt diffusers transformers accelerate einops omegaconf pyyaml tqdm \
       numpy trimesh pymeshlab pygltflib scikit-image opencv-python-headless rembg onnxruntime timm \
       runpod hf_transfer huggingface_hub
# amostragem por ponto mais distante do codificador do VAE (roda de verdade aqui, na CPU)
RUN pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.6.0+cu124.html \n    && python -c "import torch; from torch_cluster import fps; print('fps ok', fps(torch.rand(100, 3), ratio=0.1).shape)"

ARG HUNYUAN21_COMMIT=82920d643c0dc2f7bfd7255f45f62d386edfe60c
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /opt/Hunyuan3D-2.1 \
    && cd /opt/Hunyuan3D-2.1 && git checkout "$HUNYUAN21_COMMIT" \
    # só a forma: o resto (textura PBR, demos) não é usado
    && rm -rf hy3dpaint assets docker gradio_app.py

# o DiT é importado pelo nome no config (hunyuandit -> timm): importa todos os
# modelos aqui, senão a falta de pacote só aparece no worker
RUN cd /opt/Hunyuan3D-2.1/hy3dshape && python -c "import hy3dshape, hy3dshape.models, hy3dshape.models.denoisers.hunyuandit, hy3dshape.models.denoisers.hunyuan3ddit; from hy3dshape.pipelines import retrieve_timesteps; from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader; from hy3dshape.rembg import BackgroundRemover; import runpod; print('imports ok')"

COPY common/vx_stream.py refine/handler.py /app/
CMD ["python", "-u", "/app/handler.py"]
