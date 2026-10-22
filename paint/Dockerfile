# Worker de textura Hunyuan3D-Paint do Vertexing (RunPod Serverless, sem volume).
# Reproduz o ambiente validado no pod em 22/10/2026 (commit e versões fixados).
# Sem pesos na imagem: Paint e Delight vêm do cache de modelos do RunPod
# (tencent/Hunyuan3D-2).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9+PTX" \
    FORCE_CUDA=1 \
    MAX_JOBS=4 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# libopengl0/libglu: o pymeshlab (dependência do Hunyuan) não carrega sem elas
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libopengl0 libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY paint/constraints.txt /tmp/constraints.txt
RUN pip install --upgrade pip wheel setuptools ninja pybind11 \
    && pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

ARG HUNYUAN_COMMIT=f8db63096c8282cb27354314d896feba5ba6ff8a
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /opt/Hunyuan3D-2 \
    && cd /opt/Hunyuan3D-2 && git checkout "$HUNYUAN_COMMIT" \
    # sem a interface web deles e sem torch (já instalado na versão certa);
    # o setup.py do hy3dgen pede gradio, por isso --no-deps (o resto já veio)
    && grep -viE '^(gradio|fastapi|uvicorn|torch|torchvision)\b' requirements.txt | tr -d '\r' > /tmp/req.txt \
    && pip install -c /tmp/constraints.txt -r /tmp/req.txt \
    && pip install -e . --no-build-isolation --no-deps \
    && pip install -c /tmp/constraints.txt runpod hf_transfer huggingface_hub

# Extensões do Paint (rasterizador CUDA + processador de malha)
RUN cd /opt/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && pip install . --no-build-isolation
RUN cd /opt/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer && pip install . --no-build-isolation

# confere que tudo foi instalado sem importar: o runner do GitHub não tem GPU e
# alguns módulos (Triton) procuram o driver já na importação
RUN python -c "import importlib.util as u, sys; m=[x for x in 'torch custom_rasterizer mesh_processor runpod transformers diffusers hy3dgen'.split() if not u.find_spec(x)]; print('faltando:', m) if m else print('módulos ok'); sys.exit(1 if m else 0)"

COPY common/vx_stream.py paint/handler.py /app/
CMD ["python", "-u", "/app/handler.py"]
