# Worker TRELLIS.2 do Vertexing (RunPod Serverless, sem volume de rede).
# Reproduz o ambiente validado no pod em 21/10/2026: mesmos commits, mesmas
# versões (constraints.txt). Sem pesos na imagem: o TRELLIS.2-4B vem do cache de
# modelos do RunPod; DINOv3 (espelho público) e BiRefNet são baixados na partida.
#
# Base da RunPod: as máquinas deles costumam ter essas camadas em cache, o que
# acelera a primeira partida em cada máquina.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # GPUs do endpoint: 8.6 (A5000/3090/A4000) e 8.9 (L4/4090/RTX 2000 Ada).
    # Sem GPU no build: FORCE_CUDA garante que as extensões compilem a parte CUDA.
    TORCH_CUDA_ARCH_LIST="8.6;8.9+PTX" \
    FORCE_CUDA=1 \
    MAX_JOBS=4 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY trellis2/constraints.txt /tmp/constraints.txt
RUN pip install --upgrade pip wheel setuptools ninja \
    && pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install -c /tmp/constraints.txt hf_transfer runpod huggingface_hub imageio imageio-ffmpeg tqdm \
       easydict opencv-python-headless trimesh transformers pandas zstandard kornia timm einops \
    # atenção esparsa do TRELLIS.2 (sdpa não é aceito lá); --no-deps: não troca o torch
    && pip install xformers==0.0.29.post3 --no-deps --index-url https://download.pytorch.org/whl/cu124

ARG TRELLIS_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd
RUN git clone https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS.2 \
    && cd /opt/TRELLIS.2 && git checkout "$TRELLIS_COMMIT" && (for i in 1 2 3 4 5 6; do git submodule update --init --recursive && break; echo "submódulos: tentativa $i falhou"; sleep $((i*30)); done && ! git submodule status --recursive | grep -q "^-")

# Extensões CUDA (a parte demorada do build). Os submódulos têm novas tentativas:
# o Eigen vem do GitLab, que às vezes recusa por sobrecarga.
ARG NVDIFFRAST_COMMIT=253ac4fcea7de5f396371124af597e6cc957bfae
RUN git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast \
    && cd /tmp/nvdiffrast && git checkout "$NVDIFFRAST_COMMIT" \
    && pip install . --no-build-isolation && rm -rf /tmp/nvdiffrast
ARG CUMESH_COMMIT=12289e1062f0603f2f0d0771b02e1395d247f26f
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh \
    && cd /tmp/CuMesh && git checkout "$CUMESH_COMMIT" && (for i in 1 2 3 4 5 6; do git submodule update --init --recursive && break; echo "submódulos: tentativa $i falhou"; sleep $((i*30)); done && ! git submodule status --recursive | grep -q "^-") \
    && pip install . --no-build-isolation && rm -rf /tmp/CuMesh
ARG FLEXGEMM_COMMIT=6dd94a859c26ee8246888502eada3dd8ad85532e
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM \
    && cd /tmp/FlexGEMM && git checkout "$FLEXGEMM_COMMIT" && (for i in 1 2 3 4 5 6; do git submodule update --init --recursive && break; echo "submódulos: tentativa $i falhou"; sleep $((i*30)); done && ! git submodule status --recursive | grep -q "^-") \
    && pip install . --no-build-isolation && rm -rf /tmp/FlexGEMM
RUN pip install /opt/TRELLIS.2/o-voxel --no-build-isolation

# confere na própria imagem que tudo importa (sem GPU: só o import)
# confere que tudo foi instalado sem importar: o runner do GitHub não tem GPU e
# alguns módulos (Triton) procuram o driver já na importação
RUN python -c "import importlib.util as u, sys; m=[x for x in 'torch cumesh flex_gemm o_voxel nvdiffrast xformers runpod transformers'.split() if not u.find_spec(x)]; print('faltando:', m) if m else print('módulos ok'); sys.exit(1 if m else 0)"

COPY common/vx_stream.py trellis2/handler.py /app/
CMD ["python", "-u", "/app/handler.py"]
