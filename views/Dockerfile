# Worker de vistas extras do Vertexing (Qwen-Image-Edit-2511 + LoRAs de ângulo
# e Lightning; RunPod Serverless, sem volume). Sem extensões CUDA para compilar.
# Pesos fora da imagem: Qwen/Qwen-Image-Edit-2511 pelo cache de modelos do
# RunPod; as LoRAs são baixadas na partida.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# diffusers 0.39 (primeira com o zero_cond_t do 2511) / transformers 4.57: rodam no
# torch 2.6 (as mais novas puxam outro torch e huggingface_hub 1.x).
RUN pip install --upgrade pip wheel setuptools \
    && pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install diffusers==0.39.0 transformers==4.57.6 peft==0.18.0 accelerate==1.12.0 \
       tokenizers==0.22.2 huggingface_hub==0.36.2 hf_transfer==0.1.9 safetensors==0.8.0 \
       pillow==12.3.0 runpod==1.12.0 "torch==2.6.0"

# confere que importa (sem GPU no runner)
RUN python -c "from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler; import peft, runpod, torch; assert torch.__version__.startswith('2.6.0'), torch.__version__; print('ok', torch.__version__)"

COPY common/vx_stream.py views/handler.py /app/
CMD ["python", "-u", "/app/handler.py"]
