"""Worker de vistas extras (RunPod Serverless, sem volume): a partir da foto da
frente, gera o objeto visto de outros ângulos com Qwen-Image-Edit-2511 + LoRA
de ângulos de câmera (fal, Apache-2.0) + LoRA Lightning (lightx2v,
Apache-2.0). As vistas alimentam o modo "spatial" do worker TRELLIS.2.

Pesos: Qwen/Qwen-Image-Edit-2511 pelo cache de modelos do RunPod (campo
"Cached model"); as duas LoRAs são baixadas na partida (poucos MB/centenas
de MB). Em bf16 o modelo todo ocupa ~58 GB: com GPU de 80 GB fica inteiro na
placa; com menos, usa o offload por componente do diffusers (mais lento e
pede RAM).

Entrada (job["input"]):
  image        PNG/JPG em base64 (a frente)
  views        lista de descritores de ângulo da LoRA, ex. "back view",
               "left side view", "front-right quarter view" (padrão: esquerda,
               costas, direita); vira "<sks> {view} {elevation} {distance}"
  elevation    padrão "eye-level shot";  distance  padrão "medium shot"
  prompts      opcional: lista de prompts completos (substitui views)
  seed         int (padrão 42)
  steps        padrão 8 (Lightning de 8 passos); lightning=false usa 40 e CFG 4
  lora_scale   peso da LoRA de ângulos (padrão 0,9, o recomendado)

Saída (stream): meta + pedaços de um .zip com view_0.png, view_1.png, ...
"""
import base64
import io
import math
import os
import sys
import time
import traceback
import zipfile

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from vx_stream import stream_bytes  # noqa: E402

BASE = "Qwen/Qwen-Image-Edit-2511"
ANGLES = ("fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA", "qwen-image-edit-2511-multiple-angles-lora.safetensors")
LIGHTNING = ("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")
_CACHE_ROOTS = ["/runpod-volume/huggingface-cache/hub", "/runpod/cache/huggingface/hub"]
_PIPE: dict = {}
_LOAD_INFO: dict = {}


def _log(msg: str) -> None:
    print(f"[vertexing {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _snapshot(repo: str) -> tuple[str, str]:
    """Pasta dos pesos: cache de modelos do RunPod ou download do Hugging Face."""
    name = "models--" + repo.replace("/", "--")
    for root in _CACHE_ROOTS:
        snaps = os.path.join(root, name, "snapshots")
        if os.path.isdir(snaps) and os.listdir(snaps):
            return os.path.join(snaps, sorted(os.listdir(snaps))[-1]), f"cache ({root})"
    from huggingface_hub import snapshot_download

    return snapshot_download(repo), "download"


def _pipeline():
    if "qwen" not in _PIPE:
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
        from huggingface_hub import hf_hub_download

        t0 = time.time()
        path, source = _snapshot(BASE)
        # agenda do Lightning (shift 3 da destilação), do exemplo oficial do Qwen-Image-Lightning
        scheduler = FlowMatchEulerDiscreteScheduler.from_config({
            "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False,
            "max_image_seq_len": 8192, "max_shift": math.log(3), "num_train_timesteps": 1000,
            "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False,
            "time_shift_type": "exponential", "use_beta_sigmas": False, "use_dynamic_shifting": True,
            "use_exponential_sigmas": False, "use_karras_sigmas": False,
        })
        pipe = QwenImageEditPlusPipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
        default_scheduler = pipe.scheduler
        pipe.load_lora_weights(hf_hub_download(*ANGLES), adapter_name="angles")
        pipe.load_lora_weights(hf_hub_download(*LIGHTNING), adapter_name="lightning")
        if torch.cuda.get_device_properties(0).total_memory > 70 * 2**30:
            pipe.to("cuda")
            placement = "gpu"
        else:
            pipe.enable_model_cpu_offload()
            placement = "offload"
        _PIPE.update(qwen=pipe, lightning_scheduler=scheduler, default_scheduler=default_scheduler)
        _LOAD_INFO.update(model_source=source, placement=placement, load_s=round(time.time() - t0, 1))
        _log(f"Qwen-Image-Edit-2511 carregado em {time.time() - t0:.1f}s de {source} ({placement})")
    return _PIPE["qwen"]


def _image(b64: str) -> Image.Image:
    """Fundo branco: o Qwen-Image-Edit espera RGB."""
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, img).convert("RGB")


def handler(job):
    inp = job.get("input") or {}
    torch.cuda.reset_peak_memory_stats()
    try:
        pipe = _pipeline()
        image = _image(inp["image"])
        prompts = inp.get("prompts") or [
            f"<sks> {v} {inp.get('elevation', 'eye-level shot')} {inp.get('distance', 'medium shot')}"
            for v in inp.get("views", ["left side view", "back view", "right side view"])
        ]
        lightning = bool(inp.get("lightning", True))
        pipe.set_adapters(["angles", "lightning"] if lightning else ["angles"],
                          adapter_weights=[float(inp.get("lora_scale", 0.9))] + ([1.0] if lightning else []))
        pipe.scheduler = _PIPE["lightning_scheduler" if lightning else "default_scheduler"]
        steps = int(inp.get("steps", 8 if lightning else 40))
        cfg = float(inp.get("true_cfg_scale", 1.0 if lightning else 4.0))
        seed = int(inp.get("seed", 42))

        t0 = time.time()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for i, prompt in enumerate(prompts):
                out = pipe(image=[image], prompt=prompt, negative_prompt=" ", true_cfg_scale=cfg,
                           num_inference_steps=steps, generator=torch.Generator("cuda").manual_seed(seed + i)).images[0]
                png = io.BytesIO()
                out.save(png, format="PNG")
                z.writestr(f"view_{i}.png", png.getvalue())
                _log(f"vista {i} ({prompt}) pronta")
        meta = {
            "format": "zip", "prompts": prompts, "steps": steps, "true_cfg_scale": cfg,
            "timings": {"generate_s": round(time.time() - t0, 1)},
            "gpu": torch.cuda.get_device_name(0),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1), **_LOAD_INFO,
        }
        yield from stream_bytes(buf.getvalue(), meta)
    except Exception as e:  # noqa: BLE001 — o erro volta pro Vertexing mostrar
        traceback.print_exc()
        yield {"type": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import runpod

    _pipeline()  # carrega antes do primeiro job
    runpod.serverless.start({"handler": handler})
