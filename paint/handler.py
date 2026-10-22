"""Worker de textura Hunyuan3D-Paint (imagem Docker própria, sem volume).

Recebe uma malha já reduzida (~40k faces, Y pra cima, ~1 unidade de altura —
como o Paint espera) e a imagem de referência; devolve um GLB com textura UV
em pedaços pela própria resposta (vx_stream). Pesos pelo cache de modelos do
RunPod (campo "Cached model" = tencent/Hunyuan3D-2).

Entrada (job["input"]):
  mesh        GLB ou PLY em base64
  mesh_format "glb" (padrão) | "ply"
  image       PNG/JPG em base64 (sem fundo transparente: o fundo é removido aqui)

Saída (stream): {"type":"meta", chunks, bytes, sha256, format:"glb", faces, ...}
e depois {"type":"chunk", i, data}. O Paint preserva posição e escala da malha.

Licença dos pesos: Tencent Hunyuan 3D Community License (proíbe uso na UE,
Reino Unido e Coreia do Sul; acima de 1 mi de usuários/mês exige licença).
"""
import base64
import io
import os
import sys
import time
import traceback

sys.path.insert(0, "/opt/Hunyuan3D-2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402

from vx_stream import stream_bytes  # noqa: E402

REPO = "tencent/Hunyuan3D-2"
SUBFOLDER = "hunyuan3d-paint-v2-0"  # a versão testada (não a "turbo", padrão do código)
_CACHE_ROOTS = ["/runpod-volume/huggingface-cache/hub", "/runpod/cache/huggingface/hub"]
_PIPELINES: dict = {}
_LOAD_INFO: dict = {}


def _log(msg: str) -> None:
    print(f"[vertexing-paint {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _weights_root() -> str:
    """O Hunyuan procura os pesos em $HY3DGEN_MODELS/tencent/Hunyuan3D-2/<subpasta>:
    aponta esse caminho pro snapshot do cache do RunPod (ou baixa só as duas
    subpastas usadas, se o endpoint não tiver o cache)."""
    name = "models--" + REPO.replace("/", "--")
    snap, source = None, "download"
    for root in _CACHE_ROOTS:
        snaps = os.path.join(root, name, "snapshots")
        if os.path.isdir(snaps) and os.listdir(snaps):
            snap, source = os.path.join(snaps, sorted(os.listdir(snaps))[-1]), f"cache ({root})"
            break
    if snap is None:
        from huggingface_hub import snapshot_download

        snap = snapshot_download(REPO, allow_patterns=[f"{SUBFOLDER}/*", "hunyuan3d-delight-v2-0/*"])
    base = "/tmp/hy3dgen"
    os.makedirs(f"{base}/tencent", exist_ok=True)
    link = f"{base}/{REPO}"
    if not os.path.lexists(link):
        os.symlink(snap, link)
    os.environ["HY3DGEN_MODELS"] = base
    _LOAD_INFO["model_source"] = source
    return base


def _paint():
    if "paint" not in _PIPELINES:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        t0 = time.time()
        _weights_root()
        _PIPELINES["paint"] = Hunyuan3DPaintPipeline.from_pretrained(REPO, subfolder=SUBFOLDER)
        _LOAD_INFO["load_s"] = round(time.time() - t0, 1)
        _log(f"Paint carregado em {time.time() - t0:.1f}s de {_LOAD_INFO['model_source']}")
    return _PIPELINES["paint"]


def _image(b64: str) -> Image.Image:
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    if img.mode == "RGBA" and img.getextrema()[3][0] < 250:
        return img  # já tem fundo transparente
    from hy3dgen.rembg import BackgroundRemover

    if "rembg" not in _PIPELINES:
        _PIPELINES["rembg"] = BackgroundRemover()
    return _PIPELINES["rembg"](img.convert("RGB"))


def handler(job):
    """Gerador: o GLB sai em pedaços (vx_stream), lidos pelo /stream."""
    inp = job.get("input") or {}
    timings: dict = {}
    torch.cuda.reset_peak_memory_stats()
    try:
        paint = _paint()
        mesh = trimesh.load(io.BytesIO(base64.b64decode(inp["mesh"])),
                            file_type=inp.get("mesh_format", "glb"), force="mesh")
        image = _image(inp["image"])
        t0 = time.time()
        textured = paint(mesh, image=image)
        timings["paint_s"] = round(time.time() - t0, 1)
        meta = {
            "format": "glb", "faces": int(len(textured.faces)), "timings": timings,
            "gpu": torch.cuda.get_device_name(0),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            **_LOAD_INFO,
        }
        yield from stream_bytes(textured.export(file_type="glb"), meta)
    except Exception as e:  # noqa: BLE001 — o erro volta pro Vertexing mostrar
        traceback.print_exc()
        yield {"type": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import runpod

    _paint()  # carrega antes do primeiro job
    runpod.serverless.start({"handler": handler})
