"""Worker de refino de forma com Hunyuan3D-2.1 (imagem Docker própria, sem volume).

"img2img em 3D": a malha do TRELLIS.2 (fiel à imagem, mas com degraus de
voxel) é codificada no espaço latente do Hunyuan3D-2.1 pelo VAE oficial (o
2.1 publica o codificador; o 2.0 não), recebe ruído parcial e é "redesenhada"
pelo DiT guiado pela imagem. A superfície sai do marching cubes de um campo
contínuo, então os degraus somem. A intensidade controla o quanto ele pode
mudar: 0 = só codifica e decodifica (controle), ~0,3 = alisa mantendo a
forma, ~0,7 = o Hunyuan "adivinha" bem mais.

Entrada (job["input"]):
  mesh       npz em base64 com "vertices" (float32) e "faces" (int32), Y pra
             cima e frente pra +Z (convenção glTF, a mesma do to_glb do TRELLIS)
  image      PNG/JPG/WEBP em base64 (sem fundo transparente: o fundo é removido aqui)
  strengths  lista de intensidades 0–1 (padrão [0, 0.3, 0.5, 0.7]); todas num
             job só, pra pagar a partida a frio uma vez
  steps      passos do agendamento completo (padrão 50; cada intensidade roda
             a fração final dele)
  guidance   escala de guiança (padrão 5.0)
  octree     resolução do marching cubes (padrão 384)
  seed       semente do ruído (padrão 42)

Saída (stream): {"type":"meta", chunks, bytes, sha256, format:"npz", results:[...]}
e depois os pedaços. O npz tem v<i>/f<i> por intensidade, no cubo [-1, 1] do
Hunyuan (o Vertexing devolve pra escala da malha original pela caixa).

Licença dos pesos: Tencent Hunyuan 3D 2.1 Community License (proíbe uso na UE,
Reino Unido e Coreia do Sul; acima de 1 mi de usuários/mês exige licença).
"""
import base64
import io
import os
import sys
import time
import traceback

sys.path.insert(0, "/opt/Hunyuan3D-2.1/hy3dshape")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402

from vx_stream import stream_bytes  # noqa: E402

REPO = "tencent/Hunyuan3D-2.1"
_CACHE_ROOTS = ["/runpod-volume/huggingface-cache/hub", "/runpod/cache/huggingface/hub"]
_MODELS: dict = {}
_LOAD_INFO: dict = {}


def _log(msg: str) -> None:
    print(f"[vertexing-refine {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _weights_root() -> None:
    """O Hunyuan procura os pesos em $HY3DGEN_MODELS/tencent/Hunyuan3D-2.1/<subpasta>:
    aponta esse caminho pro snapshot do cache do RunPod (ou baixa as duas
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

        snap = snapshot_download(REPO, allow_patterns=["hunyuan3d-dit-v2-1/*", "hunyuan3d-vae-v2-1/*"])
    base = "/tmp/hy3dgen"
    os.makedirs(f"{base}/tencent", exist_ok=True)
    link = f"{base}/{REPO}"
    if not os.path.lexists(link):
        os.symlink(snap, link)
    os.environ["HY3DGEN_MODELS"] = base
    _LOAD_INFO["model_source"] = source


def _models():
    if "pipe" not in _MODELS:
        from hy3dshape.models.autoencoders import ShapeVAE
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        t0 = time.time()
        _weights_root()
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(REPO)
        # o VAE avulso é o que traz o codificador (o do checkpoint do DiT só
        # decodifica); usa o mesmo pros dois lados, pra ida e volta baterem
        vae = ShapeVAE.from_pretrained(REPO, use_safetensors=False, variant="fp16")
        pipe.vae = vae.to(pipe.device, dtype=pipe.dtype).eval()
        _MODELS["pipe"] = pipe
        _LOAD_INFO["load_s"] = round(time.time() - t0, 1)
        _log(f"Hunyuan3D-2.1 carregado em {time.time() - t0:.1f}s de {_LOAD_INFO['model_source']}")
    return _MODELS["pipe"]


def _image(b64: str) -> Image.Image:
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    if img.mode == "RGBA" and img.getextrema()[3][0] < 250:
        return img  # já tem fundo transparente
    from hy3dshape.rembg import BackgroundRemover

    if "rembg" not in _MODELS:
        _MODELS["rembg"] = BackgroundRemover()
    return _MODELS["rembg"](img.convert("RGB"))


def _encode(pipe, mesh: trimesh.Trimesh) -> torch.Tensor:
    """Malha -> latente "limpo" (sigma 1 no agendamento do Hunyuan), do jeito do
    minimal_vae_demo oficial: 81.920 pontos com normal, sem pontos de quina."""
    from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader

    loader = SharpEdgeSurfaceLoader(num_sharp_points=0, num_uniform_points=81920)
    surface = loader(mesh).to(pipe.device, dtype=pipe.dtype)
    latents = pipe.vae.encode(surface, sample_posterior=False)
    return latents * pipe.vae.scale_factor


def _refine(pipe, z0, cond, strength: float, steps: int, guidance: float, seed: int) -> torch.Tensor:
    """Agendamento do Hunyuan: sigma 0 = ruído puro, 1 = forma pronta, e
    x = sigma * forma + (1 - sigma) * ruído. Começa no meio do caminho
    (sigma = 1 - intensidade) e roda só os passos que faltam."""
    from hy3dshape.pipelines import retrieve_timesteps

    if strength <= 0:
        return z0
    s0 = 1.0 - strength
    n = max(2, round(steps * strength))
    timesteps, _ = retrieve_timesteps(pipe.scheduler, n, pipe.device, sigmas=np.linspace(s0, 1, n))
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    noise = torch.randn(z0.shape, generator=gen, device=pipe.device, dtype=z0.dtype)
    latents = s0 * z0 + (1 - s0) * noise
    for t in timesteps:
        x = torch.cat([latents] * 2)  # com e sem condição (guiança sem classificador)
        timestep = t.expand(x.shape[0]).to(latents.dtype) / pipe.scheduler.config.num_train_timesteps
        pred_cond, pred_uncond = pipe.model(x, timestep, cond, guidance=None).chunk(2)
        latents = pipe.scheduler.step(pred_uncond + guidance * (pred_cond - pred_uncond), t, latents).prev_sample
    return latents


def handler(job):
    """Gerador: o npz sai em pedaços (vx_stream), lidos pelo /stream."""
    inp = job.get("input") or {}
    timings: dict = {}
    torch.cuda.reset_peak_memory_stats()
    try:
        pipe = _models()
        d = np.load(io.BytesIO(base64.b64decode(inp["mesh"])))
        mesh = trimesh.Trimesh(d["vertices"], d["faces"], process=False)
        image = _image(inp["image"])
        strengths = [float(s) for s in inp.get("strengths", [0, 0.3, 0.5, 0.7])]
        steps, guidance = int(inp.get("steps", 50)), float(inp.get("guidance", 5.0))
        octree, seed = int(inp.get("octree", 384)), int(inp.get("seed", 42))

        out, results = {}, []
        with torch.inference_mode():
            t0 = time.time()
            z0 = _encode(pipe, mesh)
            cond_inputs = pipe.prepare_image(image, None)
            cond = pipe.encode_cond(image=cond_inputs.pop("image"), additional_cond_inputs=cond_inputs,
                                    do_classifier_free_guidance=True, dual_guidance=False)
            timings["encode_s"] = round(time.time() - t0, 1)
            pipe.set_surface_extractor("mc")
            for i, s in enumerate(strengths):
                t0 = time.time()
                latents = _refine(pipe, z0, cond, s, steps, guidance, seed)
                m = pipe._export(latents, "trimesh", 1.01, 0.0, 20000, octree, "mc", enable_pbar=False)[0]
                out[f"v{i}"] = np.asarray(m.vertices, dtype=np.float32)
                out[f"f{i}"] = np.asarray(m.faces, dtype=np.int32)
                results.append({"strength": s, "faces": int(len(m.faces)), "s": round(time.time() - t0, 1)})
                _log(f"intensidade {s}: {len(m.faces)} faces em {time.time() - t0:.1f}s")
        buf = io.BytesIO()
        np.savez_compressed(buf, **out)
        meta = {
            "format": "npz", "results": results, "timings": timings,
            "gpu": torch.cuda.get_device_name(0),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            **_LOAD_INFO,
        }
        yield from stream_bytes(buf.getvalue(), meta)
    except Exception as e:  # noqa: BLE001 — o erro volta pro Vertexing mostrar
        traceback.print_exc()
        yield {"type": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import runpod

    _models()  # carrega antes do primeiro job
    runpod.serverless.start({"handler": handler})
