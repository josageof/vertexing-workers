"""Worker TRELLIS.2 (imagem Docker própria, RunPod Serverless sem volume).

Mesma geração validada no volume (remalhagem do to_glb + cor por vértice),
agora autocontida: código e bibliotecas compiladas na imagem, pesos do
TRELLIS.2 pelo cache de modelos do RunPod (campo "Cached model" =
microsoft/TRELLIS.2-4B) e o resultado devolvido em pedaços pela própria
resposta (vx_stream), sem volume de rede — o endpoint pode usar GPUs de
qualquer data center.

Entrada (job["input"]):
  image            PNG/JPG em base64
  images           opcional: lista de PNG/JPG em base64 do MESMO objeto por outros
                   ângulos (a primeira é a frente). Multidifusão do TRELLIS v1: a
                   cada passo o modelo prevê com cada imagem e usa a média (sem
                   pose de câmera); custa ~N vezes o tempo de geração
  seed             int (padrão 42)
  pipeline_type    "512" | "1024" | "1024_cascade" | "1536_cascade" (padrão 1024_cascade)
  steps, guidance  opcionais (passos/guiança da forma)
  decimation_target faces da malha (padrão 1.000.000; decimada na GPU)
  post             "remesh" (padrão) | "clean" | "glb" (to_glb oficial com textura, bem mais lento)

Saída (stream): {"type":"meta", chunks, bytes, sha256, format, faces, timings...}
e depois {"type":"chunk", i, data}. format "npz" = vertices (Z pra cima, cubo
[-0,5; 0,5]), faces e colors; "glb" = glTF (Y pra cima).
"""
import base64
import inspect
import io
from contextlib import contextmanager
import os
import sys
import time
import traceback

os.environ.setdefault("ATTN_BACKEND", "sdpa")
# a atenção esparsa do TRELLIS.2 não tem sdpa (só xformers/flash_attn/flash_attn_3)
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/opt/TRELLIS.2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from vx_stream import stream_bytes  # noqa: E402

_PIPELINES: dict = {}
_LOAD_INFO: dict = {}
_COLOR_ERROR: dict = {}  # vai na resposta: o log do worker não chega ao Vertexing
# Modelos "gated" (pedem acesso no HF) ou de licença não comercial, trocados
# por equivalentes públicos (MIT): DINOv3 pelo espelho e RMBG-2.0 pelo BiRefNet.
_SWAPS = {"facebook/dinov3": "camenduru/dinov3", "briaai/RMBG-2.0": "ZhengPeng7/BiRefNet"}
_CACHE_ROOTS = ["/runpod-volume/huggingface-cache/hub", "/runpod/cache/huggingface/hub"]


def _log(msg: str) -> None:
    print(f"[vertexing {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _snapshot(repo: str) -> tuple[str, str]:
    """Pasta dos pesos: cache de modelos do RunPod (não cobra o download) ou,
    se o endpoint não tiver o cache configurado, baixa do Hugging Face."""
    name = "models--" + repo.replace("/", "--")
    for root in _CACHE_ROOTS:
        snaps = os.path.join(root, name, "snapshots")
        if os.path.isdir(snaps) and os.listdir(snaps):
            return os.path.join(snaps, sorted(os.listdir(snaps))[-1]), f"cache ({root})"
    from huggingface_hub import snapshot_download

    return snapshot_download(repo), "download"


def _model_dir(repo: str = "microsoft/TRELLIS.2-4B") -> tuple[str, str]:
    """Cópia em /tmp com links pros pesos e os .json com as trocas de _SWAPS
    (o cache é só leitura e vem com a configuração original)."""
    snap, source = _snapshot(repo)
    dst = "/tmp/vx_" + repo.replace("/", "--")
    for dirpath, _, files in os.walk(snap):
        rel = os.path.relpath(dirpath, snap)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for fn in files:
            src_f, out = os.path.join(dirpath, fn), os.path.join(dst, rel, fn)
            if os.path.lexists(out):
                continue
            if fn.endswith(".json"):
                text = open(src_f, encoding="utf-8").read()
                for a, b in _SWAPS.items():
                    text = text.replace(a, b)
                open(out, "w", encoding="utf-8").write(text)
            else:
                os.symlink(os.path.realpath(src_f), out)
    return dst, source


def _trellis2():
    if "trellis2" not in _PIPELINES:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        t0 = time.time()
        path, source = _model_dir()
        pipe = Trellis2ImageTo3DPipeline.from_pretrained(path)
        pipe.cuda()
        _PIPELINES["trellis2"] = pipe
        _LOAD_INFO.update(model_source=source, load_s=round(time.time() - t0, 1))
        _log(f"TRELLIS.2 carregado em {time.time() - t0:.1f}s de {source}")
    return _PIPELINES["trellis2"]


def _kwargs_for(fn, wanted: dict) -> dict:
    """Passa só os argumentos que a versão instalada do pipeline aceita."""
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return wanted
    skipped = sorted(set(wanted) - set(params))
    if skipped:
        _log(f"ignorando argumentos não suportados: {skipped}")
    return {k: v for k, v in wanted.items() if k in params}


def _image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")


@contextmanager
def _multi_image(pipe, images: list):
    """run() só aceita uma imagem: o get_cond passa a devolver o
    condicionamento de todas, e nos três amostradores (estrutura, forma,
    textura) a previsão de cada passo vira a média das previsões por imagem —
    o modo "multidiffusion" do run_multi_image do TRELLIS v1. A guiança é
    linear, então fazer a média depois dela dá o mesmo resultado."""
    n = len(images)
    get_cond = pipe.get_cond
    pipe.get_cond = lambda image, resolution, include_neg_cond=True: get_cond(images, resolution, include_neg_cond)
    samplers = {id(s): s for s in (pipe.sparse_structure_sampler, pipe.shape_slat_sampler, pipe.tex_slat_sampler)}
    for s in samplers.values():
        single = s._inference_model

        def averaged(model, x_t, t, cond=None, _single=single, neg_cond=None, **kwargs):
            preds = []
            for i in range(n):
                extra = {"neg_cond": neg_cond[i:i + 1]} if neg_cond is not None else {}
                preds.append(_single(model, x_t, t, cond[i:i + 1], **extra, **kwargs))
            return sum(preds) / n

        s._inference_model = averaged
    try:
        yield
    finally:
        del pipe.get_cond
        for s in samplers.values():
            del s._inference_model


def _run_trellis2(inp: dict, timings: dict):
    pipe = _trellis2()
    image = _image(inp["image"])
    extra_images = [_image(b) for b in inp.get("images") or []]
    shape = {}
    if inp.get("steps"):
        shape["steps"] = int(inp["steps"])
    if inp.get("guidance"):
        shape["guidance_strength"] = float(inp["guidance"])
    wanted = {
        "seed": int(inp.get("seed", 42)),
        "pipeline_type": inp.get("pipeline_type", "1024_cascade"),
        "preprocess_image": bool(inp.get("preprocess_image", True)),
    }
    if shape:
        wanted["shape_slat_sampler_params"] = shape
        wanted["sparse_structure_sampler_params"] = {k: v for k, v in shape.items() if k == "steps"}
    t0 = time.time()
    if extra_images:
        views = [image, *extra_images]
        if wanted["preprocess_image"]:  # o run() só preprocessaria a primeira
            views = [pipe.preprocess_image(v) for v in views]
        with _multi_image(pipe, views):
            mesh = pipe.run(views[0], **_kwargs_for(pipe.run, {**wanted, "preprocess_image": False}))[0]
        timings["views"] = len(views)
    else:
        mesh = pipe.run(image, **_kwargs_for(pipe.run, wanted))[0]
    timings["generate_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    raw_faces = int(mesh.faces.shape[0])
    target = int(inp.get("decimation_target", 1_000_000))
    post = inp.get("post", "remesh")
    if post == "glb":
        out = _to_glb(mesh, target, int(inp.get("texture_size", 1024)))
    else:
        v, f = _postprocess(mesh, target, post, float(inp.get("remesh_project", 0.0)))
        v = v.detach().float().cpu().numpy()
        out = (v, f.detach().cpu().numpy().astype(np.int32), _vertex_colors(mesh, v))
    timings["postprocess_s"] = round(time.time() - t0, 1)
    return out, raw_faces


def _vertex_colors(mesh, v: np.ndarray) -> np.ndarray | None:
    """Cor por vértice direto do volume de atributos do modelo (canal
    base_color, já 0–1) — o mesmo dado que o to_glb assa numa textura, só
    que sem o mapeamento UV, que custava ~50–75 s. Cada vértice pega a cor do
    voxel ocupado mais próximo (erro de meio voxel, ~0,5 mm numa peça de 120)."""
    try:
        layout = mesh.layout["base_color"]
        if not isinstance(layout, slice):
            layout = list(layout) if len(layout) != 2 else slice(*layout)
        # voxel_size pode vir como número ou vetor (x, y, z)
        voxel = torch.as_tensor(mesh.voxel_size).float().cpu().numpy().reshape(-1)
        coords = torch.as_tensor(mesh.coords).cpu().numpy()[:, -3:].astype(np.int64)
        rgb = torch.as_tensor(mesh.attrs)[:, layout].float().cpu().numpy()[:, :3]

        # Busca por chave inteira (sem scipy, que não está no venv do worker):
        # voxel do vértice; se estiver vazio, o vizinho ocupado mais próximo
        # entre os 26 ao redor (o vértice fica na superfície, entre voxels).
        res = int(coords.max()) + 3
        key = lambda c: (c[:, 0] * res + c[:, 1]) * res + c[:, 2]  # noqa: E731
        order = np.argsort(key(coords + 1))
        keys = key(coords + 1)[order]
        cell = np.floor((v + 0.5) / voxel).astype(np.int64) + 1
        found = np.full(len(v), -1, dtype=np.int64)
        offsets = sorted(((x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)),
                         key=lambda o: o[0] ** 2 + o[1] ** 2 + o[2] ** 2)
        for off in offsets:
            todo = np.where(found < 0)[0]
            if not len(todo):
                break
            k = key(cell[todo] + np.array(off))
            pos = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
            hit = keys[pos] == k
            found[todo[hit]] = order[pos[hit]]
        colors = np.full((len(v), 3), 128, dtype=np.uint8)  # sem voxel por perto: cinza
        ok = found >= 0
        colors[ok] = np.clip(rgb[found[ok]] * 255, 0, 255).astype(np.uint8)
        _COLOR_ERROR.pop("colors_error", None)
        if not ok.all():
            _log(f"{(~ok).sum()} vértices sem voxel vizinho (cinza)")
        return colors
    except Exception as e:  # noqa: BLE001 — sem cor, a malha continua valendo
        _log(f"sem cor por vértice: {e!r}")
        _COLOR_ERROR["colors_error"] = f"{type(e).__name__}: {e}"
        return None


def _to_glb(mesh, target: int, texture_size: int):
    """Mesmo pós-processamento da saída oficial (e do Replicate): remalha,
    decima e assa a textura. A malha que sai daqui é a que o reparo do
    Vertexing já sabe fechar — a crua, mesmo remalhada por fora do to_glb,
    não fechava. Textura pequena (1024) porque só serve de cor de prévia."""
    import o_voxel

    mesh.simplify(16_777_216)  # limite do nvdiffrast, como no exemplo oficial
    return o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=target, texture_size=texture_size, remesh=True, remesh_band=1, remesh_project=0,
        verbose=False,
    )


def _postprocess(mesh, target: int, mode: str, project: float = 0.0):
    """Só a parte de geometria do o_voxel.postprocess.to_glb (sem UV nem
    textura, que é a etapa cara). A malha crua decimada direto não fechava
    no nosso reparo; o to_glb tem dois caminhos e os dois estão aqui:
      clean  — fecha buracos, conserta não-manifold, decima em 2 passos,
               unifica a orientação (padrão do to_glb)
      remesh — refaz a superfície por dual contouring numa faixa estreita
               (o que o exemplo oficial/Replicate usa; gera a "pele dupla")"""
    import cumesh

    m = cumesh.CuMesh()
    m.init(mesh.vertices, mesh.faces)
    m.fill_holes(max_hole_perimeter=3e-2)

    def cleanup():
        m.remove_duplicate_faces()
        m.repair_non_manifold_edges()
        m.remove_small_connected_components(1e-5)
        m.fill_holes(max_hole_perimeter=3e-2)

    if mode == "remesh":
        v, f = m.read()
        bvh = cumesh.cuBVH(v, f)
        resolution = int(round(1.0 / float(mesh.voxel_size)))  # cubo [-0,5; 0,5]
        band = 1
        m.init(*cumesh.remeshing.remesh_narrow_band_dc(
            v, f, center=torch.zeros(3, device=v.device), scale=(resolution + 3 * band) / resolution,
            # 0 = valor do exemplo oficial do to_glb (o que o Replicate usava).
            resolution=resolution, band=band, project_back=project, verbose=False, bvh=bvh,
        ))
        m.simplify(target, verbose=False)
    else:
        m.simplify(target * 3, verbose=False)
        cleanup()
        m.simplify(target, verbose=False)
        cleanup()
        m.unify_face_orientations()
    return m.read()


def handler(job):
    """Gerador: o resultado sai em pedaços (vx_stream), lidos pelo /stream."""
    inp = job.get("input") or {}
    timings: dict = {}
    torch.cuda.reset_peak_memory_stats()
    try:
        result, raw_faces = _run_trellis2(inp, timings)
        if isinstance(result, tuple):  # remesh|clean: npz, Z-up, cor por vértice
            v, f, colors = result
            buf = io.BytesIO()
            extra = {"colors": colors} if colors is not None else {}
            np.savez_compressed(buf, vertices=v.astype(np.float32), faces=f, **extra)
            data, fmt, n_faces = buf.getvalue(), "npz", len(f)
        else:  # glb: to_glb oficial, com textura
            data, fmt, n_faces = result.export(file_type="glb"), "glb", len(result.faces)
        meta = {
            "format": fmt, "faces": int(n_faces), "raw_faces": raw_faces, "timings": timings,
            "gpu": torch.cuda.get_device_name(0),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            **_LOAD_INFO, **_COLOR_ERROR,
        }
        yield from stream_bytes(data, meta)
    except Exception as e:  # noqa: BLE001 — o erro volta pro Vertexing mostrar
        traceback.print_exc()
        yield {"type": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import runpod

    _trellis2()  # carrega antes do primeiro job
    runpod.serverless.start({"handler": handler})
