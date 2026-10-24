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
  views_mode       com images: "average" (padrão) | "structure" (extras só no
                   volume) | "spatial" (cada foto manda no lado que mostra)
  view_azimuths    graus de cada imagem (0 frente, 90 esquerda do objeto, 180
                   costas, 270 direita); padrão 0/90/180/270 com 4 imagens
  front_weight     peso da frente na combinação (padrão 1)
  views_t_min      abaixo deste t (1 = ruído, 0 = fim) só a frente conta (padrão 0)
  dino_lock        DINO-Lock do fork ComfyUI-Trellis2 (0 = desligado; o fork usa 0,65)
  dino_substeps    subpassos nos primeiros 40 % com dino_lock (padrão 4)
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
from contextlib import contextmanager, nullcontext
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


def _view_dirs(azimuths: list) -> torch.Tensor:
    """Direção (x, y, z do latente, Z pra cima) do lado do objeto que cada
    foto mostra. O TRELLIS.2 gera no referencial do próprio objeto, não da
    câmera: a frente fica em -Y e a esquerda do objeto em +X (conferido no
    GLB do Titã). Azimute em graus: 0 frente, 90 esquerda, 180 costas, 270 direita."""
    a = torch.tensor([float(x) for x in azimuths]) * torch.pi / 180
    return torch.stack([torch.sin(a), -torch.cos(a), torch.zeros_like(a)], 1)


def _surface_normals(coords: torch.Tensor) -> torch.Tensor:
    """Normal aproximada de cada voxel do latente esparso (só a casca da
    superfície): preenche o interior (o que não se alcança da borda da grade
    andando por voxels vazios), borra o sólido e pega o gradiente. Onde a
    casca tem furo o preenchimento vaza; por isso soma um pouco da direção
    radial (a partir do eixo vertical), que nunca é nula longe do eixo."""
    import torch.nn.functional as F

    c = coords[:, 1:].long()
    c = c - c.min(0).values + 2
    size = [int(x) + 3 for x in c.max(0).values]
    occ = torch.zeros(size, dtype=torch.bool, device=c.device)
    occ[c[:, 0], c[:, 1], c[:, 2]] = True
    out = torch.zeros_like(occ)
    for border in (out[0], out[-1], out[:, 0], out[:, -1], out[:, :, 0], out[:, :, -1]):
        border.fill_(True)
    out &= ~occ
    cross = torch.zeros(1, 1, 3, 3, 3, device=c.device)
    cross[0, 0, 1, 1, :] = cross[0, 0, 1, :, 1] = cross[0, 0, :, 1, 1] = 1  # 6 vizinhos
    for _ in range(sum(size)):
        grown = (F.conv3d(out[None, None].float(), cross, padding=1)[0, 0] > 0) & ~occ
        if torch.equal(grown, out):
            break
        out = grown
    solid = (~out).float()[None, None]
    for _ in range(2):
        solid = F.avg_pool3d(solid, 5, 1, 2, count_include_pad=False)
    grads = torch.gradient(solid[0, 0])
    n = -torch.stack([g[c[:, 0], c[:, 1], c[:, 2]] for g in grads], 1)
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-6)
    radial = c.float() - c.float().mean(0)
    radial[:, 2] = 0
    radial = radial / radial.norm(dim=1, keepdim=True).clamp_min(1e-6)
    n = n + 0.25 * radial
    return n / n.norm(dim=1, keepdim=True).clamp_min(1e-6)


def _spatial_weights(coords: torch.Tensor, dirs: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """Peso (voxels x fotos, soma 1 por voxel) de cada foto em cada voxel:
    cos² entre a normal e o lado que a foto mostra, mais um piso pequeno pra
    o que nenhuma foto vê de frente (topo da cabeça, sola), vezes o peso
    a priori de cada foto (front_weight)."""
    n = _surface_normals(coords)
    cos = (n @ dirs.to(n).T).clamp_min(0)
    w = (cos ** 2 + 0.05) * prior.to(n)
    return w / w.sum(1, keepdim=True)


@contextmanager
def _multi_image(pipe, images: list, mode: str = "average", azimuths: list | None = None,
                 front_weight: float = 1.0, t_min: float = 0.0):
    """run() só aceita uma imagem: o get_cond passa a devolver o
    condicionamento de todas, e nos amostradores a previsão de cada passo
    vira uma combinação das previsões por imagem — o "multidiffusion" do
    run_multi_image do TRELLIS v1. A guiança é linear, então combinar depois
    dela dá o mesmo resultado. Modos:
      average    média em estrutura, forma e textura (teste 4a)
      structure  fotos extras só na estrutura esparsa (volume/silhueta);
                 forma fina e textura só com a frente
      spatial    estrutura pela média; forma e textura com peso por voxel
                 conforme o lado pra onde a superfície aponta (cada foto
                 manda no lado que ela mostra)
    front_weight multiplica o peso da frente; abaixo de t_min (t vai de 1,
    ruído, a 0) só a frente conta — as extras moldam só o começo."""
    n = len(images)
    if azimuths is None:
        azimuths = [0, 90, 180, 270] if n == 4 else [360 * i / n for i in range(n)]
    if len(azimuths) != n:
        raise ValueError(f"view_azimuths tem {len(azimuths)} valores para {n} imagens")
    dirs = _view_dirs(azimuths)
    prior = torch.ones(n)
    prior[0] = float(front_weight)
    get_cond = pipe.get_cond
    pipe.get_cond = lambda image, resolution, include_neg_cond=True: get_cond(images, resolution, include_neg_cond)
    stages = {"structure": pipe.sparse_structure_sampler, "shape": pipe.shape_slat_sampler,
              "texture": pipe.tex_slat_sampler}
    patched = {}
    for stage, s in stages.items():
        if id(s) in patched:
            if mode == "average":
                continue
            raise RuntimeError("amostradores compartilhados entre estágios: o modo por estágio não se aplica")
        single = s._inference_model
        only_front = mode == "structure" and stage != "structure"
        spatial = mode == "spatial" and stage != "structure"
        cache: dict = {}

        def combined(model, x_t, t, cond=None, _single=single, _front=only_front, _spatial=spatial,
                     _cache=cache, neg_cond=None, **kwargs):
            def pred(i):
                extra = {"neg_cond": neg_cond[i:i + 1]} if neg_cond is not None else {}
                return _single(model, x_t, t, cond[i:i + 1], **extra, **kwargs)

            if _front or t < t_min:
                return pred(0)
            if not _spatial:
                return sum(pred(i) * float(prior[i]) for i in range(n)) / float(prior.sum())
            # as coordenadas não mudam ao longo da amostragem de um estágio
            key = (x_t.coords.data_ptr(), tuple(x_t.coords.shape))
            if key not in _cache:
                _cache.clear()
                _cache[key] = _spatial_weights(x_t.coords, dirs, prior)
            w = _cache[key].to(x_t.feats.dtype)
            feats = sum(pred(i).feats * w[:, i:i + 1] for i in range(n))
            return x_t.replace(feats)

        s._inference_model = combined
        patched[id(s)] = s
    try:
        yield
    finally:
        del pipe.get_cond
        for s in patched.values():
            del s._inference_model


def _lock_strength(i: int, steps: int, base: float, cap: float) -> float:
    """Agenda do DINO-Lock: cap nos primeiros 40 % dos passos, rampa cosseno
    até base entre 40 e 70 %, base no resto."""
    found_end, ramp_end = int(steps * 0.4), int(steps * 0.7)
    if i < found_end:
        return cap
    if i >= ramp_end:
        return base
    blend = 0.5 * (1.0 - np.cos(np.pi * (i - found_end) / max(1, ramp_end - found_end)))
    return float(cap + (base - cap) * blend)


@contextmanager
def _dino_lock(pipe, strength: float, substeps: int = 4, cap: float = 0.92):
    """DINO-Lock do fork ComfyUI-Trellis2 (visualbruno, MIT; DinoLockMixin em
    trellis2/pipelines/samplers/flow_euler.py), portado sem mudar o
    comportamento. Apesar do nome, é uma agenda de guiança: a cada passo
    mistura a velocidade com CFG e a só condicionada (guiança 1) — o que
    equivale a baixar a CFG —, suaviza a velocidade entre passos (EMA 0,85)
    e, nos primeiros 40 %, integra em `substeps` subpassos de Euler (esses
    subpassos usam a CFG cheia, como no original). Vale para os três
    estágios; chama s._inference_model, então combina com _multi_image."""
    samplers = {id(s): s for s in (pipe.sparse_structure_sampler, pipe.shape_slat_sampler, pipe.tex_slat_sampler)}
    for s in samplers.values():

        @torch.no_grad()
        def sample(model, noise, cond=None, neg_cond=None, steps=50, rescale_t=1.0, guidance_strength=3.0,
                   guidance_interval=(0.0, 1.0), verbose=True, tqdm_desc="Sampling", _s=s, **kwargs):
            from easydict import EasyDict as edict

            kw = {**kwargs, "neg_cond": neg_cond, "guidance_strength": guidance_strength,
                  "guidance_interval": guidance_interval}
            u = np.linspace(1, 0, steps + 1)
            ts = (rescale_t * u / (1 + (rescale_t - 1) * u)).tolist()
            x, v_ema = noise, None
            for i in range(steps):
                t, t_prev = ts[i], ts[i + 1]
                lock = _lock_strength(i, steps, strength, cap)
                guided = _s._inference_model(model, x, t, cond, **kw)
                cfg_on = guidance_strength != 1 and guidance_interval[0] <= t <= guidance_interval[1]
                # fora do intervalo da guiança as duas são iguais: poupa uma passada
                pos = _s._inference_model(model, x, t, cond, **{**kw, "guidance_strength": 1.0}) if cfg_on else guided
                v = (1.0 - lock) * guided + lock * pos if lock > 0 else guided
                if v_ema is not None:
                    v = 0.85 * v + 0.15 * v_ema
                v_ema = v
                if substeps > 1 and lock >= 0.9:
                    dt = (t - t_prev) / substeps
                    for k in range(substeps):
                        x = x - dt * _s._inference_model(model, x, t - k * dt, cond, **kw)
                else:
                    x = x - (t - t_prev) * v
            return edict({"samples": x})

        s.sample = sample
    try:
        yield
    finally:
        for s in samplers.values():
            del s.sample


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
    lock = float(inp.get("dino_lock", 0.0))
    with (_dino_lock(pipe, lock, int(inp.get("dino_substeps", 4))) if lock > 0 else nullcontext()):
        mesh = _generate(pipe, image, extra_images, inp, wanted, timings)
    timings["generate_s"] = round(time.time() - t0, 1)
    return _finish(mesh, inp, timings)


def _generate(pipe, image, extra_images: list, inp: dict, wanted: dict, timings: dict):
    if extra_images:
        views = [image, *extra_images]
        if wanted["preprocess_image"]:  # o run() só preprocessaria a primeira
            views = [pipe.preprocess_image(v) for v in views]
        mv = {
            "mode": inp.get("views_mode", "average"),
            "azimuths": inp.get("view_azimuths"),
            "front_weight": float(inp.get("front_weight", 1.0)),
            "t_min": float(inp.get("views_t_min", 0.0)),
        }
        if mv["mode"] not in ("average", "structure", "spatial"):
            raise ValueError(f"views_mode inválido: {mv['mode']}")
        with _multi_image(pipe, views, **mv):
            mesh = pipe.run(views[0], **_kwargs_for(pipe.run, {**wanted, "preprocess_image": False}))[0]
        timings["views"] = len(views)
    else:
        mesh = pipe.run(image, **_kwargs_for(pipe.run, wanted))[0]
    return mesh


def _finish(mesh, inp: dict, timings: dict):
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
