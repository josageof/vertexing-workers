"""Teste de fumaça no build (sem GPU, sem pesos): roda o caminho todo do refino
com um VAE minúsculo de pesos aleatórios — amostragem da malha, codificação
(fps do torch_cluster), agendamento, decodificação e marching cubes. Pega
pacote faltando antes de a imagem chegar no worker."""
import sys

sys.path.insert(0, "/opt/Hunyuan3D-2.1/hy3dshape")
sys.path.insert(0, "/app")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from hy3dshape.models.autoencoders import ShapeVAE  # noqa: E402
from hy3dshape.pipelines import export_to_trimesh, retrieve_timesteps  # noqa: E402
from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from vx_surface import surface_points  # noqa: E402

vae = ShapeVAE(num_latents=64, embed_dim=8, width=64, heads=4, num_decoder_layers=1, num_encoder_layers=1,
               point_feats=4, pc_size=1280, pc_sharpedge_size=0, qkv_bias=False, qk_norm=True).eval()
surface = surface_points(trimesh.creation.icosphere(3), 1280)  # esfera: lisa, sem quinas
with torch.inference_mode():
    z = vae.encode(surface, sample_posterior=False)
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    timesteps, _ = retrieve_timesteps(sched, 3, "cpu", sigmas=np.linspace(0.5, 1, 3))
    for t in timesteps:
        z = sched.step(torch.zeros_like(z), t, z).prev_sample
    meshes = vae.latents2mesh(vae(z), bounds=1.01, mc_level=0.0, num_chunks=4096, octree_resolution=32,
                              mc_algo="mc", enable_pbar=False)
print("smoke ok: latente", tuple(z.shape), "| saída", type(export_to_trimesh(meshes)[0]).__name__)
