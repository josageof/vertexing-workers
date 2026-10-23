"""Nuvem de pontos de entrada do codificador do VAE do Hunyuan3D-2.1.

Igual ao SharpEdgeSurfaceLoader oficial com num_sharp_points=0 (o que o
minimal_vae_demo usa): malha no cubo [-1, 1] pelo lado maior, pontos
uniformes na superfície com a normal e a marca de quina = 0. O oficial
amostra quinas mesmo pedindo zero e quebra em malha lisa (sem quinas);
este não passa por lá."""
import numpy as np
import torch
import trimesh


def surface_points(mesh: trimesh.Trimesh, n: int = 81920) -> torch.Tensor:
    m = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    lo, hi = m.bounds
    m.apply_translation(-(lo + hi) / 2)
    m.apply_scale(2 * 0.9999 / (hi - lo).max())
    pts, face_idx = trimesh.sample.sample_surface(m, n)
    normals = m.face_normals[face_idx]
    surface = np.concatenate([pts, normals, np.zeros((n, 1))], axis=1)
    return torch.from_numpy(surface).float().unsqueeze(0)  # (1, n, 7)
