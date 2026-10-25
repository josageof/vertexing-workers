# vertexing-workers

Worker GPU do [Vertexing](https://github.com/josageof) para o RunPod Serverless,
publicado como imagem no Docker Hub pelo GitHub Actions.

| Imagem | O que faz | Pesos (cache de modelos do RunPod) |
|---|---|---|
| `josageof/vertexing-trellis2` | imagem (ou fotos reais de vários ângulos) → malha 3D com textura (TRELLIS.2 1024_cascade + to_glb) | `microsoft/TRELLIS.2-4B` |

A imagem não contém pesos nem chaves. O resultado volta em pedaços pela
própria resposta do RunPod (`common/vx_stream.py`, lido via `/stream`).

Testados e removidos em 24/10/2026, sem ganho sobre o TRELLIS.2 (histórico no
git): textura Hunyuan3D-Paint (`paint/`), refino de forma com Hunyuan3D-2.1
(`refine/`), vistas geradas por Qwen-Image-Edit (`views/`), DINO-Lock e os
modos de várias fotos por média simples.

## Endpoint (RunPod → Serverless → New Endpoint → Docker image)

- Container image: `josageof/vertexing-trellis2:<sha do commit>`
- Container start command: vazio (usa o `CMD` da imagem)
- Cached model: o da tabela acima
- GPUs: 24 GB e 24 GB PRO (e 16 GB); qualquer data center
- Sem network volume

## Licenças

- Código deste repositório: MIT.
- TRELLIS.2 (microsoft/TRELLIS.2): MIT. BiRefNet: MIT. DINOv3: licença da Meta
  (baixado na partida, não redistribuído aqui).
