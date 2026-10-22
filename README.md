# vertexing-workers

Workers GPU do [Vertexing](https://github.com/josageof) para o RunPod Serverless,
publicados como imagens no Docker Hub pelo GitHub Actions.

| Imagem | O que faz | Pesos (cache de modelos do RunPod) |
|---|---|---|
| `josageof/vertexing-trellis2` | imagem → malha 3D (TRELLIS.2 1024, remalhada, cor por vértice) | `microsoft/TRELLIS.2-4B` |
| `josageof/vertexing-paint` | malha + imagem → GLB com textura (Hunyuan3D-Paint) | `tencent/Hunyuan3D-2` |

As imagens não contêm pesos nem chaves. O resultado volta em pedaços pela
própria resposta do RunPod (`common/vx_stream.py`, lido via `/stream`).

## Endpoint (RunPod → Serverless → New Endpoint → Docker image)

- Container image: `josageof/vertexing-trellis2:latest` (ou `-paint`)
- Container start command: vazio (usa o `CMD` da imagem)
- Cached model: o da tabela acima
- GPUs: 24 GB e 24 GB PRO (e 16 GB para o TRELLIS.2); qualquer data center
- Sem network volume

## Licenças

- Código deste repositório: MIT.
- TRELLIS.2 (microsoft/TRELLIS.2): MIT. BiRefNet: MIT. DINOv3: licença da Meta
  (baixado na partida, não redistribuído aqui).
- Hunyuan3D-2 (Tencent Hunyuan 3D Community License): proíbe uso na União
  Europeia, Reino Unido e Coreia do Sul; acima de 1 milhão de usuários ativos
  por mês exige licença comercial da Tencent. Os pesos não estão na imagem.
