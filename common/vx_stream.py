"""Devolve um arquivo grande pela própria resposta do RunPod, em pedaços.

Sem volume de rede, a malha (~5–9 MB) não cabe numa resposta só (os limites
documentados são 10 MB no /run e 20 MB no /runsync, e o de saída não é
documentado). O handler vira um gerador: primeiro um item "meta" com o
tamanho e o hash, depois os pedaços em base64. O Vertexing lê pelo /stream,
junta e confere o hash.
"""
import base64
import hashlib

CHUNK = 2_000_000  # bytes por pedaço (~2,7 MB em base64)


def stream_bytes(data: bytes, meta: dict):
    n = (len(data) + CHUNK - 1) // CHUNK
    yield {"type": "meta", "chunks": n, "bytes": len(data),
           "sha256": hashlib.sha256(data).hexdigest(), **meta}
    for i in range(n):
        yield {"type": "chunk", "i": i,
               "data": base64.b64encode(data[i * CHUNK:(i + 1) * CHUNK]).decode()}
