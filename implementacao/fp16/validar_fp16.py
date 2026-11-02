"""
Compara os modelos FP16 com os FP32 em quadros reais: filtros e quadro
sintetizado. A sintese reproduz o kernel do motor C++ (bordas replicadas,
normalizacao pela soma dos filtros e truncamento para 8 bits), com os filtros
FP16 convertidos para FP32, como o motor fara.

Uso:
    python implementacao/fp16/validar_fp16.py [--res 1080p] [--pares 0 10 20]
"""

import argparse
import os

import cv2
import numpy
import onnxruntime
import torch
import torch.nn.functional as F

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K = 51
R = K // 2

argumentos = argparse.ArgumentParser()
argumentos.add_argument("--res", default="1080p")
argumentos.add_argument("--pares", type=int, nargs="+", default=[0, 10, 20],
                        help="indice do primeiro quadro de cada par no video de 15 FPS")
args = argumentos.parse_args()


def sessao(nome):
    # Mesmas opcoes de memoria do motor; sem elas, duas sessoes em 1080p e a
    # sintese no PyTorch nao cabem juntas em 8 GB.
    opcoes = onnxruntime.SessionOptions()
    opcoes.enable_mem_pattern = False

    return onnxruntime.InferenceSession(
        os.path.join(RAIZ, "modelos", nome),
        sess_options=opcoes,
        providers=[("CUDAExecutionProvider", {"use_tf32": 0, "arena_extend_strategy": "kNextPowerOfTwo"})],
    )


def ler_par(indice):
    captura = cv2.VideoCapture(os.path.join(RAIZ, "dados", "testes", f"gta_{args.res}_15fps.mkv"))
    captura.set(cv2.CAP_PROP_POS_FRAMES, indice)
    quadros = [captura.read()[1] for _ in range(2)]

    # BGR planar em [0, 1], como o pre-processamento do motor.
    return [numpy.ascontiguousarray(q.transpose(2, 0, 1)[None].astype(numpy.float32) / 255.0) for q in quadros]


def sintetizar(quadros, filtros):
    """filtros na ordem do ONNX: v1, v2, h1, h2 (cada um 1 x 51 x H x W)."""
    v1, v2, h1, h2 = [torch.from_numpy(f.astype(numpy.float32)).cuda()[0] for f in filtros]
    _, altura, largura = v1.shape
    acumulado = torch.zeros(3, altura, largura, device="cuda")
    norma = v1.sum(0) * h1.sum(0) + v2.sum(0) * h2.sum(0)

    for quadro, ver, hor in ((quadros[0], v1, h1), (quadros[1], v2, h2)):
        borda = F.pad(torch.from_numpy(quadro).cuda(), (R, R, R, R), mode="replicate")[0]

        for fy in range(K):
            linha = torch.zeros(3, altura, largura, device="cuda")

            for fx in range(K):
                linha += hor[fx] * borda[:, fy:fy + altura, fx:fx + largura]

            acumulado += ver[fy] * linha

    norma = torch.where(norma.abs() < 0.01, torch.ones_like(norma), norma)
    saida = (acumulado / norma).clamp(0.0, 1.0) * 255.0
    return saida.to(torch.uint8).cpu().numpy()


def psnr(a, b):
    mse = numpy.mean((a.astype(numpy.float64) - b.astype(numpy.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * numpy.log10(255.0 ** 2 / mse)


def main():
    torch.set_grad_enabled(False)

    for nome, arquivo in (("inteiro", "sepconv_paper"), ("blocos", "sepconv_blocos")):
        for indice in args.pares:
            um, dois = ler_par(indice)
            dados = {"input_frame1": um, "input_frame2": dois}

            if nome == "blocos":
                # Mesma normalizacao global que o motor calcula para os blocos.
                media = numpy.float32((um.mean() + dois.mean()) / 2)
                desvio = numpy.float32(numpy.sqrt(((um.var() + (media - um.mean()) ** 2) +
                                                   (dois.var() + (media - dois.mean()) ** 2)) / 2))
                dados["mean"] = numpy.full((1, 1, 1, 1), media, numpy.float32)
                dados["std"] = numpy.full((1, 1, 1, 1), desvio, numpy.float32)

            # Uma sessao por vez: a arena de cada uma so e liberada ao destrui-la.
            filtros32 = sessao(f"{arquivo}.onnx").run(None, dados)
            filtros16 = sessao(f"{arquivo}_fp16.onnx").run(None, dados)

            finitos = all(numpy.isfinite(f.astype(numpy.float32)).all() for f in filtros16)
            erro = max(float(numpy.abs(a - b.astype(numpy.float32)).max()) for a, b in zip(filtros32, filtros16))
            escala = max(float(numpy.abs(a).max()) for a in filtros32)

            saida32 = sintetizar((um, dois), filtros32)
            saida16 = sintetizar((um, dois), filtros16)
            diferenca = numpy.abs(saida32.astype(int) - saida16.astype(int))

            print(f"par {indice:2} | {nome:7} | filtros: finitos={finitos} erro_max={erro:.2e} "
                  f"(relativo {erro / escala:.1e}) | quadro: PSNR={psnr(saida32, saida16):.2f} dB "
                  f"dif_max={diferenca.max()} pixels_dif={100 * (diferenca > 0).mean():.3f}%", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
