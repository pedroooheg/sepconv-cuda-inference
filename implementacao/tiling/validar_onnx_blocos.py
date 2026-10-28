"""
Valida o modelo sepconv_blocos.onnx (media e desvio recebidos como entrada)
contra a rede original em PyTorch, em blocos de varios formatos.

Uso:
    python implementacao/tiling/validar_onnx_blocos.py
"""

import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.argv = sys.argv[:1]  # run.py faz getopt no import
sys.path.insert(0, os.path.join(RAIZ, "implementacao", "exportacao_onnx"))

import numpy
import onnxruntime
import torch

from export_onnx import SepConvTileExtractor
from run import Network

# (altura, largura): blocos internos, de borda e retangulares
FORMATOS = [(704, 704), (416, 416), (704, 152), (248, 704), (1080, 1920)]


def main():
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    modelo = SepConvTileExtractor(Network()).cuda().eval()
    sessao = onnxruntime.InferenceSession(
        os.path.join(RAIZ, "modelos", "sepconv_blocos.onnx"),
        providers=[("CUDAExecutionProvider", {"use_tf32": 0})],
    )

    # Media e desvio de um quadro "inteiro" qualquer, diferentes dos do bloco
    media = torch.full((1, 1, 1, 1), 0.42, device="cuda")
    desvio = torch.full((1, 1, 1, 1), 0.27, device="cuda")

    for altura, largura in FORMATOS:
        um = torch.rand(1, 3, altura, largura, device="cuda")
        dois = torch.rand(1, 3, altura, largura, device="cuda")

        saidas_onnx = sessao.run(None, {
            "input_frame1": um.cpu().numpy(), "input_frame2": dois.cpu().numpy(),
            "mean": media.cpu().numpy(), "std": desvio.cpu().numpy(),
        })
        saidas_torch = [t.cpu().numpy() for t in modelo(um, dois, media, desvio)]

        erro = max(float(numpy.abs(a - b).max()) for a, b in zip(saidas_onnx, saidas_torch))
        escala = max(float(numpy.abs(b).max()) for b in saidas_torch)

        print(f"{largura}x{altura}: erro_max={erro:.2e} (valor_max={escala:.2e}, relativo={erro / escala:.1e})")


if __name__ == "__main__":
    main()
