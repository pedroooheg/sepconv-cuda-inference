"""
Valida o modelo ONNX exportado comparando suas saidas (kernels 1D) com as da
rede original em PyTorch, em diferentes resolucoes.

Uso (com o venv ativo):
    python implementacao/exportacao_onnx/validar_onnx.py
"""

import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.argv = sys.argv[:1]  # run.py faz getopt no import
sys.path.insert(0, os.path.join(RAIZ, "original"))

import numpy
import onnxruntime
import torch

from export_onnx import SepConvFeatureExtractor
from run import Network

ONNX_PATH = os.path.join(RAIZ, "modelos", "sepconv_paper.onnx")

RESOLUCOES = [
    (256, 448),    # Vimeo-90K
    (720, 1280),   # HD
    (1080, 1920),  # Full HD (resolucao usada no export)
    (540, 960),
    (1088, 1920),
]


def main():
    torch.set_grad_enabled(False)

    # TF32 desligado nos dois lados para comparar em FP32 puro.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    modelo_torch = SepConvFeatureExtractor(Network()).cuda().eval()

    sessao = onnxruntime.InferenceSession(
        ONNX_PATH,
        providers=[("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"],
    )
    print("Providers ativos:", sessao.get_providers())

    for altura, largura in RESOLUCOES:
        um = torch.rand(1, 3, altura, largura, device="cuda")
        dois = torch.rand(1, 3, altura, largura, device="cuda")

        try:
            saidas_onnx = sessao.run(None, {
                "input_frame1": um.cpu().numpy(),
                "input_frame2": dois.cpu().numpy(),
            })
        except Exception as erro:
            print(f"{altura}x{largura}: FALHOU no ONNX -> {str(erro).splitlines()[0]}")
            continue

        saidas_torch = [t.cpu().numpy() for t in modelo_torch(um, dois)]

        erros = [
            float(numpy.abs(a - b).max()) if a.shape == b.shape else float("inf")
            for a, b in zip(saidas_onnx, saidas_torch)
        ]
        escala = max(float(numpy.abs(b).max()) for b in saidas_torch)

        print(
            f"{altura}x{largura}: shape={saidas_onnx[0].shape} "
            f"erro_max={max(erros):.2e} (valor_max={escala:.2e})"
        )


if __name__ == "__main__":
    main()
