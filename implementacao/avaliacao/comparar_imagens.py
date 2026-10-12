"""
Compara duas imagens (ex.: saida do motor C++ x saida do run.py, ou quadro
sintetizado x ground truth) usando PSNR, SSIM e diferenca absoluta.

Uso:
    python implementacao/avaliacao/comparar_imagens.py referencia.png teste.png
"""

import sys

import cv2
import numpy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    referencia = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    teste = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    if referencia is None or teste is None:
        print("Erro: nao foi possivel abrir as imagens.")
        sys.exit(1)

    if referencia.shape != teste.shape:
        print(f"Erro: shapes diferentes {referencia.shape} x {teste.shape}")
        sys.exit(1)

    diferenca = numpy.abs(referencia.astype(numpy.int16) - teste.astype(numpy.int16))

    psnr = peak_signal_noise_ratio(referencia, teste, data_range=255)
    ssim = structural_similarity(referencia, teste, channel_axis=2, data_range=255)

    print(f"PSNR:                {psnr:.2f} dB")
    print(f"SSIM:                {ssim:.6f}")
    print(f"Diferenca maxima:    {int(diferenca.max())} niveis")
    print(f"Pixels diferentes:   {100.0 * numpy.count_nonzero(diferenca) / diferenca.size:.3f} %")


if __name__ == "__main__":
    main()
