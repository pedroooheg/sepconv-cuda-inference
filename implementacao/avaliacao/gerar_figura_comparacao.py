"""
Figura qualitativa para o TCC: quadro real x quadro sintetizado x mapa de erro.

Para nao escolher um exemplo favoravel, usa o quadro sintetizado de PSNR
MEDIANO da resolucao indicada (segundo quadros/<impl>_<res>.csv).

Uso:
    python implementacao/avaliacao/gerar_figura_comparacao.py resultados/testes_gta/<data_hora> [--res 1080p] [--impl cpp]
"""

import argparse
import csv
import os

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avaliar_qualidade import area_ativa  # noqa: E402


def ler_quadro(caminho, indice):
    captura = cv2.VideoCapture(caminho)
    captura.set(cv2.CAP_PROP_POS_FRAMES, indice)
    ok, quadro = captura.read()

    if not ok:
        raise SystemExit(f"Erro: nao foi possivel ler o quadro {indice} de {caminho}")

    return quadro


def main():
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument("pasta")
    argumentos.add_argument("--res", default="1080p")
    argumentos.add_argument("--impl", default="cpp")
    args = argumentos.parse_args()

    with open(os.path.join(args.pasta, "quadros", f"{args.impl}_{args.res}.csv"), newline="") as arquivo:
        linhas = sorted(csv.DictReader(arquivo), key=lambda linha: float(linha["psnr"]))

    mediana = linhas[len(linhas) // 2]
    indice = int(mediana["quadro"])

    real = ler_quadro(os.path.join(RAIZ, "dados", "testes", f"gta_{args.res}_30fps.mkv"), indice)
    sintetizado = ler_quadro(os.path.join(args.pasta, "saidas", f"{args.impl}_{args.res}.mkv"), indice)

    # Mesma area avaliada nas metricas: sem as tarjas pretas (letterbox).
    y0, y1, x0, x1 = area_ativa(os.path.join(RAIZ, "dados", "testes", f"gta_{args.res}_30fps.mkv"))
    real = real[y0:y1, x0:x1]
    sintetizado = sintetizado[y0:y1, x0:x1]

    # Erro absoluto medio entre os canais, em niveis de intensidade (0-255).
    erro = numpy.abs(real.astype(numpy.float32) - sintetizado.astype(numpy.float32)).mean(axis=2)

    # (a) e (b) lado a lado e (c) embaixo, para a figura nao ficar alta demais.
    figura = plt.figure(figsize=(6.3, 4.9))
    grade = figura.add_gridspec(2, 2, height_ratios=[1, 2.05], hspace=0.12, wspace=0.04)
    eixos = [figura.add_subplot(grade[0, 0]), figura.add_subplot(grade[0, 1]), figura.add_subplot(grade[1, :])]

    paineis = [
        (cv2.cvtColor(real, cv2.COLOR_BGR2RGB), f"(a) Quadro real {indice}"),
        (cv2.cvtColor(sintetizado, cv2.COLOR_BGR2RGB),
         f"(b) Sintetizado: PSNR {float(mediana['psnr']):.2f} dB, SSIM {float(mediana['ssim']):.4f}".replace(".", ",")),
    ]

    for eixo, (imagem, titulo) in zip(eixos[:2], paineis):
        eixo.imshow(imagem)
        eixo.set_title(titulo, fontsize=8, loc="left", color="#0b0b0b")
        eixo.axis("off")

    # Escala limitada a 64 niveis para destacar os erros pequenos
    # (acima disso fica tudo no tom mais escuro).
    mapa = eixos[2].imshow(erro, cmap="Blues", vmin=0, vmax=64)
    eixos[2].set_title("(c) Erro absoluto |a − b| (média dos canais)", fontsize=8, loc="left", color="#0b0b0b")
    eixos[2].axis("off")

    barra = figura.colorbar(mapa, ax=eixos[2], orientation="horizontal", fraction=0.05, pad=0.03, aspect=50)
    barra.set_label("Níveis de intensidade (0–255; ≥ 64 saturado)", fontsize=7.5, color="#52514e")
    barra.ax.tick_params(labelsize=7.5, colors="#52514e")
    barra.outline.set_visible(False)

    destino = os.path.join(args.pasta, "graficos")
    os.makedirs(destino, exist_ok=True)

    for extensao in ("pdf", "png"):
        figura.savefig(os.path.join(destino, f"comparacao_quadro_{args.res}.{extensao}"),
                       dpi=200, bbox_inches="tight", facecolor="#ffffff")

    print(f"Quadro {indice} (PSNR mediano {float(mediana['psnr']):.2f} dB) -> "
          f"{os.path.join(destino, f'comparacao_quadro_{args.res}.pdf')}")


if __name__ == "__main__":
    main()
