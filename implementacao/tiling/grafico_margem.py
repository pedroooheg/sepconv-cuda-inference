"""
Grafico do estudo da margem de sobreposicao (estudo_margem.csv), no mesmo
estilo dos graficos da bateria de testes.

Tres paineis, em funcao da margem M, um traco por tamanho de bloco:
  - maior diferenca, em niveis de cinza (0-255), entre o quadro montado com
    blocos e o quadro inteiro;
  - tempo por quadro;
  - pico de memoria da GPU,
com o quadro inteiro como referencia (linha tracejada).

Uso:
    python implementacao/tiling/grafico_margem.py [--csv resultados/tiling/estudo_margem.csv] [--margem-adotada 96]
"""

import argparse
import csv
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RAIZ, "implementacao", "avaliacao"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from gerar_graficos import (  # noqa: E402
    CORES_BLOCOS, ESTILO_DO_BLOCO, LARGURA_FIGURA, MARCADORES_BLOCOS, TEXTO_SECUNDARIO,
    formatador, salvar,
)

argumentos = argparse.ArgumentParser()
argumentos.add_argument("--csv", default=os.path.join(RAIZ, "resultados", "tiling", "estudo_margem.csv"))
argumentos.add_argument("--margem-adotada", type=int, default=96)
args = argumentos.parse_args()


def main():
    with open(args.csv, newline="") as arquivo:
        linhas = list(csv.DictReader(arquivo))

    inteiro = next(linha for linha in linhas if linha["bloco"] == "inteiro")
    blocos = sorted({int(linha["bloco"]) for linha in linhas if linha["bloco"] != "inteiro"}, reverse=True)

    paineis = [
        ("dif_max", "Maior diferença\n(níveis de 0 a 255)", 0),
        ("tempo_ms", "Tempo por\nquadro (ms)", 0),
        ("pico_mib", "Pico de memória\nda GPU (MiB)", 0),
    ]

    figura, eixos = plt.subplots(1, len(paineis), figsize=(LARGURA_FIGURA, 2.8))
    alcas = []

    for indice, bloco in enumerate(blocos):
        estilo = ESTILO_DO_BLOCO.get(bloco, indice % len(CORES_BLOCOS))
        cor, marcador = CORES_BLOCOS[estilo], MARCADORES_BLOCOS[estilo]
        serie = sorted((linha for linha in linhas if linha["bloco"] == str(bloco)), key=lambda l: int(l["margem"]))
        margens = [int(linha["margem"]) for linha in serie]

        for eixo, (coluna, _, _) in zip(eixos, paineis):
            eixo.plot(margens, [float(linha[coluna]) for linha in serie], color=cor, linewidth=2,
                      marker=marcador, markersize=4.5, linestyle="--")

        alcas.append(Line2D([], [], color=cor, linewidth=2, linestyle="--", marker=marcador, markersize=5,
                            label=f"Blocos {bloco}×{bloco}"))

    for eixo, (coluna, titulo, casas) in zip(eixos, paineis):
        eixo.set_title(titulo, loc="left", fontsize=8.5)
        eixo.set_xlabel("Margem (pixels)")
        eixo.set_xticks(range(0, 161, 32))
        eixo.tick_params(axis="x", labelsize=7.5)
        eixo.yaxis.set_major_formatter(formatador(casas))
        eixo.axvline(args.margem_adotada, color=TEXTO_SECUNDARIO, linewidth=0.8, linestyle=":", zorder=1)

        if coluna != "dif_max":
            eixo.axhline(float(inteiro[coluna]), color=TEXTO_SECUNDARIO, linewidth=1, linestyle="--", zorder=1)

        eixo.set_ylim(bottom=0)

    eixos[0].set_ylim(top=1.1 * max(float(linha["dif_max"]) for linha in linhas))

    alcas += [
        Line2D([], [], color=TEXTO_SECUNDARIO, linewidth=1, linestyle="--", label="Quadro inteiro"),
        Line2D([], [], color=TEXTO_SECUNDARIO, linewidth=0.8, linestyle=":",
               label=f"Margem adotada ({args.margem_adotada} px)"),
    ]
    figura.legend(handles=alcas, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    figura.tight_layout()

    pasta = os.path.dirname(os.path.abspath(args.csv))
    print(f"Gerando grafico em {pasta}:")
    salvar(figura, pasta, "estudo_margem")


if __name__ == "__main__":
    main()
