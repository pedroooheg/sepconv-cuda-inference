"""
Gera os graficos e a tabela LaTeX de uma bateria de testes para o texto do TCC.

Le desempenho.csv, qualidade.csv, falhas.csv e quadros/*.csv da pasta da
bateria e grava em <pasta>/graficos/ (PDF vetorial + PNG 300 dpi):

  latencia.*        tempo por quadro interpolado x resolucao (escala log)
  vram.*            pico de VRAM do processo x resolucao
  etapas_cpp.*      distribuicao do tempo do motor C++ por estagio
  qualidade.*       PSNR, SSIM e VMAF medios por resolucao
  psnr_quadros.*    PSNR de cada quadro sintetizado (motor C++)
  tabela_resultados.tex

Uso:
    python implementacao/avaliacao/gerar_graficos.py resultados/testes_gta/<data_hora>
"""

import csv
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# =============================================================
# Estilo
# =============================================================

COR = {
    "cpp": "#2a78d6",      # azul
    "python": "#eb6834",   # laranja
}
NOME = {
    "cpp": "C++/CUDA (ONNX Runtime)",
    "python": "Python/PyTorch (original)",
}
# Hachura no Python para manter a distincao em impressao preto e branco.
HACHURA = {"cpp": "", "python": "///"}

CORES_ETAPAS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# Tons de azul do mais claro (720p) ao mais escuro (2160p).
CORES_RESOLUCAO = ["#9ec5f4", "#5c9eea", "#2a78d6", "#18508f"]

TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
GRADE = "#e4e3df"
SUPERFICIE = "#ffffff"

ORDEM = ["720p", "1080p", "1440p", "2160p"]
DIMENSOES = {"720p": "1280×720", "1080p": "1920×1080", "1440p": "2560×1440", "2160p": "3840×2160"}
LARGURA_FIGURA = 6.3  # polegadas (~16 cm, largura util de uma pagina A4 em ABNT)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": GRADE,
    "axes.labelcolor": TEXTO_SECUNDARIO,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.titlecolor": TEXTO,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRADE,
    "grid.linewidth": 0.6,
    "xtick.color": TEXTO_SECUNDARIO,
    "ytick.color": TEXTO_SECUNDARIO,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": SUPERFICIE,
    "savefig.facecolor": SUPERFICIE,
    "hatch.color": SUPERFICIE,
    "hatch.linewidth": 0.8,
})


def numero(valor, casas=0):
    """Formata no padrao brasileiro: 1.234,5"""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def formatador(casas=0):
    return FuncFormatter(lambda valor, _: numero(valor, casas))


def rotulo_resolucao(resolucao):
    return f"{resolucao}\n{DIMENSOES.get(resolucao, '')}"


def salvar(figura, pasta, nome):
    for extensao in ("pdf", "png"):
        figura.savefig(os.path.join(pasta, f"{nome}.{extensao}"), dpi=300, bbox_inches="tight")

    plt.close(figura)
    print(f"  {nome}.pdf / .png")


# =============================================================
# Leitura dos CSV
# =============================================================

def ler_csv(caminho):
    if not os.path.exists(caminho):
        return []

    with open(caminho, newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def resolucao_de(nome):
    encontrado = re.search(r"(\d+p)", nome)
    return encontrado.group(1) if encontrado else nome


def carregar(pasta):
    desempenho = {}

    for linha in ler_csv(os.path.join(pasta, "desempenho.csv")):
        desempenho[(resolucao_de(linha["entrada"]), linha["implementacao"])] = linha

    qualidade = {}

    for linha in ler_csv(os.path.join(pasta, "qualidade.csv")):
        qualidade[(resolucao_de(linha["referencia"]), linha["implementacao"])] = linha

    falhas = set()
    caminho_falhas = os.path.join(pasta, "falhas.csv")

    if os.path.exists(caminho_falhas):
        with open(caminho_falhas) as arquivo:
            for linha in arquivo:
                if linha.strip():
                    implementacao, resolucao = linha.strip().split(",")
                    falhas.add((resolucao, implementacao))

    quadros = {}

    for caminho in glob.glob(os.path.join(pasta, "quadros", "*.csv")):
        linhas = ler_csv(caminho)

        if linhas:
            chave = (resolucao_de(linhas[0]["referencia"]), linhas[0]["implementacao"])
            quadros[chave] = linhas

    resolucoes = sorted(
        {chave[0] for chave in list(desempenho) + list(falhas)},
        key=lambda r: ORDEM.index(r) if r in ORDEM else 99,
    )

    return desempenho, qualidade, falhas, quadros, resolucoes


def legenda_superior(eixo, alcas, colunas=None):
    eixo.legend(
        handles=alcas,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=colunas or len(alcas),
        borderaxespad=0,
        handlelength=1.6,
    )


# =============================================================
# Graficos
# =============================================================

def grafico_latencia(desempenho, falhas, resolucoes, pasta):
    figura, eixo = plt.subplots(figsize=(LARGURA_FIGURA, 3.4))
    posicoes = range(len(resolucoes))
    houve_saturacao = False
    houve_falha = False

    # Em cada resolucao, o rotulo do maior valor vai acima do ponto e o do
    # menor vai abaixo, para os dois nao se sobreporem.
    maior = {}

    for resolucao in resolucoes:
        valores = {i: float(desempenho[(resolucao, i)]["total_ms"])
                   for i in ("cpp", "python") if (resolucao, i) in desempenho}

        if valores:
            maior[resolucao] = max(valores, key=valores.get)

    for implementacao in ("cpp", "python"):
        xs, ys = [], []

        for x, resolucao in zip(posicoes, resolucoes):
            linha = desempenho.get((resolucao, implementacao))

            if linha is None:
                continue

            valor = float(linha["total_ms"])
            saturada = linha["vram_saturada"] == "1"
            houve_saturacao |= saturada
            xs.append(x)
            ys.append(valor)

            # Marcador vazado = medicao com a VRAM saturada.
            eixo.plot(
                x, valor, "o", markersize=8,
                markerfacecolor=SUPERFICIE if saturada else COR[implementacao],
                markeredgecolor=COR[implementacao], markeredgewidth=2, zorder=3,
            )

            deslocamento = 9 if maior.get(resolucao) == implementacao else -15
            eixo.annotate(
                f"{numero(valor / 1000, 1)} s" if valor >= 1000 else f"{numero(valor)} ms",
                (x, valor), textcoords="offset points", xytext=(0, deslocamento),
                ha="center", fontsize=8, color=TEXTO,
            )

        eixo.plot(xs, ys, color=COR[implementacao], linewidth=2, zorder=2)

    eixo.set_yscale("log")
    eixo.yaxis.set_major_formatter(formatador())
    eixo.set_xticks(list(posicoes), [rotulo_resolucao(r) for r in resolucoes])
    eixo.set_xlim(-0.4, len(resolucoes) - 0.6)
    eixo.set_ylabel("Tempo por quadro interpolado (ms, escala log)")

    # Falhas ficam numa faixa acima do maior valor medido.
    baixo, alto = eixo.get_ylim()
    topo = alto * 4
    eixo.set_ylim(baixo / 1.8, topo * 1.6)

    for x, resolucao in zip(posicoes, resolucoes):
        for implementacao in ("cpp", "python"):
            if (resolucao, implementacao) in falhas:
                houve_falha = True
                eixo.plot(x, topo, "X", markersize=9, color=COR[implementacao], zorder=3)
                nome = "C++" if implementacao == "cpp" else "Python"
                eixo.annotate(f"{nome}: falhou\n(sem memória)",
                              (x, topo), textcoords="offset points", xytext=(-9, 0),
                              ha="right", va="center", fontsize=8, color=TEXTO)

    alcas = [Line2D([], [], color=COR[i], linewidth=2, marker="o", markersize=6, label=NOME[i])
             for i in ("cpp", "python")]

    if houve_saturacao:
        alcas.append(Line2D([], [], linestyle="", marker="o", markersize=6, markerfacecolor=SUPERFICIE,
                            markeredgecolor=TEXTO_SECUNDARIO, markeredgewidth=1.5,
                            label="VRAM saturada"))

    if houve_falha:
        alcas.append(Line2D([], [], linestyle="", marker="X", markersize=7, color=TEXTO_SECUNDARIO,
                            label="Execução falhou"))

    legenda_superior(eixo, alcas, colunas=2)
    salvar(figura, pasta, "latencia")


def barras_agrupadas(eixo, resolucoes, valores, erros=None, casas=0, sufixo="", marcadas=()):
    """valores[(resolucao, implementacao)] -> numero ou None (ausente)."""
    largura = 0.36

    for indice, implementacao in enumerate(("cpp", "python")):
        deslocamento = (indice - 0.5) * largura

        for x, resolucao in enumerate(resolucoes):
            valor = valores.get((resolucao, implementacao))

            if valor is None:
                continue

            erro = erros.get((resolucao, implementacao)) if erros else None

            eixo.bar(
                x + deslocamento, valor, width=largura,
                color=COR[implementacao], hatch=HACHURA[implementacao],
                edgecolor=SUPERFICIE, linewidth=1.5,  # borda branca separa barras vizinhas
                yerr=erro, error_kw={"ecolor": TEXTO_SECUNDARIO, "elinewidth": 1, "capsize": 2},
            )
            eixo.annotate(
                numero(valor, casas) + sufixo + ("*" if (resolucao, implementacao) in marcadas else ""),
                (x + deslocamento, valor + (erro or 0)),
                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5, color=TEXTO,
                bbox={"facecolor": SUPERFICIE, "edgecolor": "none", "pad": 0.6}, zorder=4,
            )

    eixo.set_xticks(range(len(resolucoes)), [rotulo_resolucao(r) for r in resolucoes])
    eixo.set_xlim(-0.6, len(resolucoes) - 0.4)


def alcas_barras():
    return [Patch(facecolor=COR[i], hatch=HACHURA[i], edgecolor=SUPERFICIE, label=NOME[i])
            for i in ("cpp", "python")]


def grafico_vram(desempenho, falhas, resolucoes, pasta):
    figura, eixo = plt.subplots(figsize=(LARGURA_FIGURA, 3.4))

    valores = {
        chave: float(linha["vram_consumo_mib"]) for chave, linha in desempenho.items()
    }
    saturadas = {chave for chave, linha in desempenho.items() if linha["vram_saturada"] == "1"}
    barras_agrupadas(eixo, resolucoes, valores, marcadas=saturadas)

    # Memoria que o processo tinha de fato disponivel (capacidade - linha de base).
    exemplo = next(iter(desempenho.values()))
    disponivel = float(exemplo["vram_total_mib"]) - float(exemplo["vram_base_mib"])
    eixo.axhline(disponivel, color=TEXTO_SECUNDARIO, linewidth=1, linestyle="--", zorder=1)
    eixo.annotate(f"VRAM livre no início: {numero(disponivel)} MiB", (-0.45, disponivel),
                  textcoords="offset points", xytext=(0, 4), ha="left", fontsize=8, color=TEXTO_SECUNDARIO)

    for x, resolucao in enumerate(resolucoes):
        for indice, implementacao in enumerate(("cpp", "python")):
            if (resolucao, implementacao) in falhas:
                eixo.annotate("falhou", (x + (indice - 0.5) * 0.36, 0), textcoords="offset points",
                              xytext=(0, 4), ha="center", fontsize=7.5, color=TEXTO, rotation=90)

    eixo.set_ylim(0, max(disponivel, max(valores.values())) * 1.15)

    if saturadas:
        figura.text(0.0, -0.04, "* VRAM saturada: o excedente vai para a memória compartilhada do sistema e não "
                    "é medido pela NVML; o consumo real é maior que o valor mostrado.",
                    fontsize=7.5, color=TEXTO_SECUNDARIO, ha="left", va="top", wrap=True)
    eixo.yaxis.set_major_formatter(formatador())
    eixo.set_ylabel("Pico de VRAM do processo (MiB)")
    legenda_superior(eixo, alcas_barras())
    salvar(figura, pasta, "vram")


def grafico_etapas(desempenho, resolucoes, pasta):
    etapas = [
        ("upload_ms", "Host → Device"),
        ("pre_ms", "Pré-processamento"),
        ("inferencia_ms", "Inferência ONNX"),
        ("sepconv_ms", "Convolução sepconv"),
        ("download_ms", "Device → Host"),
    ]
    presentes = [r for r in resolucoes if (r, "cpp") in desempenho]

    if not presentes:
        return

    figura, eixo = plt.subplots(figsize=(LARGURA_FIGURA, 0.55 * len(presentes) + 1.2))

    for y, resolucao in enumerate(reversed(presentes)):
        linha = desempenho[(resolucao, "cpp")]
        tempos = [max(float(linha[coluna]), 0.0) for coluna, _ in etapas]
        total = sum(tempos)
        inicio = 0.0

        for cor, tempo in zip(CORES_ETAPAS, tempos):
            fracao = 100.0 * tempo / total
            eixo.barh(y, fracao, left=inicio, height=0.6, color=cor, edgecolor=SUPERFICIE, linewidth=1.5)

            if fracao >= 6:
                eixo.text(inicio + fracao / 2, y, f"{numero(fracao, 1)}%", ha="center", va="center",
                          fontsize=7.5, color=SUPERFICIE if cor in ("#2a78d6",) else TEXTO)

            inicio += fracao

    eixo.set_yticks(range(len(presentes)), list(reversed(presentes)))
    eixo.set_xlim(0, 100)
    eixo.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    eixo.set_xlabel("Parcela do tempo por quadro no motor C++")
    eixo.grid(axis="x")
    eixo.grid(axis="y", visible=False)
    eixo.spines["left"].set_visible(False)

    alcas = [Patch(facecolor=cor, label=nome) for cor, (_, nome) in zip(CORES_ETAPAS, etapas)]
    legenda_superior(eixo, alcas, colunas=3)
    salvar(figura, pasta, "etapas_cpp")


def grafico_qualidade(qualidade, resolucoes, pasta):
    metricas = [
        ("psnr", "PSNR (dB)", 2),
        ("ssim", "SSIM", 4),
        ("vmaf", "VMAF", 1),
    ]
    metricas = [m for m in metricas if any(linha.get(f"{m[0]}_media") for linha in qualidade.values())]
    presentes = [r for r in resolucoes if any((r, i) in qualidade for i in ("cpp", "python"))]

    if not metricas or not presentes:
        return

    # Uma metrica por linha, com a largura toda, para os rotulos caberem.
    figura, eixos = plt.subplots(len(metricas), 1, figsize=(LARGURA_FIGURA, 2.3 * len(metricas) + 0.4))
    eixos = eixos if len(metricas) > 1 else [eixos]

    for eixo, (metrica, titulo, casas) in zip(eixos, metricas):
        valores, erros = {}, {}

        for chave, linha in qualidade.items():
            if linha.get(f"{metrica}_media"):
                valores[chave] = float(linha[f"{metrica}_media"])
                erros[chave] = float(linha[f"{metrica}_desvio"])

        barras_agrupadas(eixo, presentes, valores, erros, casas=casas)
        eixo.set_title(titulo, loc="left")
        eixo.yaxis.set_major_formatter(formatador(2 if metrica == "ssim" else 0))
        eixo.set_ylim(0, (1.15 if metrica == "ssim" else 1.18) * max(v + erros[k] for k, v in valores.items()))

    figura.legend(handles=alcas_barras(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    figura.tight_layout()
    salvar(figura, pasta, "qualidade")


def grafico_psnr_quadros(quadros, resolucoes, pasta):
    series = [(r, quadros[(r, "cpp")]) for r in resolucoes if (r, "cpp") in quadros]

    if not series:
        return

    figura, eixo = plt.subplots(figsize=(LARGURA_FIGURA, 3.2))
    cores = CORES_RESOLUCAO[-len(series):] if len(series) < len(CORES_RESOLUCAO) else CORES_RESOLUCAO

    for cor, (resolucao, linhas) in zip(cores, series):
        xs = [int(linha["quadro"]) for linha in linhas]
        ys = [float(linha["psnr"]) for linha in linhas]
        eixo.plot(xs, ys, color=cor, linewidth=2, marker="o", markersize=4, label=resolucao)

    eixo.yaxis.set_major_formatter(formatador())
    eixo.set_xlabel("Índice do quadro sintetizado no vídeo de 30 FPS")
    eixo.set_ylabel("PSNR (dB)")
    eixo.grid(axis="x")
    legenda_superior(eixo, eixo.get_legend_handles_labels()[0])
    salvar(figura, pasta, "psnr_quadros")


# =============================================================
# Tabela LaTeX (booktabs, como no main.tex)
# =============================================================

def tabela_latex(desempenho, qualidade, falhas, resolucoes, pasta):
    linhas = [
        "% Gerado por implementacao/avaliacao/gerar_graficos.py",
        "\\begin{table}[htb]",
        "\\centering",
        "\\caption{Desempenho e fidelidade visual por resolução (interpolação 2$\\times$)}",
        "\\label{tab:resultados_gta}",
        "\\small",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "\\textbf{Resolução} & \\textbf{Implementação} & \\textbf{ms/quadro} & \\textbf{VRAM (MiB)} "
        "& \\textbf{PSNR (dB)} & \\textbf{SSIM} & \\textbf{VMAF} \\\\",
        "\\midrule",
    ]

    for resolucao in resolucoes:
        for implementacao in ("cpp", "python"):
            nome = "C++/CUDA" if implementacao == "cpp" else "Python/PyTorch"
            chave = (resolucao, implementacao)

            if chave in falhas:
                linhas.append(f"{resolucao} & {nome} & \\multicolumn{{5}}{{c}}{{falhou por falta de memória}} \\\\")
                continue

            if chave not in desempenho:
                continue

            linha = desempenho[chave]
            marca = "$^{*}$" if linha["vram_saturada"] == "1" else ""
            q = qualidade.get(chave, {})

            def campo(nome_campo, casas):
                return numero(float(q[nome_campo]), casas) if q.get(nome_campo) else "--"

            linhas.append(
                f"{resolucao} & {nome} & {numero(float(linha['total_ms']), 1)}{marca} & "
                f"{numero(float(linha['vram_consumo_mib']))} & {campo('psnr_media', 2)} & "
                f"{campo('ssim_media', 4)} & {campo('vmaf_media', 2)} \\\\"
            )

        if resolucao != resolucoes[-1]:
            linhas.append("\\midrule")

    linhas += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\legend{$^{*}$ VRAM saturada durante a execução. Fonte: Elaborado pelo autor (2026)}",
        "\\end{table}",
    ]

    with open(os.path.join(pasta, "tabela_resultados.tex"), "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas) + "\n")

    print("  tabela_resultados.tex")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    pasta_bateria = sys.argv[1]
    desempenho, qualidade, falhas, quadros, resolucoes = carregar(pasta_bateria)

    if not desempenho:
        raise SystemExit(f"Erro: nenhum dado de desempenho em {pasta_bateria}")

    pasta = os.path.join(pasta_bateria, "graficos")
    os.makedirs(pasta, exist_ok=True)
    print(f"Gerando graficos em {pasta}:")

    grafico_latencia(desempenho, falhas, resolucoes, pasta)
    grafico_vram(desempenho, falhas, resolucoes, pasta)
    grafico_etapas(desempenho, resolucoes, pasta)
    grafico_qualidade(qualidade, resolucoes, pasta)
    grafico_psnr_quadros(quadros, resolucoes, pasta)
    tabela_latex(desempenho, qualidade, falhas, resolucoes, pasta)


if __name__ == "__main__":
    main()
