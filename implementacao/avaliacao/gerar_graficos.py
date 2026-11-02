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

As series sao as implementacoes presentes nos CSV: cpp (quadro inteiro),
cpp_b<T> (motor em blocos T x T) e python.

Uso:
    python implementacao/avaliacao/gerar_graficos.py resultados/testes_gta/<data_hora>
"""

import csv
import glob
import math
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

COR_CPP = "#2a78d6"      # azul
COR_CPP_FP16 = "#18508f" # azul escuro: mesmo motor, quadro inteiro, em FP16
COR_PYTHON = "#eb6834"   # laranja
# Motor em blocos: cada tamanho de bloco tem sempre o mesmo estilo, em
# todos os graficos (outros tamanhos usam o estilo pela ordem).
CORES_BLOCOS = ["#1baf7a", "#eda100", "#e87ba4"]
# Hachuras mantem a distincao em impressao preto e branco.
HACHURAS_BLOCOS = ["...", "xxx", "\\\\\\"]
MARCADORES_BLOCOS = ["s", "^", "D"]
ESTILO_DO_BLOCO = {1024: 0, 512: 1, 256: 2}

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
# Series (implementacoes)
# =============================================================

def bloco_de(implementacao):
    encontrado = re.fullmatch(r"cpp_b(\d+)(?:_fp16)?", implementacao)
    return int(encontrado.group(1)) if encontrado else 0


def fp16_de(implementacao):
    return implementacao.startswith("cpp") and implementacao.endswith("_fp16")


def ordenar_series(implementacoes):
    """cpp e cpp_fp16 (quadro inteiro), cpp_b<T>[_fp16] do maior bloco ao
    menor, python."""
    def chave(implementacao):
        if not implementacao.startswith("cpp"):
            return (2, 0, 0)
        return (1 if bloco_de(implementacao) else 0, -bloco_de(implementacao), fp16_de(implementacao))

    return sorted(set(implementacoes), key=chave)


class Estilo:
    """Cor, hachura, marcador e nomes de cada serie."""

    def __init__(self, series):
        self.series = series
        blocos = [s for s in series if bloco_de(s)]
        self.indice_bloco = {
            s: ESTILO_DO_BLOCO.get(bloco_de(s), i % len(CORES_BLOCOS)) for i, s in enumerate(blocos)
        }
        # Com alguma serie em FP16, as demais series C++ sao rotuladas FP32.
        self.ha_fp16 = any(fp16_de(s) for s in series)

    def cor(self, serie):
        if serie == "cpp":
            return COR_CPP
        if serie == "cpp_fp16":
            return COR_CPP_FP16
        if serie in self.indice_bloco:
            return CORES_BLOCOS[self.indice_bloco[serie]]
        return COR_PYTHON

    def hachura(self, serie):
        if serie in self.indice_bloco:
            return HACHURAS_BLOCOS[self.indice_bloco[serie]]
        return "///" if serie == "python" else ""

    def marcador(self, serie):
        if serie in self.indice_bloco:
            return MARCADORES_BLOCOS[self.indice_bloco[serie]]
        return "p" if serie == "cpp_fp16" else "o"

    def precisao(self, serie):
        if fp16_de(serie):
            return " FP16"
        return " FP32" if self.ha_fp16 else ""

    def nome(self, serie):
        if not serie.startswith("cpp"):
            return "Python/PyTorch (original)"
        if bloco_de(serie):
            lado = bloco_de(serie)
            return f"C++/CUDA{self.precisao(serie)}, blocos {lado}×{lado}"
        if self.indice_bloco or self.ha_fp16:
            return f"C++/CUDA{self.precisao(serie)}, quadro inteiro"
        return "C++/CUDA (ONNX Runtime)"

    def nome_curto(self, serie):
        if not serie.startswith("cpp"):
            return "Python"
        return "C++" + self.precisao(serie) + (f" b{bloco_de(serie)}" if bloco_de(serie) else "")

    def nome_tabela(self, serie):
        if not serie.startswith("cpp"):
            return "Python/PyTorch"
        if not (self.indice_bloco or self.ha_fp16):
            return "C++/CUDA"
        regiao = f"blocos {bloco_de(serie)}" if bloco_de(serie) else "inteiro"
        return f"C++/CUDA{self.precisao(serie)} ({regiao})"


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
    series = ordenar_series(chave[1] for chave in list(desempenho) + list(falhas))

    return desempenho, qualidade, falhas, quadros, resolucoes, series


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

def espalhar_rotulos(figura, eixo, rotulos, distancia=12):
    """Rotulos (x, valor, texto, dx, alinhamento) num eixo log: do mesmo lado
    de um mesmo x, cada rotulo fica pelo menos `distancia` pontos acima do
    anterior."""
    baixo, alto = eixo.get_ylim()
    altura_pontos = eixo.get_position().height * figura.get_figheight() * 72
    pontos_por_decada = altura_pontos / math.log10(alto / baixo)

    grupos = {}

    for rotulo in rotulos:
        grupos.setdefault((rotulo[0], rotulo[3]), []).append(rotulo)

    for grupo in grupos.values():
        anterior = -math.inf

        for x, valor, texto, dx, alinhamento in sorted(grupo, key=lambda r: r[1]):
            posicao = math.log10(valor) * pontos_por_decada
            alvo = max(posicao, anterior + distancia)
            anterior = alvo
            eixo.annotate(texto, (x, valor), textcoords="offset points", xytext=(dx, alvo - posicao),
                          ha=alinhamento, va="center", fontsize=7.5, color=TEXTO)


def grafico_latencia(desempenho, falhas, resolucoes, estilo, pasta):
    series = estilo.series
    figura, eixo = plt.subplots(figsize=(LARGURA_FIGURA, 3.4 if len(series) <= 2 else 3.8))
    posicoes = range(len(resolucoes))
    houve_saturacao = False
    houve_falha = False

    # Posicao do rotulo de cada ponto. Com duas series, o maior valor vai
    # acima do ponto e o menor abaixo; com mais series, os rotulos alternam
    # entre a esquerda e a direita na ordem dos valores, para que valores
    # vizinhos nunca fiquem do mesmo lado, e sao afastados na vertical
    # depois que a escala do eixo esta definida (ver espalhar_rotulos).
    deslocamentos = {}
    rotulos = []

    for resolucao in resolucoes:
        valores = {s: float(desempenho[(resolucao, s)]["total_ms"])
                   for s in series if (resolucao, s) in desempenho}

        for posto, serie in enumerate(sorted(valores, key=valores.get)):
            if len(series) <= 2:
                acima = serie == max(valores, key=valores.get)
                deslocamentos[(resolucao, serie)] = ((0, 9 if acima else -15), "center", "baseline")
            elif posto % 2 == 0:
                deslocamentos[(resolucao, serie)] = ((-8, 0), "right", "center")
            else:
                deslocamentos[(resolucao, serie)] = ((8, 0), "left", "center")

    for serie in series:
        xs, ys = [], []

        for x, resolucao in zip(posicoes, resolucoes):
            linha = desempenho.get((resolucao, serie))

            if linha is None:
                continue

            valor = float(linha["total_ms"])
            saturada = linha["vram_saturada"] == "1"
            houve_saturacao |= saturada
            xs.append(x)
            ys.append(valor)

            # Marcador vazado = medicao com a VRAM saturada.
            eixo.plot(
                x, valor, estilo.marcador(serie), markersize=8 if estilo.marcador(serie) == "o" else 7,
                markerfacecolor=SUPERFICIE if saturada else estilo.cor(serie),
                markeredgecolor=estilo.cor(serie), markeredgewidth=2, zorder=3,
            )

            texto = f"{numero(valor / 1000, 1)} s" if valor >= 1000 else f"{numero(valor)} ms"
            deslocamento, horizontal, vertical = deslocamentos[(resolucao, serie)]

            if len(series) > 2:
                rotulos.append((x, valor, texto, deslocamento[0], horizontal))
                continue

            eixo.annotate(
                texto, (x, valor), textcoords="offset points", xytext=deslocamento,
                ha=horizontal, va=vertical, fontsize=8, color=TEXTO,
            )

        eixo.plot(xs, ys, color=estilo.cor(serie), linewidth=2, zorder=2,
                  linestyle="--" if bloco_de(serie) else "-")

    eixo.set_yscale("log")
    eixo.yaxis.set_major_formatter(formatador())
    eixo.set_xticks(list(posicoes), [rotulo_resolucao(r) for r in resolucoes])
    eixo.set_xlim(-0.6 if len(series) > 2 else -0.4, len(resolucoes) - (0.4 if len(series) > 2 else 0.6))
    eixo.set_ylabel("Tempo por quadro interpolado (ms, escala log)")

    # Falhas ficam numa faixa acima do maior valor medido.
    baixo, alto = eixo.get_ylim()
    topo = alto * 4
    eixo.set_ylim(baixo / 1.8, topo * 1.6)
    espalhar_rotulos(figura, eixo, rotulos)

    for x, resolucao in zip(posicoes, resolucoes):
        falharam = [s for s in series if (resolucao, s) in falhas]

        for indice, serie in enumerate(falharam):
            houve_falha = True
            eixo.plot(x, topo, "X", markersize=9, color=estilo.cor(serie), zorder=3)
            eixo.annotate(f"{estilo.nome_curto(serie)}: falhou\n(sem memória)",
                          (x, topo), textcoords="offset points", xytext=(-9, -22 * indice),
                          ha="right", va="center", fontsize=8, color=TEXTO)

    alcas = [Line2D([], [], color=estilo.cor(s), linewidth=2, marker=estilo.marcador(s), markersize=6,
                    linestyle="--" if bloco_de(s) else "-", label=estilo.nome(s))
             for s in series]

    if houve_saturacao:
        alcas.append(Line2D([], [], linestyle="", marker="o", markersize=6, markerfacecolor=SUPERFICIE,
                            markeredgecolor=TEXTO_SECUNDARIO, markeredgewidth=1.5,
                            label="VRAM saturada"))

    if houve_falha:
        alcas.append(Line2D([], [], linestyle="", marker="X", markersize=7, color=TEXTO_SECUNDARIO,
                            label="Execução falhou"))

    legenda_superior(eixo, alcas, colunas=2 if len(alcas) <= 4 else 3)
    salvar(figura, pasta, "latencia")


def largura_barra(estilo):
    return 0.72 if len(estilo.series) <= 2 else 0.84


def barras_agrupadas(eixo, resolucoes, valores, estilo, erros=None, casas=0, sufixo="", marcadas=()):
    """valores[(resolucao, implementacao)] -> numero ou None (ausente)."""
    series = estilo.series
    largura = largura_barra(estilo) / len(series)
    # Com muitas barras por grupo os rotulos ficam na vertical para caber.
    vertical = len(series) > 2

    for indice, serie in enumerate(series):
        deslocamento = (indice - (len(series) - 1) / 2) * largura

        for x, resolucao in enumerate(resolucoes):
            valor = valores.get((resolucao, serie))

            if valor is None:
                continue

            erro = erros.get((resolucao, serie)) if erros else None

            eixo.bar(
                x + deslocamento, valor, width=largura,
                color=estilo.cor(serie), hatch=estilo.hachura(serie),
                edgecolor=SUPERFICIE, linewidth=1.5,  # borda branca separa barras vizinhas
                yerr=erro, error_kw={"ecolor": TEXTO_SECUNDARIO, "elinewidth": 1, "capsize": 2},
            )
            eixo.annotate(
                numero(valor, casas) + sufixo + ("*" if (resolucao, serie) in marcadas else ""),
                (x + deslocamento, valor + (erro or 0)),
                textcoords="offset points", xytext=(0, 3), ha="center", va="bottom" if vertical else "baseline",
                rotation=90 if vertical else 0, fontsize=7 if vertical else 7.5, color=TEXTO,
                bbox={"facecolor": SUPERFICIE, "edgecolor": "none", "pad": 0.6}, zorder=4,
            )

    eixo.set_xticks(range(len(resolucoes)), [rotulo_resolucao(r) for r in resolucoes])
    eixo.set_xlim(-0.6, len(resolucoes) - 0.4)


def posicao_barra(indice, estilo):
    largura = largura_barra(estilo) / len(estilo.series)
    return (indice - (len(estilo.series) - 1) / 2) * largura


def alcas_barras(estilo):
    return [Patch(facecolor=estilo.cor(s), hatch=estilo.hachura(s), edgecolor=SUPERFICIE, label=estilo.nome(s))
            for s in estilo.series]


def grafico_vram(desempenho, falhas, resolucoes, estilo, pasta):
    figura, eixo = plt.subplots(figsize=(LARGURA_FIGURA, 3.4 if len(estilo.series) <= 2 else 3.8))

    valores = {
        chave: float(linha["vram_consumo_mib"]) for chave, linha in desempenho.items()
    }
    saturadas = {chave for chave, linha in desempenho.items() if linha["vram_saturada"] == "1"}
    barras_agrupadas(eixo, resolucoes, valores, estilo, marcadas=saturadas)

    # Memoria que o processo tinha de fato disponivel (capacidade - linha de base).
    exemplo = next(iter(desempenho.values()))
    disponivel = float(exemplo["vram_total_mib"]) - float(exemplo["vram_base_mib"])
    eixo.axhline(disponivel, color=TEXTO_SECUNDARIO, linewidth=1, linestyle="--", zorder=1)
    texto_disponivel = f"VRAM livre no início: {numero(disponivel)} MiB"
    alcas = alcas_barras(estilo)

    # Com muitas barras nao sobra espaco sobre a linha: o texto vai para a legenda.
    if len(estilo.series) > 2:
        alcas.append(Line2D([], [], color=TEXTO_SECUNDARIO, linewidth=1, linestyle="--", label=texto_disponivel))
    else:
        eixo.annotate(texto_disponivel, (-0.45, disponivel), textcoords="offset points", xytext=(0, 4),
                      ha="left", fontsize=8, color=TEXTO_SECUNDARIO)

    for x, resolucao in enumerate(resolucoes):
        for indice, serie in enumerate(estilo.series):
            if (resolucao, serie) in falhas:
                eixo.annotate("falhou", (x + posicao_barra(indice, estilo), 0), textcoords="offset points",
                              xytext=(0, 4), ha="center", fontsize=7.5, color=TEXTO, rotation=90)

    eixo.set_ylim(0, max(disponivel, max(valores.values())) * (1.15 if len(estilo.series) <= 2 else 1.25))

    if saturadas:
        figura.text(0.0, -0.04, "* VRAM saturada: o excedente vai para a memória compartilhada do sistema e não "
                    "é medido pela NVML; o consumo real é maior que o valor mostrado.",
                    fontsize=7.5, color=TEXTO_SECUNDARIO, ha="left", va="top", wrap=True)
    eixo.yaxis.set_major_formatter(formatador())
    eixo.set_ylabel("Pico de VRAM do processo (MiB)")
    legenda_superior(eixo, alcas, colunas=2 if len(estilo.series) > 2 else None)
    salvar(figura, pasta, "vram")


def grafico_etapas(desempenho, resolucoes, estilo, pasta):
    etapas = [
        ("upload_ms", "Host → Device"),
        ("pre_ms", "Pré-processamento"),
        ("inferencia_ms", "Inferência ONNX"),
        ("sepconv_ms", "Convolução sepconv"),
        ("download_ms", "Device → Host"),
    ]
    variantes = [s for s in estilo.series if s != "python"]
    # Uma barra por resolucao e variante do motor C++.
    barras = [(r, s) for r in resolucoes for s in variantes if (r, s) in desempenho]

    if not barras:
        return

    def rotulo(resolucao, serie):
        if len(variantes) == 1:
            return resolucao
        regiao = f"b{bloco_de(serie)}" if bloco_de(serie) else "inteiro"
        return f"{resolucao} · {regiao}" + (" FP16" if fp16_de(serie) else "")

    figura, eixo = plt.subplots(figsize=(LARGURA_FIGURA, (0.55 if len(variantes) == 1 else 0.45) * len(barras) + 1.2))

    for y, (resolucao, serie) in enumerate(reversed(barras)):
        linha = desempenho[(resolucao, serie)]
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

    eixo.set_yticks(range(len(barras)), [rotulo(*b) for b in reversed(barras)])
    eixo.set_xlim(0, 100)
    eixo.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    eixo.set_xlabel("Parcela do tempo por quadro no motor C++")
    eixo.grid(axis="x")
    eixo.grid(axis="y", visible=False)
    eixo.spines["left"].set_visible(False)

    if len(variantes) > 1:
        eixo.text(1.0, -0.12 - 0.3 / len(barras), "No motor em blocos, o pré-processamento inclui o cálculo "
                  "das estatísticas globais e os recortes dos blocos.", transform=eixo.transAxes,
                  ha="right", va="top", fontsize=7.5, color=TEXTO_SECUNDARIO)

    alcas = [Patch(facecolor=cor, label=nome) for cor, (_, nome) in zip(CORES_ETAPAS, etapas)]
    legenda_superior(eixo, alcas, colunas=3)
    salvar(figura, pasta, "etapas_cpp")


def grafico_qualidade(qualidade, resolucoes, estilo, pasta):
    metricas = [
        ("psnr", "PSNR (dB)", 2),
        ("ssim", "SSIM", 4),
        ("vmaf", "VMAF", 1),
    ]
    metricas = [m for m in metricas if any(linha.get(f"{m[0]}_media") for linha in qualidade.values())]
    presentes = [r for r in resolucoes if any((r, s) in qualidade for s in estilo.series)]

    if not metricas or not presentes:
        return

    # Uma metrica por linha, com a largura toda, para os rotulos caberem.
    altura = 2.3 if len(estilo.series) <= 2 else 2.6
    figura, eixos = plt.subplots(len(metricas), 1, figsize=(LARGURA_FIGURA, altura * len(metricas) + 0.4))
    eixos = eixos if len(metricas) > 1 else [eixos]

    for eixo, (metrica, titulo, casas) in zip(eixos, metricas):
        valores, erros = {}, {}

        for chave, linha in qualidade.items():
            if linha.get(f"{metrica}_media"):
                valores[chave] = float(linha[f"{metrica}_media"])
                erros[chave] = float(linha[f"{metrica}_desvio"])

        barras_agrupadas(eixo, presentes, valores, estilo, erros, casas=casas)
        eixo.set_title(titulo, loc="left")
        eixo.yaxis.set_major_formatter(formatador(2 if metrica == "ssim" else 0))
        folga = (1.15 if metrica == "ssim" else 1.18) if len(estilo.series) <= 2 else 1.35
        eixo.set_ylim(0, folga * max(v + erros[k] for k, v in valores.items()))

    figura.legend(handles=alcas_barras(estilo), loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  ncol=2 if len(estilo.series) > 2 else len(estilo.series))
    figura.tight_layout()
    salvar(figura, pasta, "qualidade")


def grafico_psnr_quadros(quadros, resolucoes, pasta):
    # Motor C++ com o quadro inteiro; onde ele falhou (4K), usa o motor em
    # blocos de maior bloco disponivel.
    series = []

    for resolucao in resolucoes:
        candidatas = ordenar_series(s for (r, s) in quadros if r == resolucao and s != "python")

        if candidatas:
            series.append((resolucao, candidatas[0], quadros[(resolucao, candidatas[0])]))

    if not series:
        return

    figura, eixo = plt.subplots(figsize=(LARGURA_FIGURA, 3.2))
    cores = CORES_RESOLUCAO[-len(series):] if len(series) < len(CORES_RESOLUCAO) else CORES_RESOLUCAO

    for cor, (resolucao, serie, linhas) in zip(cores, series):
        xs = [int(linha["quadro"]) for linha in linhas]
        ys = [float(linha["psnr"]) for linha in linhas]
        if serie == "cpp":
            rotulo = resolucao
        else:
            detalhes = ([f"blocos {bloco_de(serie)}"] if bloco_de(serie) else []) + (["FP16"] if fp16_de(serie) else [])
            rotulo = f"{resolucao} ({', '.join(detalhes)})"
        eixo.plot(xs, ys, color=cor, linewidth=2, marker="o", markersize=4, label=rotulo)

    eixo.yaxis.set_major_formatter(formatador())
    eixo.set_xlabel("Índice do quadro sintetizado no vídeo de 30 FPS")
    eixo.set_ylabel("PSNR (dB)")
    eixo.grid(axis="x")
    legenda_superior(eixo, eixo.get_legend_handles_labels()[0])
    salvar(figura, pasta, "psnr_quadros")


# =============================================================
# Tabela LaTeX (booktabs, como no main.tex)
# =============================================================

def tabela_latex(desempenho, qualidade, falhas, resolucoes, estilo, pasta):
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
        for serie in estilo.series:
            nome = estilo.nome_tabela(serie)
            chave = (resolucao, serie)

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

    margens = {linha["margem"] for linha in desempenho.values() if bloco_de(linha["implementacao"])}
    nota_blocos = f" Motor em blocos com margem de {'/'.join(sorted(margens))} pixels." if margens else ""

    linhas += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\legend{{$^{{*}}$ VRAM saturada durante a execução.{nota_blocos} Fonte: Elaborado pelo autor (2026)}}",
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
    desempenho, qualidade, falhas, quadros, resolucoes, series = carregar(pasta_bateria)

    if not desempenho:
        raise SystemExit(f"Erro: nenhum dado de desempenho em {pasta_bateria}")

    estilo = Estilo(series)
    pasta = os.path.join(pasta_bateria, "graficos")
    os.makedirs(pasta, exist_ok=True)
    print(f"Gerando graficos em {pasta} (series: {', '.join(series)}):")

    grafico_latencia(desempenho, falhas, resolucoes, estilo, pasta)
    grafico_vram(desempenho, falhas, resolucoes, estilo, pasta)
    grafico_etapas(desempenho, resolucoes, estilo, pasta)
    grafico_qualidade(qualidade, resolucoes, estilo, pasta)
    grafico_psnr_quadros(quadros, resolucoes, pasta)
    tabela_latex(desempenho, qualidade, falhas, resolucoes, estilo, pasta)


if __name__ == "__main__":
    main()
