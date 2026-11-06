"""
Figuras dos capitulos de introducao e fundamentos do TCC, a partir de imagens
reais (sequencia Beanbags do Middlebury) e dos filtros estimados pelo modelo.

Grava em resultados/figuras_tcc/fundamentos/ (PDF + PNG 300 dpi) e imprime os
valores citados no texto. Os filtros reais exigem a GPU e os modelos ONNX; as
demais figuras usam apenas a CPU.

Uso:
    python implementacao/figuras/gerar_figuras_fundamentos.py [nome ...]
    python implementacao/figuras/gerar_figuras_fundamentos.py vfi_metodos metricas
"""

import glob
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RAIZ, "implementacao", "avaliacao"))

import cv2
import matplotlib.pyplot as plt
import numpy
import skimage.metrics
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from gerar_graficos import (
    COR_CPP, COR_PYTHON, GRADE, LARGURA_FIGURA, SUPERFICIE, TEXTO, TEXTO_SECUNDARIO,
    formatador, numero, salvar,
)

CENA = os.path.join(RAIZ, "original", "middlebury", "Beanbags")
DESTINO = os.path.join(RAIZ, "resultados", "figuras_tcc", "fundamentos")
VERDE, AMARELO = "#1baf7a", "#eda100"

# Regiao com as bolas e a mao: (x0, y0, x1, y1)
RECORTE = (100, 160, 260, 280)
# Regiao maior, com as bolas, a mao e o texto da camiseta.
REGIAO_DETALHE = (110, 170, 370, 365)


def ler(nome):
    """Imagem da cena em RGB, para exibicao."""
    return cv2.cvtColor(cv2.imread(os.path.join(CENA, nome)), cv2.COLOR_BGR2RGB)


def saida_sepconv():
    """Quadro sintetizado pela implementacao original (avaliar_conjuntos.py)."""
    caminhos = sorted(glob.glob(os.path.join(RAIZ, "resultados", "conjuntos", "*", "saidas", "python", "middlebury", "Beanbags.png")))

    if not caminhos:
        raise SystemExit("Saida do Beanbags ausente: rode implementacao/avaliacao/avaliar_conjuntos.py")

    return cv2.cvtColor(cv2.imread(caminhos[-1]), cv2.COLOR_BGR2RGB)


def recortar(imagem, regiao=RECORTE):
    x0, y0, x1, y1 = regiao
    return imagem[y0:y1, x0:x1]


def psnr(real, teste):
    return skimage.metrics.peak_signal_noise_ratio(real, teste, data_range=255)


def ssim(real, teste):
    return skimage.metrics.structural_similarity(real, teste, data_range=255, channel_axis=2)


def painel(eixo, imagem, titulo=None, cmap=None):
    eixo.imshow(imagem, cmap=cmap, interpolation="nearest")
    eixo.set_xticks([])
    eixo.set_yticks([])
    eixo.grid(False)

    for lado in eixo.spines.values():
        lado.set_visible(False)

    if titulo:
        eixo.set_title(titulo, fontsize=8.5, fontweight="normal", color=TEXTO)


def caixa(eixo, x, y, largura, altura, texto, cor=SUPERFICIE, borda=TEXTO_SECUNDARIO, tamanho=8, cor_texto=TEXTO):
    eixo.add_patch(FancyBboxPatch((x, y), largura, altura, boxstyle="round,pad=0.02,rounding_size=0.08",
                                  facecolor=cor, edgecolor=borda, linewidth=1))
    eixo.text(x + largura / 2, y + altura / 2, texto, ha="center", va="center", fontsize=tamanho, color=cor_texto)


def seta(eixo, inicio, fim, cor=TEXTO_SECUNDARIO, texto=None, deslocamento=0.12, duas=False):
    eixo.add_patch(FancyArrowPatch(inicio, fim, arrowstyle="<|-|>" if duas else "-|>", mutation_scale=10,
                                   color=cor, linewidth=1.2, shrinkA=0, shrinkB=0))

    if texto:
        eixo.text((inicio[0] + fim[0]) / 2, (inicio[1] + fim[1]) / 2 + deslocamento, texto, ha="center", va="bottom",
                  fontsize=7.5, color=TEXTO_SECUNDARIO)


def tela(largura, altura, xmax, ymax):
    figura, eixo = plt.subplots(figsize=(largura, altura))
    eixo.set_xlim(0, xmax)
    eixo.set_ylim(0, ymax)
    eixo.axis("off")
    return figura, eixo


# =============================================================
# Capitulo 1
# =============================================================

def vfi_sequencia():
    figura, eixos = plt.subplots(1, 3, figsize=(LARGURA_FIGURA, 1.85))
    rotulos = [("frame10.png", "Quadro real no instante $t$"),
               ("frame10i11.png", "Quadro intermediário, $t + 0{,}5$"),
               ("frame11.png", "Quadro real no instante $t + 1$")]

    for eixo, (nome, titulo) in zip(eixos, rotulos):
        painel(eixo, ler(nome), titulo)

    for lado in eixos[1].spines.values():
        lado.set_visible(True)
        lado.set_edgecolor(COR_CPP)
        lado.set_linewidth(2)

    figura.tight_layout(pad=0.3)
    salvar(figura, DESTINO, "vfi_sequencia")


def vfi_metodos():
    um, real, dois = ler("frame10.png"), ler("frame10i11.png"), ler("frame11.png")
    media = ((um.astype(numpy.float32) + dois.astype(numpy.float32)) / 2).astype(numpy.uint8)
    sepconv = saida_sepconv()

    metodos = [("(a) Repetição", um), ("(b) Mistura linear", media), ("(c) SepConv++", sepconv), ("(d) Quadro real", real)]
    figura, eixos = plt.subplots(1, 4, figsize=(LARGURA_FIGURA, 1.75))

    for eixo, (titulo, imagem) in zip(eixos, metodos):
        rodape = "" if imagem is real else f"\n{numero(psnr(real, imagem), 2)} dB"
        painel(eixo, recortar(imagem), titulo + rodape)
        print(f"  {titulo}: PSNR da imagem inteira = {psnr(real, imagem):.2f} dB" if imagem is not real else "")

    figura.tight_layout(pad=0.3)
    salvar(figura, DESTINO, "vfi_metodos")


# =============================================================
# Capitulo 2: imagem digital e organizacao na memoria
# =============================================================

def imagem_digital():
    imagem = ler("frame10.png")
    x0, y0, lado = 212, 196, 8
    recorte = imagem[y0:y0 + lado, x0:x0 + lado]

    figura = plt.figure(figsize=(LARGURA_FIGURA, 2.3))
    grade = figura.add_gridspec(1, 2, left=0.01, right=0.79, top=0.88, bottom=0.03, width_ratios=[1.3, 1.0], wspace=0.08)

    eixo = figura.add_subplot(grade[0])
    painel(eixo, imagem, "(a) Imagem de 640 × 480 pixels")
    eixo.add_patch(Rectangle((x0 - 3, y0 - 3), lado + 6, lado + 6, fill=False, edgecolor="#ffffff", linewidth=1.2))

    eixo = figura.add_subplot(grade[1])
    painel(eixo, recorte, "(b) 8 × 8 pixels, valores do canal G")

    for y in range(lado):
        for x in range(lado):
            valor = int(recorte[y, x, 1])
            brilho = recorte[y, x].mean()
            eixo.text(x, y, str(valor), ha="center", va="center", fontsize=5.8, color="#000000" if brilho > 110 else "#ffffff")

    for i in range(lado + 1):
        eixo.axhline(i - 0.5, color="#ffffff", linewidth=0.3)
        eixo.axvline(i - 0.5, color="#ffffff", linewidth=0.3)

    canais = figura.add_gridspec(3, 1, left=0.82, right=0.99, top=0.88, bottom=0.03, hspace=0.42)

    for linha, (canal, nome) in enumerate([(0, "(c) Canal R"), (1, "Canal G"), (2, "Canal B")]):
        painel(figura.add_subplot(canais[linha]), imagem[:, :, canal], nome, cmap="gray")

    salvar(figura, DESTINO, "imagem_digital")


def linearizacao():
    figura, eixo = tela(LARGURA_FIGURA, 2.55, 16, 6.4)

    eixo.text(1.75, 5.95, "(a) Matriz 2D", ha="center", fontsize=8.5, color=TEXTO)
    cores = ["#dfe9f7", "#c0d6f0", "#a0c2ea", "#80aee3"]

    for y in range(4):
        for x in range(4):
            eixo.add_patch(Rectangle((0.2 + x * 0.78, 5.45 - (y + 1) * 0.78), 0.78, 0.78, facecolor=cores[y], edgecolor=SUPERFICIE))
            eixo.text(0.59 + x * 0.78, 5.06 - y * 0.78, f"{y},{x}", ha="center", va="center", fontsize=6.5, color=TEXTO)

    eixo.text(1.75, 1.8, "linha, coluna", ha="center", fontsize=7, color=TEXTO_SECUNDARIO)
    seta(eixo, (3.55, 3.75), (4.8, 3.75))
    eixo.text(4.17, 3.28, "$i = y \\cdot W + x$", ha="center", va="top", fontsize=7, color=TEXTO_SECUNDARIO)

    eixo.text(10.4, 5.95, "(b) Vetor contíguo na memória", ha="center", fontsize=8.5, color=TEXTO)

    for i in range(16):
        y, x = divmod(i, 4)
        eixo.add_patch(Rectangle((4.9 + i * 0.68, 4.6), 0.68, 0.78, facecolor=cores[y], edgecolor=SUPERFICIE))
        eixo.text(5.24 + i * 0.68, 4.99, str(i), ha="center", va="center", fontsize=6.5, color=TEXTO)

    eixo.text(10.4, 3.2, "(c) Organização dos canais de cor", ha="center", fontsize=8.5, color=TEXTO)
    canais = {"B": "#8fb8ee", "G": "#9edcc0", "R": "#f3a98c"}

    eixo.text(4.82, 2.34, "OpenCV", ha="right", va="center", fontsize=7.5, color=TEXTO_SECUNDARIO)

    for i, canal in enumerate("BGR" * 4):
        eixo.add_patch(Rectangle((4.9 + i * 0.9, 1.95), 0.9, 0.78, facecolor=canais[canal], edgecolor=SUPERFICIE))
        eixo.text(5.35 + i * 0.9, 2.34, canal, ha="center", va="center", fontsize=7, color=TEXTO)

    eixo.text(4.82, 0.74, "Rede", ha="right", va="center", fontsize=7.5, color=TEXTO_SECUNDARIO)

    for i, canal in enumerate("BBBBGGGGRRRR"):
        eixo.add_patch(Rectangle((4.9 + i * 0.9, 0.35), 0.9, 0.78, facecolor=canais[canal], edgecolor=SUPERFICIE))
        eixo.text(5.35 + i * 0.9, 0.74, canal, ha="center", va="center", fontsize=7, color=TEXTO)

    seta(eixo, (10.3, 1.92), (10.3, 1.16), texto=None)
    eixo.text(10.5, 1.5, "pré-processamento na GPU", ha="left", va="center", fontsize=7, color=TEXTO_SECUNDARIO)

    salvar(figura, DESTINO, "linearizacao")


# =============================================================
# Capitulo 2: convolucao, fluxo optico e filtros
# =============================================================

def filtragem():
    imagem = cv2.cvtColor(ler("frame10.png"), cv2.COLOR_RGB2GRAY)
    regiao = REGIAO_DETALHE
    gauss = cv2.getGaussianKernel(15, 4)
    gauss = gauss @ gauss.T
    suave = cv2.filter2D(imagem, -1, gauss)
    sobel = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=numpy.float32)
    sx = cv2.filter2D(imagem, cv2.CV_32F, sobel)
    bordas = numpy.clip(numpy.abs(sx), 0, 255).astype(numpy.uint8)

    figura = plt.figure(figsize=(LARGURA_FIGURA, 3.2))
    grade = figura.add_gridspec(2, 3, height_ratios=[1.0, 2.5], hspace=0.28, wspace=0.08)

    colunas = [
        ("Identidade, 3 × 3", numpy.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), "(a) Imagem original", imagem),
        ("Gaussiano, 15 × 15, $\\sigma = 4$", gauss, "(b) Suavização", suave),
        ("Sobel horizontal, 3 × 3", sobel, "(c) Bordas verticais", bordas),
    ]

    for coluna, (nome, kernel, titulo, resultado) in enumerate(colunas):
        eixo = figura.add_subplot(grade[0, coluna])
        limite = numpy.abs(kernel).max()
        painel(eixo, kernel, nome, cmap="RdBu_r")
        eixo.images[0].set_clim(-limite, limite)

        if kernel.shape[0] == 3:
            for y in range(3):
                for x in range(3):
                    forte = abs(kernel[y, x]) > 0.6 * limite
                    eixo.text(x, y, f"{kernel[y, x]:g}", ha="center", va="center", fontsize=7,
                              color="#ffffff" if forte else TEXTO)

        painel(figura.add_subplot(grade[1, coluna]), recortar(resultado, regiao), titulo, cmap="gray")

    salvar(figura, DESTINO, "filtragem")


def _fluxo_raft():
    """Fluxo optico do quadro t para t + 1, com o RAFT do torchvision."""
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    pesos = Raft_Large_Weights.DEFAULT
    modelo = raft_large(weights=pesos).cuda().eval()

    def entrada(nome):
        return torch.from_numpy(ler(nome)).permute(2, 0, 1)[None]

    um, dois = pesos.transforms()(entrada("frame10.png"), entrada("frame11.png"))

    with torch.no_grad():
        return modelo(um.cuda(), dois.cuda())[-1][0].permute(1, 2, 0).cpu().numpy()


def fluxo_optico():
    fluxo = _fluxo_raft()
    um = cv2.cvtColor(ler("frame10.png"), cv2.COLOR_RGB2GRAY)

    magnitude, angulo = cv2.cartToPolar(fluxo[..., 0], fluxo[..., 1])
    hsv = numpy.zeros((*um.shape, 3), dtype=numpy.uint8)
    hsv[..., 0] = angulo * 180 / numpy.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = numpy.clip(magnitude * 255 / max(numpy.percentile(magnitude, 99.5), 1e-3), 0, 255)
    cores = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    figura, eixos = plt.subplots(1, 2, figsize=(LARGURA_FIGURA, 2.25))
    painel(eixos[0], ler("frame10.png"), "(a) Vetores de deslocamento $(u, v)$")

    passo = 24
    ys, xs = numpy.mgrid[passo // 2:um.shape[0]:passo, passo // 2:um.shape[1]:passo]
    us, vs = fluxo[ys, xs, 0], fluxo[ys, xs, 1]
    visiveis = numpy.hypot(us, vs) > 1.0
    eixos[0].quiver(xs[visiveis], ys[visiveis], us[visiveis], vs[visiveis], color="#ffffff", angles="xy",
                    scale_units="xy", scale=0.4, width=0.004, headwidth=4)

    painel(eixos[1], cores, "(b) Direção (cor) e magnitude (brilho) do fluxo")
    figura.tight_layout(pad=0.3)
    salvar(figura, DESTINO, "fluxo_optico")
    print(f"  deslocamento maximo (percentil 99,5): {numpy.percentile(magnitude, 99.5):.1f} pixels")
    x, y = 222, 213  # bola, no quadro t
    print(f"  fluxo na bola ({x},{y}): u={fluxo[y, x, 0]:.1f}, v={fluxo[y, x, 1]:.1f}")


def memoria_filtros():
    resolucoes = [("720p", 1280, 720), ("1080p", 1920, 1080), ("1440p", 2560, 1440), ("2160p", 3840, 2160)]
    gib = 1024.0 ** 3
    bidimensional = [w * h * 51 * 51 * 2 * 4 / gib for _, w, h in resolucoes]
    separavel = [w * h * 51 * 4 * 4 / gib for _, w, h in resolucoes]

    figura, eixo = plt.subplots(figsize=(LARGURA_FIGURA, 2.75))
    largura = 0.36

    for i, (valores, cor, hachura, nome) in enumerate([
        (bidimensional, COR_PYTHON, "///", "Filtros 2D: 2.601 coeficientes por pixel"),
        (separavel, COR_CPP, "", "Filtros separáveis: 102 coeficientes por pixel"),
    ]):
        posicoes = [x + (i - 0.5) * largura for x in range(len(resolucoes))]
        eixo.bar(posicoes, valores, width=largura, color=cor, hatch=hachura, edgecolor=SUPERFICIE, linewidth=1.5, label=nome)

        for x, valor in zip(posicoes, valores):
            eixo.annotate(f"{numero(valor, 1)} GiB", (x, valor), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5,
                          color=TEXTO, bbox={"facecolor": SUPERFICIE, "edgecolor": "none", "pad": 0.6}, zorder=4)

    eixo.axhline(8188 / 1024, color=TEXTO_SECUNDARIO, linewidth=1, linestyle="--", label="VRAM da GPU: 8 GB", zorder=1)
    eixo.set_yscale("log")
    eixo.set_ylim(0.3, 700)
    eixo.yaxis.set_major_formatter(formatador())
    eixo.set_xticks(range(len(resolucoes)), [r[0] for r in resolucoes])
    eixo.set_xlim(-0.6, len(resolucoes) - 0.4)
    eixo.set_ylabel("Memória dos filtros (GiB, escala log)")
    eixo.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, borderaxespad=0, handlelength=1.6, fontsize=7.5)
    salvar(figura, DESTINO, "memoria_filtros")

    for (nome, _, _), a, b in zip(resolucoes, bidimensional, separavel):
        print(f"  {nome}: 2D {a:.1f} GiB, separavel {b:.2f} GiB")


def _filtros_da_rede():
    """Filtros (v1, v2, h1, h2) do par Beanbags, no modelo ONNX em FP32."""
    import onnxruntime

    um = cv2.imread(os.path.join(CENA, "frame10.png"))
    dois = cv2.imread(os.path.join(CENA, "frame11.png"))

    def entrada(quadro):
        return numpy.ascontiguousarray(quadro.transpose(2, 0, 1)[None].astype(numpy.float32) * (1.0 / 255.0))

    sessao = onnxruntime.InferenceSession(os.path.join(RAIZ, "modelos", "sepconv_paper.onnx"),
                                          providers=["CUDAExecutionProvider"])
    saidas = sessao.run(None, {"input_frame1": entrada(um), "input_frame2": entrada(dois)})
    return [s[0] for s in saidas], entrada(um)[0], entrada(dois)[0]


def _pixel_sintetizado(quadros, filtros, x, y):
    """Recalcula um pixel como o kernel do motor, para conferir os filtros exibidos."""
    v1, v2, h1, h2 = filtros
    acumulado = numpy.zeros(3)
    soma = 0.0

    for quadro, ver, hor in ((quadros[0], v1, h1), (quadros[1], v2, h2)):
        v, h = ver[:, y, x], hor[:, y, x]
        linhas = numpy.clip(y + numpy.arange(51) - 25, 0, quadro.shape[1] - 1)
        colunas = numpy.clip(x + numpy.arange(51) - 25, 0, quadro.shape[2] - 1)
        acumulado += ((v[:, None] * h[None, :])[None] * quadro[:, linhas][:, :, colunas]).sum((1, 2))
        soma += v.sum() * h.sum()

    return numpy.clip(acumulado / soma, 0, 1) * 255.0


def filtro_adaptativo():
    filtros, um, dois = _filtros_da_rede()
    v1, v2, h1, h2 = filtros
    saida = cv2.imread(glob.glob(os.path.join(RAIZ, "resultados", "conjuntos", "*", "saidas", "python", "middlebury", "Beanbags.png"))[-1])

    # Na bola (com movimento) e na porta (parada), no quadro intermediario.
    pontos = [("Pixel na bola em movimento", 227, 195, COR_PYTHON), ("Pixel em região parada", 340, 90, COR_CPP)]

    figura = plt.figure(figsize=(LARGURA_FIGURA, 3.55))
    grade = figura.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], hspace=0.3, wspace=0.1)
    imagem = ler("frame10i11.png")

    for linha, (nome, x, y, cor) in enumerate(pontos):
        calculado = _pixel_sintetizado((um, dois), filtros, x, y)
        print(f"  {nome} ({x},{y}): recalculado {calculado.round(1)}, saida {saida[y, x]}")

        # Pesos efetivos: divididos pela norma da sintese, somam 1 nos dois quadros.
        k1 = v1[:, y, x][:, None] * h1[:, y, x][None, :]
        k2 = v2[:, y, x][:, None] * h2[:, y, x][None, :]
        norma = k1.sum() + k2.sum()
        k1, k2 = k1 / norma, k2 / norma
        limite = max(numpy.abs(k1).max(), numpy.abs(k2).max())
        print(f"    soma dos pesos: quadro t {k1.sum():.3f}, quadro t+1 {k2.sum():.3f}")

        eixo = figura.add_subplot(grade[linha, 0])
        painel(eixo, imagem, f"({'ab'[linha]}) {nome}")
        eixo.plot(x, y, "o", markersize=8, markerfacecolor="none", markeredgecolor="#ffffff", markeredgewidth=1.6)
        eixo.plot(x, y, "o", markersize=2, color=cor)

        for coluna, (k, titulo) in enumerate([(k1, "Filtro no quadro $t$"), (k2, "Filtro no quadro $t + 1$")]):
            eixo = figura.add_subplot(grade[linha, coluna + 1])
            painel(eixo, k, titulo, cmap="RdBu_r")
            eixo.images[0].set_clim(-limite, limite)
            eixo.axhline(25, color=TEXTO_SECUNDARIO, linewidth=0.5, linestyle=":")
            eixo.axvline(25, color=TEXTO_SECUNDARIO, linewidth=0.5, linestyle=":")

            pesos = numpy.abs(k)
            cy, cx = numpy.unravel_index(numpy.argmax(pesos), pesos.shape)
            print(f"    {titulo}: maior peso em dx={cx - 25}, dy={cy - 25}")

    salvar(figura, DESTINO, "filtro_adaptativo")


def separabilidade():
    filtros, _, _ = _filtros_da_rede()
    v1, _, h1, _ = filtros
    x, y = 227, 195
    # O sinal de v e h se cancela na normalizacao da sintese: ambos sao exibidos
    # com soma positiva, para que o filtro 2D mostre os pesos efetivos.
    v, h = v1[:, y, x], h1[:, y, x]
    v, h = v * numpy.sign(v.sum()), h * numpy.sign(h.sum())
    k = v[:, None] * h[None, :]
    print(f"  posto numerico do filtro: {numpy.linalg.matrix_rank(k)}")

    figura = plt.figure(figsize=(LARGURA_FIGURA, 2.65))
    grade = figura.add_gridspec(2, 3, width_ratios=[0.34, 0.1, 1.0], height_ratios=[0.34, 1.0], hspace=0.08, wspace=0.08,
                                left=0.03, right=0.72, top=0.86, bottom=0.05)
    posicoes = numpy.arange(51) - 25

    eixo_h = figura.add_subplot(grade[0, 2])
    eixo_h.bar(posicoes, h, color=COR_PYTHON, width=0.9)
    eixo_h.set_xlim(-25.5, 25.5)
    eixo_h.set_xticks([])
    eixo_h.set_yticks([])
    eixo_h.grid(False)
    eixo_h.set_title("$h$: 51 coeficientes", fontsize=8.5, fontweight="normal", color=TEXTO)

    eixo_v = figura.add_subplot(grade[1, 0])
    eixo_v.barh(posicoes, v, color=COR_CPP, height=0.9)
    eixo_v.set_ylim(25.5, -25.5)
    eixo_v.invert_xaxis()
    eixo_v.set_xticks([])
    eixo_v.set_yticks([])
    eixo_v.grid(False)
    eixo_v.set_ylabel("$v$: 51 coeficientes", fontsize=8.5, color=TEXTO)

    eixo_k = figura.add_subplot(grade[1, 2])
    painel(eixo_k, k, None, cmap="RdBu_r")
    limite = numpy.abs(k).max()
    eixo_k.images[0].set_clim(-limite, limite)
    eixo_k.set_aspect("auto")

    for eixo in (eixo_h, eixo_v):
        for lado in eixo.spines.values():
            lado.set_visible(False)

    figura.text(0.86, 0.72, "Armazenado", ha="center", fontsize=8.5, color=TEXTO_SECUNDARIO)
    figura.text(0.86, 0.60, "51 + 51 = 102", ha="center", fontsize=11, fontweight="semibold", color=TEXTO)
    figura.text(0.86, 0.47, "coeficientes", ha="center", fontsize=8.5, color=TEXTO_SECUNDARIO)
    figura.text(0.86, 0.30, "Filtro 2D implícito", ha="center", fontsize=8.5, color=TEXTO_SECUNDARIO)
    figura.text(0.86, 0.18, "51 × 51 = 2.601", ha="center", fontsize=11, fontweight="semibold", color=TEXTO)
    figura.text(0.86, 0.05, "$K = v \\cdot h^{T}$", ha="center", fontsize=9, color=TEXTO_SECUNDARIO)
    salvar(figura, DESTINO, "separabilidade")


# =============================================================
# Capitulo 2: GPU
# =============================================================

def host_device():
    figura, eixo = tela(LARGURA_FIGURA, 2.2, 16, 5.6)

    eixo.add_patch(FancyBboxPatch((0.2, 0.25), 5.6, 4.95, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor="#f4f3ef", edgecolor=GRADE))
    eixo.text(3.0, 4.75, "Host", ha="center", fontsize=9, fontweight="semibold", color=TEXTO)
    caixa(eixo, 0.7, 2.75, 4.6, 1.3, "CPU\nleitura e gravação do vídeo", tamanho=7)
    caixa(eixo, 0.7, 0.75, 4.6, 1.3, "RAM (16 GB)\nquadros de entrada e de saída", cor="#fbeadf", borda=COR_PYTHON, tamanho=7)
    seta(eixo, (3.0, 2.05), (3.0, 2.75), duas=True)

    eixo.add_patch(FancyBboxPatch((10.2, 0.25), 5.6, 4.95, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor="#f4f3ef", edgecolor=GRADE))
    eixo.text(13.0, 4.75, "Device (GPU)", ha="center", fontsize=9, fontweight="semibold", color=TEXTO)
    caixa(eixo, 10.7, 2.75, 4.6, 1.3, "Núcleos da GPU\nrede e síntese do quadro", tamanho=7)
    caixa(eixo, 10.7, 0.75, 4.6, 1.3, "VRAM (8 GB)\nquadros, ativações e filtros", cor="#dfe9f7", borda=COR_CPP, tamanho=7)
    seta(eixo, (13.0, 2.05), (13.0, 2.75), duas=True)
    eixo.text(13.25, 2.4, "banda alta", ha="left", va="center", fontsize=7, color=TEXTO_SECUNDARIO)

    seta(eixo, (5.3, 1.75), (10.7, 1.75), cor=COR_CPP, texto="quadros de entrada")
    seta(eixo, (10.7, 1.1), (5.3, 1.1), cor=COR_PYTHON, texto=None)
    eixo.text(8.0, 0.55, "quadro final", ha="center", fontsize=7.5, color=TEXTO_SECUNDARIO)
    eixo.text(8.0, 2.65, "Barramento PCIe", ha="center", fontsize=8.5, fontweight="semibold", color=TEXTO)
    eixo.text(8.0, 3.15, "gargalo", ha="center", fontsize=7.5, color=TEXTO_SECUNDARIO)

    salvar(figura, DESTINO, "host_device")


def simt():
    imagem = ler("frame10.png")
    x0, y0, largura, altura = 96, 160, 128, 64
    regiao = imagem[y0:y0 + altura, x0:x0 + largura]

    figura = plt.figure(figsize=(LARGURA_FIGURA, 1.6))
    grade = figura.add_gridspec(1, 3, width_ratios=[1.0, 1.55, 1.55], wspace=0.08, left=0.01, right=0.99, top=0.84, bottom=0.03)

    eixo = figura.add_subplot(grade[0])
    painel(eixo, imagem, "(a) Grade sobre a imagem")
    eixo.add_patch(Rectangle((x0, y0), largura, altura, fill=False, edgecolor="#ffffff", linewidth=1.2))

    eixo = figura.add_subplot(grade[1])
    painel(eixo, regiao, "(b) Blocos de 32 × 8 threads")

    for x in range(0, largura + 1, 32):
        eixo.axvline(x - 0.5, color="#ffffff", linewidth=0.7)

    for y in range(0, altura + 1, 8):
        eixo.axhline(y - 0.5, color="#ffffff", linewidth=0.7)

    eixo.add_patch(Rectangle((31.5, 23.5), 32, 8, fill=False, edgecolor=AMARELO, linewidth=1.8))

    eixo = figura.add_subplot(grade[2])
    bloco = regiao[24:32, 32:64]
    painel(eixo, bloco, "(c) Um bloco: cada thread, um pixel")

    for x in range(33):
        eixo.axvline(x - 0.5, color="#ffffff", linewidth=0.25)

    for y in range(9):
        eixo.axhline(y - 0.5, color="#ffffff", linewidth=0.25)

    eixo.add_patch(Rectangle((13.5, 3.5), 1, 1, facecolor=AMARELO, edgecolor=AMARELO))

    salvar(figura, DESTINO, "simt")


# =============================================================
# Capitulo 2: metricas
# =============================================================

def metricas():
    referencia = ler("frame10i11.png")
    borrada = cv2.GaussianBlur(referencia, (0, 0), 2.2)
    erro = numpy.sqrt(((referencia.astype(numpy.float64) - borrada) ** 2).mean())

    # Ruido com o mesmo erro quadratico do borrado, para o mesmo PSNR.
    gerador = numpy.random.default_rng(2026)
    desvio = erro

    for _ in range(6):
        ruidosa = numpy.clip(referencia + gerador.normal(0, desvio, referencia.shape), 0, 255).round().astype(numpy.uint8)
        desvio *= erro / numpy.sqrt(((referencia.astype(numpy.float64) - ruidosa) ** 2).mean())

    imagens = [("(a) Referência", referencia, None), ("(b) Borrada", borrada, referencia), ("(c) Com ruído", ruidosa, referencia)]
    figura, eixos = plt.subplots(1, 3, figsize=(LARGURA_FIGURA, 2.05))
    regiao = REGIAO_DETALHE

    for eixo, (titulo, imagem, real) in zip(eixos, imagens):
        if real is not None:
            titulo += f"\nPSNR {numero(psnr(real, imagem), 2)} dB, SSIM {numero(ssim(real, imagem), 3)}"
            print(f"  {titulo}")
        else:
            titulo += "\n "

        painel(eixo, recortar(imagem, regiao), titulo)

    figura.tight_layout(pad=0.3)
    salvar(figura, DESTINO, "metricas")


FIGURAS = {f.__name__: f for f in (
    vfi_sequencia, vfi_metodos, imagem_digital, linearizacao, filtragem, fluxo_optico, memoria_filtros,
    host_device, simt, metricas, filtro_adaptativo, separabilidade,
)}


def main():
    os.makedirs(DESTINO, exist_ok=True)

    for nome in sys.argv[1:] or FIGURAS:
        print(nome)
        FIGURAS[nome]()


if __name__ == "__main__":
    main()
