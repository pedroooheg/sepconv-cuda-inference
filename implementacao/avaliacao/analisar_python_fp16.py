"""
Analise da bateria 2 x 2 (Python e C++, FP32 e FP16): separa o efeito da forma
de execucao do efeito da precisao e gera, em <pasta>/graficos/, os graficos de
latencia e VRAM e a tabela de efeitos (efeitos.md).

  forma de execucao   C++ x Python, com a mesma precisao
  precisao            FP16 x FP32, na mesma implementacao
  combinado           C++ FP16 x Python FP32 (a comparacao da hipotese)

As latencias sao comparadas pelo teste t de Welch unilateral, calculado a
partir da media e do desvio de cada serie (desvio populacional no CSV).
Medicoes com a VRAM saturada sao marcadas, porque medem a saturacao.

Uso:
    python implementacao/avaliacao/analisar_python_fp16.py resultados/testes_python_fp16/<data_hora>
"""

import math
import os
import sys

from scipy import stats

from gerar_graficos import (
    COR_CPP, COR_CPP_FP16, COR_PYTHON, Estilo, carregar, grafico_latencia, grafico_vram,
)

SERIES = ["python", "python_fp16", "cpp", "cpp_fp16"]
COR_PYTHON_FP16 = "#a8431c"  # laranja escuro: mesma implementacao, em FP16
ORDEM_MOSTRADA = ["720p", "1080p", "1440p", "2160p"]

# Saturacao que o limiar de 97% nao detectou, mantida em separado do CSV bruto.
# C++ FP32 em 1440p, bateria 2026-11-05_21-09: pico de 96,5% da VRAM, mas
# inferencia de 2.576 ms, igual aos 2.578 ms da bateria 2026-11-01_22-09, em
# que a mesma condicao foi marcada (pico de 98,1%). Sem saturacao, a escala de
# 1080p para 1440p levaria a cerca de 545 ms.
SATURADAS_PELO_TEMPO = {("1440p", "cpp")}


class EstiloDuasPorDuas(Estilo):
    """As quatro series: mesma cor por implementacao, tom escuro para FP16."""

    def __init__(self):
        super().__init__(SERIES)

    def cor(self, serie):
        return {"python": COR_PYTHON, "python_fp16": COR_PYTHON_FP16,
                "cpp": COR_CPP, "cpp_fp16": COR_CPP_FP16}[serie]

    def hachura(self, serie):
        return "///" if serie.startswith("python") else ""

    def marcador(self, serie):
        return "p" if serie.endswith("_fp16") else "o"

    def nome(self, serie):
        implementacao = "Python/PyTorch" if serie.startswith("python") else "C++/CUDA"
        return f"{implementacao} {'FP16' if serie.endswith('_fp16') else 'FP32'}"

    def nome_curto(self, serie):
        return self.nome(serie).replace("/PyTorch", "").replace("/CUDA", "")


def comparar(linha_a, linha_b):
    """Variacao de A em relacao a B e teste de Welch para A mais rapido que B."""
    m1, s1, n1 = float(linha_a["total_ms"]), float(linha_a["total_desvio_ms"]), int(linha_a["medicoes"])
    m2, s2, n2 = float(linha_b["total_ms"]), float(linha_b["total_desvio_ms"]), int(linha_b["medicoes"])

    v1, v2 = s1 ** 2 * n1 / (n1 - 1), s2 ** 2 * n2 / (n2 - 1)
    erro = math.sqrt(v1 / n1 + v2 / n2)
    t = (m2 - m1) / erro
    livres = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    vram = 100.0 * float(linha_a["vram_consumo_mib"]) / float(linha_b["vram_consumo_mib"])
    saturada = linha_a["vram_saturada"] == "1" or linha_b["vram_saturada"] == "1"

    return 100.0 * (m1 - m2) / m2, t, stats.t.sf(t, livres), vram, saturada


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    pasta = sys.argv[1]
    desempenho, _, falhas, _, resolucoes, _ = carregar(pasta)

    for chave in SATURADAS_PELO_TEMPO & set(desempenho):
        desempenho[chave] = dict(desempenho[chave], vram_saturada="1")
        print(f"Marcada como saturada pelo tempo: {chave}")

    comparacoes = [
        ("Forma de execucao, FP32", "cpp", "python"),
        ("Forma de execucao, FP16", "cpp_fp16", "python_fp16"),
        ("Precisao, Python", "python_fp16", "python"),
        ("Precisao, C++", "cpp_fp16", "cpp"),
        ("Combinado", "cpp_fp16", "python"),
    ]

    linhas = [
        "| Resolucao | Comparacao | A x B | Latencia | t | p | VRAM de A / B | Saturada |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for resolucao in [r for r in ORDEM_MOSTRADA if r in resolucoes]:
        for nome, a, b in comparacoes:
            if (resolucao, a) not in desempenho or (resolucao, b) not in desempenho:
                linhas.append(f"| {resolucao} | {nome} | {a} x {b} | sem dados | - | - | - | - |")
                continue

            variacao, t, p, vram, saturada = comparar(desempenho[(resolucao, a)], desempenho[(resolucao, b)])
            linhas.append(f"| {resolucao} | {nome} | {a} x {b} | {variacao:+.1f}% | {t:.1f} | {p:.2g} | "
                          f"{vram:.1f}% | {'sim' if saturada else 'nao'} |")

    texto = "\n".join(linhas)
    print(texto)

    with open(os.path.join(pasta, "efeitos.md"), "w", encoding="utf-8") as arquivo:
        arquivo.write(texto + "\n")

    graficos = os.path.join(pasta, "graficos")
    os.makedirs(graficos, exist_ok=True)
    estilo = EstiloDuasPorDuas()
    grafico_latencia(desempenho, falhas, resolucoes, estilo, graficos)
    grafico_vram(desempenho, falhas, resolucoes, estilo, graficos)


if __name__ == "__main__":
    main()
