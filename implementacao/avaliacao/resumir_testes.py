"""
Junta desempenho.csv, qualidade.csv e falhas.csv de uma bateria de testes
em uma tabela comparativa (impressa e salva como resumo.md na mesma pasta).

Uso:
    python implementacao/avaliacao/resumir_testes.py resultados/testes_gta/<data_hora>
"""

import csv
import os
import re
import sys


def ler_csv(caminho):
    if not os.path.exists(caminho):
        return []

    with open(caminho, newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def resolucao(nome_arquivo):
    encontrado = re.search(r"(\d+p)", nome_arquivo)
    return encontrado.group(1) if encontrado else nome_arquivo


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    pasta = sys.argv[1]
    linhas = {}

    for item in ler_csv(os.path.join(pasta, "desempenho.csv")):
        chave = (resolucao(item["entrada"]), item["implementacao"])
        linhas[chave] = {
            "ms": f"{float(item['total_ms']):.1f} +/- {float(item['total_desvio_ms']):.1f}",
            "vram": f"{float(item['vram_consumo_mib']):.0f}",
            "saturada": "sim" if item["vram_saturada"] == "1" else "nao",
        }

    for item in ler_csv(os.path.join(pasta, "qualidade.csv")):
        chave = (resolucao(item["referencia"]), item["implementacao"])
        linhas.setdefault(chave, {}).update({
            "psnr": f"{float(item['psnr_media']):.2f}",
            "ssim": f"{float(item['ssim_media']):.4f}",
            "vmaf": f"{float(item['vmaf_media']):.2f}" if item.get("vmaf_media") else "-",
        })

    caminho_falhas = os.path.join(pasta, "falhas.csv")

    if os.path.exists(caminho_falhas):
        with open(caminho_falhas) as arquivo:
            for linha in arquivo:
                implementacao, res = linha.strip().split(",")
                linhas[(res, implementacao)] = {"ms": "FALHOU"}

    ordem = {"720p": 0, "1080p": 1, "1440p": 2, "2160p": 3}

    tabela = [
        "| Resolucao | Implementacao | ms/quadro | VRAM do processo (MiB) | VRAM saturada | PSNR (dB) | SSIM | VMAF |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for (res, implementacao) in sorted(linhas, key=lambda c: (ordem.get(c[0], 9), c[1])):
        dados = linhas[(res, implementacao)]
        tabela.append(
            f"| {res} | {implementacao} | {dados.get('ms', '-')} | {dados.get('vram', '-')} | "
            f"{dados.get('saturada', '-')} | {dados.get('psnr', '-')} | {dados.get('ssim', '-')} | {dados.get('vmaf', '-')} |"
        )

    texto = "\n".join(tabela)
    print("\n" + texto)

    with open(os.path.join(pasta, "resumo.md"), "w", encoding="utf-8") as arquivo:
        arquivo.write(texto + "\n")


if __name__ == "__main__":
    main()
