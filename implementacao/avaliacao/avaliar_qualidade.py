"""
Avalia a fidelidade dos quadros sintetizados contra o ground truth
(PSNR, SSIM e VMAF).

O video interpolado deve ter sido gerado com --factor 2 a partir de
gta_<res>_15fps.mkv. Assim, o quadro k da saida corresponde ao quadro k do
gta_<res>_30fps.mkv: os indices pares sao copias dos originais e os impares
sao sintetizados. APENAS os impares entram nas metricas, para que as copias
dos quadros originais nao inflem os resultados.

Tarjas pretas (letterbox): o trailer tem faixas pretas em cima e embaixo, que
sao reconstruidas sem erro e tambem inflariam as metricas. A area ativa e
detectada no video de referencia (linhas/colunas que em algum quadro passam
de um limiar de intensidade) e apenas ela e avaliada.

VMAF: os quadros impares das duas sequencias sao gravados em videos
temporarios sem perdas (FFV1) e comparados pelo filtro libvmaf do FFmpeg
(modelo padrao vmaf_v0.6.1, apos conversao de ambos para yuv420p).

Uso:
    python implementacao/avaliacao/avaliar_qualidade.py \
        --referencia dados/testes/gta_1080p_30fps.mkv \
        --interpolado resultados/testes_gta/<data>/saidas/cpp_1080p.mkv \
        --rotulo cpp [--csv qualidade.csv] [--csv-quadros quadros.csv] [--sem-vmaf]
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calcular_vmaf(distorcido, referencia, pasta):
    """Retorna (media, lista por quadro) do VMAF entre dois videos."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("Erro: ffmpeg nao encontrado no PATH (use --sem-vmaf para pular)")

    filtro = (
        "[0:v]format=yuv420p,setpts=PTS-STARTPTS[distorcido];"
        "[1:v]format=yuv420p,setpts=PTS-STARTPTS[referencia];"
        f"[distorcido][referencia]libvmaf=log_fmt=json:log_path=vmaf.json:n_threads={os.cpu_count()}"
    )

    # O libvmaf espera o video avaliado na primeira entrada e a referencia
    # na segunda. O log e gravado no diretorio temporario (cwd) para evitar
    # problemas com ":" de caminhos do Windows na sintaxe de filtros.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", distorcido, "-i", referencia,
         "-lavfi", filtro, "-f", "null", "-"],
        cwd=pasta,
        check=True,
    )

    with open(os.path.join(pasta, "vmaf.json"), encoding="utf-8") as arquivo:
        log = json.load(arquivo)

    por_quadro = [quadro["metrics"]["vmaf"] for quadro in log["frames"]]
    return log["pooled_metrics"]["vmaf"]["mean"], por_quadro


def area_ativa(caminho, limiar=20):
    """Retorna (y0, y1, x0, x1) da regiao sem tarjas pretas do video."""
    captura = cv2.VideoCapture(caminho)
    maximo = None

    while True:
        ok, quadro = captura.read()

        if not ok:
            break

        atual = quadro.max(axis=2)
        maximo = atual if maximo is None else numpy.maximum(maximo, atual)

    captura.release()

    linhas = numpy.where(maximo.max(axis=1) > limiar)[0]
    colunas = numpy.where(maximo.max(axis=0) > limiar)[0]

    return linhas[0], linhas[-1] + 1, colunas[0], colunas[-1] + 1


def main():
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument("--referencia", required=True)
    argumentos.add_argument("--interpolado", required=True)
    argumentos.add_argument("--rotulo", required=True, help="ex.: cpp ou python")
    argumentos.add_argument("--csv", help="acrescenta uma linha de resumo")
    argumentos.add_argument("--csv-quadros", help="grava as metricas de cada quadro sintetizado")
    argumentos.add_argument("--sem-vmaf", action="store_true")
    argumentos.add_argument("--sem-recorte", action="store_true", help="avalia o quadro inteiro, com tarjas")
    args = argumentos.parse_args()

    if args.sem_recorte:
        y0, y1, x0, x1 = 0, None, 0, None
        descricao_area = "quadro inteiro"
    else:
        y0, y1, x0, x1 = area_ativa(args.referencia)
        descricao_area = f"{x1 - x0}x{y1 - y0} (linhas {y0}-{y1 - 1})"
        print(f"  area avaliada (sem tarjas): {descricao_area}")

    referencia = cv2.VideoCapture(args.referencia)
    interpolado = cv2.VideoCapture(args.interpolado)
    fps = referencia.get(cv2.CAP_PROP_FPS) / 2

    pasta_temporaria = tempfile.mkdtemp(prefix="vmaf_")
    caminho_sintetizados = os.path.join(pasta_temporaria, "sintetizados.mkv")
    caminho_reais = os.path.join(pasta_temporaria, "reais.mkv")
    escritores = None

    indices = []
    psnrs = []
    ssims = []
    indice = 0

    try:
        while True:
            ok_ref, quadro_ref = referencia.read()
            ok_int, quadro_int = interpolado.read()

            if not ok_ref or not ok_int:
                break

            if quadro_ref.shape != quadro_int.shape:
                raise SystemExit(f"Erro: resolucoes diferentes {quadro_ref.shape} x {quadro_int.shape}")

            if indice % 2 == 1:
                quadro_ref = numpy.ascontiguousarray(quadro_ref[y0:y1, x0:x1])
                quadro_int = numpy.ascontiguousarray(quadro_int[y0:y1, x0:x1])
                indices.append(indice)
                psnrs.append(peak_signal_noise_ratio(quadro_ref, quadro_int, data_range=255))
                ssims.append(structural_similarity(quadro_ref, quadro_int, channel_axis=2, data_range=255))

                if not args.sem_vmaf:
                    if escritores is None:
                        altura, largura = quadro_ref.shape[:2]
                        codec = cv2.VideoWriter_fourcc(*"FFV1")
                        escritores = (
                            cv2.VideoWriter(caminho_sintetizados, codec, fps, (largura, altura)),
                            cv2.VideoWriter(caminho_reais, codec, fps, (largura, altura)),
                        )

                    escritores[0].write(quadro_int)
                    escritores[1].write(quadro_ref)

                print(f"\r  quadro {indice}: PSNR {psnrs[-1]:.2f} dB  SSIM {ssims[-1]:.4f}", end="", flush=True)

            indice += 1

        print()

        if not psnrs:
            raise SystemExit("Erro: nenhum quadro sintetizado encontrado")

        vmaf_media = None
        vmafs = [None] * len(psnrs)

        if escritores is not None:
            for escritor in escritores:
                escritor.release()

            print("  calculando VMAF...")
            vmaf_media, vmafs = calcular_vmaf(caminho_sintetizados, caminho_reais, pasta_temporaria)
    finally:
        shutil.rmtree(pasta_temporaria, ignore_errors=True)

    psnrs = numpy.array(psnrs)
    ssims = numpy.array(ssims)

    print(f"\n{args.rotulo} - {len(psnrs)} quadros sintetizados")
    print(f"  PSNR: {psnrs.mean():.2f} dB (+/- {psnrs.std():.2f}), min {psnrs.min():.2f}")
    print(f"  SSIM: {ssims.mean():.4f} (+/- {ssims.std():.4f}), min {ssims.min():.4f}")

    if vmaf_media is not None:
        vmaf_array = numpy.array(vmafs)
        print(f"  VMAF: {vmaf_media:.2f} (+/- {vmaf_array.std():.2f}), min {vmaf_array.min():.2f}")

    if args.csv:
        novo = not os.path.exists(args.csv)
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)

        with open(args.csv, "a", newline="") as arquivo:
            escritor = csv.writer(arquivo)

            if novo:
                escritor.writerow([
                    "implementacao", "referencia", "interpolado", "quadros_avaliados",
                    "psnr_media", "psnr_desvio", "psnr_min", "ssim_media", "ssim_desvio", "ssim_min",
                    "vmaf_media", "vmaf_desvio", "vmaf_min", "area_avaliada",
                ])

            linha = [
                args.rotulo, os.path.basename(args.referencia), os.path.basename(args.interpolado), len(psnrs),
                f"{psnrs.mean():.4f}", f"{psnrs.std():.4f}", f"{psnrs.min():.4f}",
                f"{ssims.mean():.6f}", f"{ssims.std():.6f}", f"{ssims.min():.6f}",
            ]

            if vmaf_media is not None:
                linha += [f"{vmaf_media:.4f}", f"{vmaf_array.std():.4f}", f"{vmaf_array.min():.4f}"]
            else:
                linha += ["", "", ""]

            escritor.writerow(linha + [descricao_area])

        print(f"Resumo acrescentado em: {args.csv}")

    if args.csv_quadros:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv_quadros)), exist_ok=True)

        with open(args.csv_quadros, "w", newline="") as arquivo:
            escritor = csv.writer(arquivo)
            escritor.writerow(["implementacao", "referencia", "quadro", "psnr", "ssim", "vmaf"])

            for quadro, psnr, ssim, vmaf in zip(indices, psnrs, ssims, vmafs):
                escritor.writerow([
                    args.rotulo, os.path.basename(args.referencia), quadro,
                    f"{psnr:.4f}", f"{ssim:.6f}", "" if vmaf is None else f"{vmaf:.4f}",
                ])


if __name__ == "__main__":
    main()
