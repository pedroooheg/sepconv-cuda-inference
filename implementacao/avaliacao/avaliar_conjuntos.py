"""
Fidelidade visual nos conjuntos de referencia da literatura: Middlebury
(12 sequencias "Other" com o quadro intermediario real, distribuidas com o
codigo original em original/middlebury) e uma amostra do conjunto de teste
do Vimeo-90K (trios de 448 x 256).

Cada implementacao sintetiza o quadro intermediario de cada par, e o
resultado e comparado com o quadro real por PSNR e SSIM, calculados como no
benchmark.py original (scikit-image, data_range=255). O motor C++ roda uma
vez por par (modo imagem); o Python chama run.estimate em um unico processo
e converte a saida para 8 bits por truncamento, como o motor e o run.py.

O conjunto de teste do Vimeo-90K foi obtido de um espelho no Hugging Face
(paschalc24/vimeo-90k-test-full), porque o servidor oficial recusou o
download. A amostra e sorteada com semente fixa entre os 3.782 trios, e a
lista sorteada e gravada junto dos resultados.

Uso:
    python implementacao/avaliacao/avaliar_conjuntos.py [--amostra-vimeo 200] [--semente 2026]
                                                        [--series cpp cpp_fp16 python]
"""

import argparse
import csv
import glob
import io
import os
import random
import subprocess
import sys
import time

import cv2
import numpy
import skimage.metrics

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOTOR = os.path.join(RAIZ, "implementacao", "motor_cpp", "build", "Release", "sepconv_engine.exe")
MODELOS = {
    "cpp": os.path.join(RAIZ, "modelos", "sepconv_paper.onnx"),
    "cpp_fp16": os.path.join(RAIZ, "modelos", "sepconv_paper_fp16.onnx"),
}
PASTA_VIMEO = os.path.join(RAIZ, "dados", "vimeo90k")

argumentos = argparse.ArgumentParser()
argumentos.add_argument("--amostra-vimeo", type=int, default=200)
argumentos.add_argument("--semente", type=int, default=2026)
argumentos.add_argument("--series", nargs="+", default=["cpp", "cpp_fp16", "python"])
argumentos.add_argument("--pasta", default=os.path.join(RAIZ, "resultados", "conjuntos", time.strftime("%Y-%m-%d_%H-%M")))
args = argumentos.parse_args()


# =============================================================
# Conjuntos
# =============================================================

def pares_middlebury():
    """(conjunto, nome, quadro 1, quadro 2, quadro real)"""
    for real in sorted(glob.glob(os.path.join(RAIZ, "original", "middlebury", "*", "frame10i11.png"))):
        pasta = os.path.dirname(real)
        yield "middlebury", os.path.basename(pasta), os.path.join(pasta, "frame10.png"), os.path.join(pasta, "frame11.png"), real


def pares_vimeo():
    """Extrai a amostra sorteada dos arquivos parquet para PNG (uma vez)."""
    import pyarrow.parquet

    arquivos = sorted(glob.glob(os.path.join(PASTA_VIMEO, "test-*.parquet")))

    if not arquivos:
        raise SystemExit(f"Vimeo-90K ausente em {PASTA_VIMEO} (ver o cabecalho deste script)")

    identificadores = []

    for arquivo in arquivos:
        tabela = pyarrow.parquet.read_table(arquivo, columns=["sequence_id"])
        identificadores += [(arquivo, linha, sid) for linha, sid in enumerate(tabela.column("sequence_id").to_pylist())]

    sorteados = sorted(random.Random(args.semente).sample(identificadores, args.amostra_vimeo), key=lambda i: i[2])
    destino = os.path.join(PASTA_VIMEO, f"amostra_{args.amostra_vimeo}_semente_{args.semente}")

    def pasta_do(sid):
        return os.path.join(destino, sid.replace("/", "_"))

    # Cada arquivo parquet (~450 MB) e lido uma unica vez.
    faltantes = [s for s in sorteados if not os.path.exists(os.path.join(pasta_do(s[2]), "im3.png"))]

    for arquivo in sorted({s[0] for s in faltantes}):
        tabela = pyarrow.parquet.read_table(arquivo)

        for _, linha, sid in (s for s in faltantes if s[0] == arquivo):
            registro = tabela.slice(linha, 1).to_pylist()[0]
            os.makedirs(pasta_do(sid), exist_ok=True)

            for chave in ("im1", "im2", "im3"):
                with open(os.path.join(pasta_do(sid), f"{chave}.png"), "wb") as imagem:
                    imagem.write(registro[chave]["bytes"])

    with open(os.path.join(args.pasta, "amostra_vimeo.txt"), "w") as lista:
        lista.write(f"# {args.amostra_vimeo} de {len(identificadores)} trios, semente {args.semente}\n")
        lista.writelines(sid + "\n" for _, _, sid in sorteados)

    for _, _, sid in sorteados:
        pasta = pasta_do(sid)
        yield "vimeo90k", sid, os.path.join(pasta, "im1.png"), os.path.join(pasta, "im3.png"), os.path.join(pasta, "im2.png")


# =============================================================
# Implementacoes
# =============================================================

def sintetizar_cpp(serie, pares, pasta):
    saidas = []

    for conjunto, nome, um, dois, _ in pares:
        saida = os.path.join(pasta, serie, conjunto, nome.replace("/", "_") + ".png")
        os.makedirs(os.path.dirname(saida), exist_ok=True)

        processo = subprocess.run(
            [MOTOR, "--model", MODELOS[serie], "--one", um, "--two", dois, "--out", saida, "--runs", "1", "--warmup", "0"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )

        if processo.returncode != 0:
            print(processo.stdout[-800:], processo.stderr[-800:])
            raise SystemExit(f"motor falhou em {conjunto}/{nome}")

        saidas.append(saida)

    return saidas


def sintetizar_python(pares, pasta):
    sys.argv = sys.argv[:1]  # run.py faz getopt no import
    sys.path.insert(0, os.path.join(RAIZ, "original"))

    import torch
    import run

    def para_tensor(quadro):
        return torch.FloatTensor(numpy.ascontiguousarray(
            quadro.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)
        ))

    saidas = []

    for conjunto, nome, um, dois, _ in pares:
        estimado = run.estimate(para_tensor(cv2.imread(um)), para_tensor(cv2.imread(dois)))
        quadro = (estimado.clip(0.0, 1.0).numpy().transpose(1, 2, 0) * 255.0).astype(numpy.uint8)

        saida = os.path.join(pasta, "python", conjunto, nome.replace("/", "_") + ".png")
        os.makedirs(os.path.dirname(saida), exist_ok=True)
        cv2.imwrite(saida, quadro)
        saidas.append(saida)

    return saidas


# =============================================================
# Metricas
# =============================================================

def metricas(real, estimado):
    a, b = cv2.imread(real), cv2.imread(estimado)
    return (
        skimage.metrics.peak_signal_noise_ratio(image_true=a, image_test=b, data_range=255),
        skimage.metrics.structural_similarity(im1=a, im2=b, data_range=255, channel_axis=2),
    )


def main():
    os.makedirs(args.pasta, exist_ok=True)
    pares = list(pares_middlebury()) + list(pares_vimeo())
    print(f"{len(pares)} pares ({sum(p[0] == 'middlebury' for p in pares)} Middlebury, "
          f"{sum(p[0] == 'vimeo90k' for p in pares)} Vimeo-90K)")

    pasta_saidas = os.path.join(args.pasta, "saidas")
    linhas = []

    # O Python por ultimo: o PyTorch mantem a memoria reservada ate o fim do processo.
    for serie in sorted(args.series, key=lambda s: s == "python"):
        inicio = time.perf_counter()
        print(f"{serie} ...", flush=True)
        saidas = sintetizar_python(pares, pasta_saidas) if serie == "python" else sintetizar_cpp(serie, pares, pasta_saidas)

        for (conjunto, nome, _, _, real), saida in zip(pares, saidas):
            psnr, ssim = metricas(real, saida)
            linhas.append({"implementacao": serie, "conjunto": conjunto, "par": nome,
                           "psnr": f"{psnr:.4f}", "ssim": f"{ssim:.6f}"})

        print(f"  {time.perf_counter() - inicio:.0f} s")

    with open(os.path.join(args.pasta, "quadros.csv"), "w", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=["implementacao", "conjunto", "par", "psnr", "ssim"])
        escritor.writeheader()
        escritor.writerows(linhas)

    resumo = [f"| Conjunto | Implementacao | Pares | PSNR (dB) | SSIM |", "|---|---|---|---|---|"]

    for conjunto in ("middlebury", "vimeo90k"):
        for serie in args.series:
            selecao = [l for l in linhas if l["conjunto"] == conjunto and l["implementacao"] == serie]
            psnrs = numpy.array([float(l["psnr"]) for l in selecao])
            ssims = numpy.array([float(l["ssim"]) for l in selecao])
            resumo.append(f"| {conjunto} | {serie} | {len(selecao)} | {psnrs.mean():.2f} +/- {psnrs.std():.2f} "
                          f"| {ssims.mean():.4f} +/- {ssims.std():.4f} |")

    texto = "\n".join(resumo)
    print("\n" + texto)

    with open(os.path.join(args.pasta, "resumo.md"), "w", encoding="utf-8") as arquivo:
        arquivo.write(texto + "\n")

    print(f"\nResultados em {args.pasta}")


if __name__ == "__main__":
    main()
