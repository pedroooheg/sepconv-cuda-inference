"""
Baseline Python/PyTorch: mede o tempo por quadro interpolado e o consumo de
VRAM do run.py original, com a mesma metodologia e a mesma interface do motor
C++ (aquecimento, VRAM da GPU inteira via NVML menos a linha de base, mesmo
esquema de interpolacao e mesmo formato de CSV).

Modo imagem:
    python implementacao/avaliacao/benchmark_python.py --one a.png --two b.png --out out.png [--runs 10] [--warmup 3]

Modo video:
    python implementacao/avaliacao/benchmark_python.py --video in.mkv --out out.mkv [--factor 2|4] [--max-frames N] [--warmup 3]

Opcao comum:
    --csv resumo.csv   acrescenta uma linha de resumo ao CSV
"""

import argparse
import csv
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

argumentos = argparse.ArgumentParser()
argumentos.add_argument("--one")
argumentos.add_argument("--two")
argumentos.add_argument("--video")
argumentos.add_argument("--out", required=True)
argumentos.add_argument("--csv")
argumentos.add_argument("--runs", type=int, default=10)
argumentos.add_argument("--warmup", type=int, default=3)
argumentos.add_argument("--factor", type=int, default=2, choices=[2, 4])
argumentos.add_argument("--max-frames", type=int, default=0)
args = argumentos.parse_args()

if bool(args.video) == bool(args.one and args.two):
    argumentos.error("informe --video OU --one e --two")

sys.argv = sys.argv[:1]  # run.py faz getopt no import
sys.path.insert(0, os.path.join(RAIZ, "original"))

import cv2
import numpy
import pynvml
import torch

pynvml.nvmlInit()
dispositivo = pynvml.nvmlDeviceGetHandleByIndex(0)


def vram_usada_mib():
    return pynvml.nvmlDeviceGetMemoryInfo(dispositivo).used / (1024 * 1024)


vram_total = pynvml.nvmlDeviceGetMemoryInfo(dispositivo).total / (1024 * 1024)
vram_base = vram_usada_mib()
vram_pico = vram_base

import run  # noqa: E402  (importado depois da linha de base de VRAM)

tempos = []


# Os quadros sao lidos e gravados com OpenCV (BGR), como no motor C++,
# e convertidos para o formato que o run.py espera (BGR, CHW, float [0, 1]).
def para_tensor(quadro):
    return torch.FloatTensor(numpy.ascontiguousarray(
        quadro.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)
    ))


def para_quadro(tensor):
    return numpy.ascontiguousarray(
        (tensor.clip(0.0, 1.0).numpy().transpose(1, 2, 0) * 255.0).astype(numpy.uint8)
    )


def interpolar(um, dois, medir):
    """run.estimate inclui Host->Device, inferencia, sepconv e Device->Host."""
    global vram_pico

    torch.cuda.synchronize()
    inicio = time.perf_counter()

    saida = run.estimate(um, dois)

    torch.cuda.synchronize()
    fim = time.perf_counter()

    vram_pico = max(vram_pico, vram_usada_mib())

    if medir:
        tempos.append((fim - inicio) * 1000.0)

    return saida


def modo_imagem():
    quadro_um = cv2.imread(args.one, cv2.IMREAD_COLOR)
    quadro_dois = cv2.imread(args.two, cv2.IMREAD_COLOR)

    um = para_tensor(quadro_um)
    dois = para_tensor(quadro_dois)

    for iteracao in range(args.warmup + args.runs):
        saida = interpolar(um, dois, iteracao >= args.warmup)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, para_quadro(saida))

    print(f"Resolucao: {quadro_um.shape[1]}x{quadro_um.shape[0]}")
    return "imagem", args.one, quadro_um.shape[1], quadro_um.shape[0], 2, 1, 0.0


def modo_video():
    captura = cv2.VideoCapture(args.video)

    if not captura.isOpened():
        raise SystemExit(f"Erro: nao foi possivel abrir {args.video}")

    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS)
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.max_frames > 0:
        total = min(total, args.max_frames)

    print(f"Video: {largura}x{altura} @ {fps:g} FPS, {total} quadros de entrada, fator {args.factor}x")

    inicio = time.perf_counter()

    extensao = os.path.splitext(args.out)[1].lower()
    codec = cv2.VideoWriter_fourcc(*("FFV1" if extensao in (".mkv", ".avi") else "mp4v"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    escritor = cv2.VideoWriter(args.out, codec, fps * args.factor, (largura, altura))

    interpolacoes = 0

    def medir():
        nonlocal interpolacoes
        interpolacoes += 1
        return interpolacoes > args.warmup

    gravados = 0
    _, anterior = captura.read()

    for indice in range(1, total):
        ok, proximo = captura.read()

        if not ok:
            break

        escritor.write(anterior)
        gravados += 1

        # Mesmo esquema do run.py original para o fator 4.
        tensor_anterior = para_tensor(anterior)
        tensor_proximo = para_tensor(proximo)

        meio = interpolar(tensor_anterior, tensor_proximo, medir())

        if args.factor == 4:
            quarto = interpolar(tensor_anterior, meio, medir())
            tres_quartos = interpolar(meio, tensor_proximo, medir())

            for tensor in (quarto, meio, tres_quartos):
                escritor.write(para_quadro(tensor))

            gravados += 3
        else:
            escritor.write(para_quadro(meio))
            gravados += 1

        anterior = proximo
        print(f"\r  par {indice}/{total - 1}", end="", flush=True)

    escritor.write(anterior)
    gravados += 1
    escritor.release()

    tempo_video = time.perf_counter() - inicio
    print()

    return "video", args.video, largura, altura, args.factor, gravados, tempo_video


def main():
    modo, entrada, largura, altura, fator, gravados, tempo_video = (
        modo_video() if args.video else modo_imagem()
    )

    medidas = numpy.array(tempos)
    # A NVML so enxerga a memoria fisica da GPU. Quando o PyTorch reserva mais
    # do que a VRAM livre, o excedente vai para a memoria compartilhada do
    # sistema (Sysmem Fallback) sem aparecer no pico da NVML.
    saturada = (
        vram_pico >= 0.97 * vram_total
        or torch.cuda.max_memory_reserved() / 2**20 > vram_total - vram_base
    )

    print(f"\nTempo por quadro interpolado ({len(medidas)} medicoes):")
    print(f"  TOTAL  {medidas.mean():10.3f} ms  (+/- {medidas.std():.3f})")
    print(f"  Vazao da interpolacao: {1000.0 / medidas.mean():.2f} quadros/s")

    if modo == "video":
        print(f"  Tempo total do video (com decodificacao e gravacao): {tempo_video:.2f} s"
              f"  ->  {gravados / tempo_video:.2f} quadros de saida/s")

    print("\nVRAM:")
    print(f"  Capacidade (NVML):             {vram_total:.1f} MiB")
    print(f"  Linha de base (NVML):          {vram_base:.1f} MiB")
    print(f"  Pico observado (NVML):         {vram_pico:.1f} MiB")
    print(f"  Consumo do processo (NVML):    {vram_pico - vram_base:.1f} MiB")
    print(f"  torch max_memory_allocated:    {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
    print(f"  torch max_memory_reserved:     {torch.cuda.max_memory_reserved() / 2**20:.1f} MiB")

    if saturada:
        print("\n  ATENCAO: VRAM saturada. O driver pode ter recorrido a memoria\n"
              "  compartilhada do sistema; os tempos medidos ficam inflados.")

    if args.csv:
        novo = not os.path.exists(args.csv)
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)

        with open(args.csv, "a", newline="") as arquivo:
            escritor = csv.writer(arquivo)

            if novo:
                escritor.writerow([
                    "implementacao", "modo", "entrada", "largura", "altura", "fator", "medicoes",
                    "upload_ms", "pre_ms", "inferencia_ms", "sepconv_ms", "download_ms", "total_ms", "total_desvio_ms",
                    "quadros_saida", "tempo_video_s", "vram_total_mib", "vram_base_mib", "vram_pico_mib",
                    "vram_consumo_mib", "vram_saturada", "bloco", "margem",
                ])

            # O run.estimate nao separa os estagios: apenas o total e medido.
            escritor.writerow([
                "python", modo, os.path.basename(entrada), largura, altura, fator, len(medidas),
                "", "", "", "", "", f"{medidas.mean():.3f}", f"{medidas.std():.3f}",
                gravados, f"{tempo_video:.3f}", f"{vram_total:.3f}", f"{vram_base:.3f}", f"{vram_pico:.3f}",
                f"{vram_pico - vram_base:.3f}", int(saturada), 0, 0,
            ])

        print(f"Resumo acrescentado em: {args.csv}")

    print(f"\nSaida salva em: {args.out}")


if __name__ == "__main__":
    main()
