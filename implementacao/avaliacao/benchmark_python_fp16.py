"""
Baseline Python/PyTorch em meia precisao (FP16): mesma medicao de
benchmark_python.py, com run.estimate executado dentro de torch.autocast.

O codigo original nao e alterado: a rede passa a calcular em FP16 pelo
autocast, e sepconv.sepconv_func, que ja declara suporte a precisao mista,
converte os filtros para FP32 antes da sintese. Os pesos continuam em FP32.
O motor C++ em FP16, por outro lado, mantem os filtros em FP16 na VRAM.

Modo video (o mesmo da bateria):
    python implementacao/avaliacao/benchmark_python_fp16.py --video in.mkv --out out.mkv [--factor 2] [--max-frames N] [--warmup 3]

Opcao:
    --csv resumo.csv   acrescenta uma linha de resumo ao CSV, com o rotulo python_fp16
"""

import argparse
import csv
import os
import sys
import threading
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

argumentos = argparse.ArgumentParser()
argumentos.add_argument("--video", required=True)
argumentos.add_argument("--out", required=True)
argumentos.add_argument("--csv")
argumentos.add_argument("--warmup", type=int, default=3)
argumentos.add_argument("--factor", type=int, default=2, choices=[2])
argumentos.add_argument("--max-frames", type=int, default=0)
args = argumentos.parse_args()

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


def amostrar_vram():
    global vram_pico

    while not parar_amostragem.is_set():
        vram_pico = max(vram_pico, vram_usada_mib())
        time.sleep(0.001)


if os.name == "nt":
    import ctypes
    ctypes.windll.winmm.timeBeginPeriod(1)  # resolucao de 1 ms para o sleep

parar_amostragem = threading.Event()
amostragem = threading.Thread(target=amostrar_vram, daemon=True)
amostragem.start()

import run  # noqa: E402  (importado depois da linha de base de VRAM)

tempos = []
tipos_dos_filtros = set()


# Confirma que a rede calculou em FP16: sem isso, um autocast sem efeito
# passaria por FP16 nos resultados.
def registrar_tipo(_modulo, _entrada, saida):
    tipos_dos_filtros.add(saida.dtype)


def para_tensor(quadro):
    return torch.FloatTensor(numpy.ascontiguousarray(
        quadro.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)
    ))


def para_quadro(tensor):
    return numpy.ascontiguousarray(
        (tensor.clip(0.0, 1.0).numpy().transpose(1, 2, 0) * 255.0).astype(numpy.uint8)
    )


def interpolar(um, dois, medir):
    global vram_pico

    torch.cuda.synchronize()
    inicio = time.perf_counter()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        saida = run.estimate(um, dois)

    torch.cuda.synchronize()
    fim = time.perf_counter()

    vram_pico = max(vram_pico, vram_usada_mib())

    if medir:
        tempos.append((fim - inicio) * 1000.0)

    return saida


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

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    escritor = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"FFV1"), fps * args.factor, (largura, altura))

    interpolacoes = 0
    gravados = 0
    _, anterior = captura.read()

    for indice in range(1, total):
        ok, proximo = captura.read()

        if not ok:
            break

        escritor.write(anterior)
        gravados += 1

        interpolacoes += 1
        meio = interpolar(para_tensor(anterior), para_tensor(proximo), interpolacoes > args.warmup)

        escritor.write(para_quadro(meio))
        gravados += 1

        anterior = proximo
        print(f"\r  par {indice}/{total - 1}", end="", flush=True)

    escritor.write(anterior)
    gravados += 1
    escritor.release()

    tempo_video = time.perf_counter() - inicio
    print()

    return largura, altura, gravados, tempo_video


def main():
    run.netNetwork = run.Network().cuda().train(False)
    gancho = run.netNetwork.netVerone.register_forward_hook(registrar_tipo)

    largura, altura, gravados, tempo_video = modo_video()

    gancho.remove()
    parar_amostragem.set()
    amostragem.join()

    if tipos_dos_filtros != {torch.float16}:
        raise SystemExit(f"Erro: os filtros da rede nao estao em FP16: {tipos_dos_filtros}")

    print(f"Tipo dos filtros na saida da rede: {tipos_dos_filtros}")

    medidas = numpy.array(tempos)
    saturada = (
        vram_pico >= 0.97 * vram_total
        or torch.cuda.max_memory_reserved() / 2**20 > vram_total - vram_base
    )

    print(f"\nTempo por quadro interpolado ({len(medidas)} medicoes):")
    print(f"  TOTAL  {medidas.mean():10.3f} ms  (+/- {medidas.std():.3f})")
    print(f"\nVRAM: consumo do processo (NVML) {vram_pico - vram_base:.1f} MiB, saturada: {bool(saturada)}")
    print(f"  torch max_memory_allocated: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
    print(f"  torch max_memory_reserved:  {torch.cuda.max_memory_reserved() / 2**20:.1f} MiB")

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

            escritor.writerow([
                "python_fp16", "video", os.path.basename(args.video), largura, altura, args.factor, len(medidas),
                "", "", "", "", "", f"{medidas.mean():.3f}", f"{medidas.std():.3f}",
                gravados, f"{tempo_video:.3f}", f"{vram_total:.3f}", f"{vram_base:.3f}", f"{vram_pico:.3f}",
                f"{vram_pico - vram_base:.3f}", int(saturada), 0, 0,
            ])

        print(f"Resumo acrescentado em: {args.csv}")

    print(f"\nSaida salva em: {args.out}")


if __name__ == "__main__":
    main()
