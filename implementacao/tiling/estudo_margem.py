"""
Estudo da margem de sobreposicao para o processamento em blocos (tiling).

Para cada tamanho de bloco T e margem M, o quadro e dividido em blocos T x T.
A rede roda em cada bloco acrescido de M pixels de cada lado, usando a media e
o desvio padrao do quadro INTEIRO na normalizacao, e apenas o miolo do bloco
e aproveitado. O resultado e comparado com a interpolacao do quadro inteiro.

Os blocos comecam em multiplos de 16 (fator de reducao da rede), para que a
grade interna da rede coincida com a do quadro inteiro.

Uso:
    python implementacao/tiling/estudo_margem.py [--res 1080p] [--blocos 256 512] [--margens 0 16 32 64 96 128]
"""

import argparse
import csv
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

argumentos = argparse.ArgumentParser()
argumentos.add_argument("--res", default="1080p")
argumentos.add_argument("--pares", type=int, nargs="+", default=[0, 20, 40],
                        help="indice do primeiro quadro de cada par (o segundo e +2)")
argumentos.add_argument("--blocos", type=int, nargs="+", default=[256, 512])
argumentos.add_argument("--margens", type=int, nargs="+", default=[0, 16, 32, 48, 64, 96, 128, 160])
argumentos.add_argument("--csv", default=os.path.join(RAIZ, "resultados", "tiling", "estudo_margem.csv"))
args = argumentos.parse_args()

sys.argv = sys.argv[:1]  # run.py faz getopt no import
sys.path.insert(0, os.path.join(RAIZ, "original"))

import cv2
import numpy
import torch

import run
import sepconv

torch.set_grad_enabled(False)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

RAIO = 25  # metade do filtro de 51 coeficientes


def ler_par(caminho, indice):
    captura = cv2.VideoCapture(caminho)
    captura.set(cv2.CAP_PROP_POS_FRAMES, indice)
    quadros = []

    for _ in range(3):
        ok, quadro = captura.read()
        if not ok:
            raise SystemExit(f"Erro: nao foi possivel ler o quadro {indice} de {caminho}")
        quadros.append(quadro)

    def tensor(quadro):
        return torch.from_numpy(quadro.transpose(2, 0, 1).astype(numpy.float32) / 255.0)[None].cuda()

    return tensor(quadros[0]), tensor(quadros[2])


def estatisticas(um, dois):
    """Media e desvio usados pela rede, calculados como em run.Network.forward."""
    media = (um.mean([1, 2, 3], True) + dois.mean([1, 2, 3], True)) / 2
    desvio = ((um.std([1, 2, 3], False, True).square() + (media - um.mean([1, 2, 3], True)).square() +
               dois.std([1, 2, 3], False, True).square() + (media - dois.mean([1, 2, 3], True)).square()) / 2).sqrt()
    return media, desvio


def filtros(rede, um, dois, media, desvio):
    um = (um - media) / (desvio + 0.0000001)
    dois = (dois - media) / (desvio + 0.0000001)

    saida = rede.netDecode(rede.netEncode(
        [torch.cat([rede.netInput(um), rede.netInput(dois)], 1)] + [0.0] * (len(rede.intChannels) - 1)
    ))[1]

    return rede.netVerone(saida), rede.netVertwo(saida), rede.netHorone(saida), rede.netHortwo(saida)


def com_borda(quadro):
    """Quadro + canal de uns, com borda replicada de 25 px (como em run.py)."""
    uns = quadro.new_ones([1, 1, quadro.shape[2], quadro.shape[3]])
    return torch.nn.functional.pad(torch.cat([quadro, uns], 1), [RAIO] * 4, mode="replicate")


def sintetizar(borda_um, borda_dois, ver1, ver2, hor1, hor2, y0, y1, x0, x1):
    """Aplica a convolucao separavel na regiao [y0, y1) x [x0, x1)."""
    entrada_um = borda_um[:, :, y0:y1 + 2 * RAIO, x0:x1 + 2 * RAIO].contiguous()
    entrada_dois = borda_dois[:, :, y0:y1 + 2 * RAIO, x0:x1 + 2 * RAIO].contiguous()

    saida = (sepconv.sepconv_func.apply(entrada_um, ver1.contiguous(), hor1.contiguous()) +
             sepconv.sepconv_func.apply(entrada_dois, ver2.contiguous(), hor2.contiguous()))

    normalizacao = saida[:, -1:]
    normalizacao[normalizacao.abs() < 0.01] = 1.0
    return saida[:, :-1] / normalizacao


def interpolar_inteiro(rede, um, dois):
    media, desvio = estatisticas(um, dois)
    v1, v2, h1, h2 = filtros(rede, um, dois, media, desvio)
    altura, largura = um.shape[2:]
    return sintetizar(com_borda(um), com_borda(dois), v1, v2, h1, h2, 0, altura, 0, largura)


def interpolar_blocos(rede, um, dois, bloco, margem):
    media, desvio = estatisticas(um, dois)
    borda_um, borda_dois = com_borda(um), com_borda(dois)
    altura, largura = um.shape[2:]
    saida = torch.zeros_like(um)

    for y0 in range(0, altura, bloco):
        for x0 in range(0, largura, bloco):
            y1, x1 = min(y0 + bloco, altura), min(x0 + bloco, largura)
            cy0, cy1 = max(0, y0 - margem), min(altura, y1 + margem)
            cx0, cx1 = max(0, x0 - margem), min(largura, x1 + margem)

            v1, v2, h1, h2 = filtros(rede, um[:, :, cy0:cy1, cx0:cx1], dois[:, :, cy0:cy1, cx0:cx1], media, desvio)
            miolo = (slice(None), slice(None), slice(y0 - cy0, y1 - cy0), slice(x0 - cx0, x1 - cx0))

            saida[:, :, y0:y1, x0:x1] = sintetizar(
                borda_um, borda_dois, v1[miolo], v2[miolo], h1[miolo], h2[miolo], y0, y1, x0, x1
            )

    return saida


def para_uint8(tensor):
    """Mesma conversao do run.py: clip e truncamento."""
    return (tensor.clamp(0.0, 1.0) * 255.0).to(torch.uint8)


def main():
    rede = run.Network().cuda().eval()
    video = os.path.join(RAIZ, "dados", "testes", f"gta_{args.res}_30fps.mkv")
    pares = [ler_par(video, indice) for indice in args.pares]

    altura, largura = pares[0][0].shape[2:]
    print(f"{args.res} ({largura}x{altura}), {len(pares)} pares de quadros\n")

    # Conferencia: a reimplementacao do quadro inteiro deve reproduzir o run.py.
    um, dois = pares[0]
    diferenca = (para_uint8(interpolar_inteiro(rede, um, dois)).int() -
                 para_uint8(rede(um, dois)).int()).abs().max().item()
    print(f"Conferencia com run.Network (quadro inteiro): diferenca maxima = {diferenca} nivel(is)\n")

    # Referencia: quadro inteiro (tempo e pico de memoria de ativacoes).
    referencias = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    inicio = time.perf_counter()

    for um, dois in pares:
        referencias.append(para_uint8(interpolar_inteiro(rede, um, dois)))

    torch.cuda.synchronize()
    tempo_inteiro = (time.perf_counter() - inicio) / len(pares) * 1000
    pico_inteiro = torch.cuda.max_memory_allocated() / 2**20

    linhas = [{
        "resolucao": args.res, "bloco": "inteiro", "margem": "", "psnr_db": "inf", "dif_max": 0,
        "pixels_diferentes_pct": 0.0, "tempo_ms": round(tempo_inteiro, 1), "pico_mib": round(pico_inteiro),
    }]

    print(f"{'bloco':>7} {'margem':>6} {'PSNR (dB)':>10} {'dif. max':>8} {'% dif.':>7} {'ms/quadro':>10} {'pico MiB':>9}")
    print(f"{'inteiro':>7} {'':>6} {'inf':>10} {0:>8} {0:>7.3f} {tempo_inteiro:>10.1f} {pico_inteiro:>9.0f}")

    for bloco in args.blocos:
        for margem in args.margens:
            if bloco % 16 or margem % 16:
                raise SystemExit("Erro: bloco e margem devem ser multiplos de 16")

            erro_quadratico = 0.0
            dif_max = 0
            diferentes = 0
            total = 0

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            inicio = time.perf_counter()

            for (um, dois), referencia in zip(pares, referencias):
                resultado = para_uint8(interpolar_blocos(rede, um, dois, bloco, margem))
                diferenca = (resultado.int() - referencia.int()).abs()
                erro_quadratico += diferenca.float().square().sum().item()
                dif_max = max(dif_max, diferenca.max().item())
                diferentes += (diferenca > 0).sum().item()
                total += diferenca.numel()

            torch.cuda.synchronize()
            tempo = (time.perf_counter() - inicio) / len(pares) * 1000
            pico = torch.cuda.max_memory_allocated() / 2**20

            mse = erro_quadratico / total
            psnr = float("inf") if mse == 0 else 10 * numpy.log10(255.0**2 / mse)

            linhas.append({
                "resolucao": args.res, "bloco": bloco, "margem": margem, "psnr_db": round(psnr, 2),
                "dif_max": dif_max, "pixels_diferentes_pct": round(100.0 * diferentes / total, 4),
                "tempo_ms": round(tempo, 1), "pico_mib": round(pico),
            })

            print(f"{bloco:>7} {margem:>6} {psnr:>10.2f} {dif_max:>8} {100.0 * diferentes / total:>7.3f} "
                  f"{tempo:>10.1f} {pico:>9.0f}")

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    with open(args.csv, "w", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)

    print(f"\nResultados salvos em {args.csv}")


if __name__ == "__main__":
    main()
