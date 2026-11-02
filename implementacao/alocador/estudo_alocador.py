"""
Estudo do alocador do ONNX Runtime: quanto da VRAM do motor C++ e do proprio
motor (buffers calculados) e quanto e do ONNX Runtime (arena, workspace do
cuDNN e contexto CUDA), e quanto as opcoes do alocador conseguem reduzir.

Cada configuracao roda o motor sobre o mesmo video, com o quadro inteiro e em
blocos, e a saida e comparada com a da configuracao padrao (o resultado nao
pode mudar).

Uso:
    python implementacao/alocador/estudo_alocador.py [--res 1080p] [--quadros 11] [--aquecimento 3] [--blocos 0 512]
                                                    [--configuracoes padrao sem_padrao_de_memoria ...]
"""

import argparse
import csv
import os
import re
import subprocess
import tempfile

import cv2
import numpy

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOTOR = os.path.join(RAIZ, "implementacao", "motor_cpp", "build", "Release", "sepconv_engine.exe")
MODELOS = {
    "inteiro": os.path.join(RAIZ, "modelos", "sepconv_paper.onnx"),
    "blocos": os.path.join(RAIZ, "modelos", "sepconv_blocos.onnx"),
}

# Configuracao do motor antes deste estudo, base de todas as variacoes
# (as opcoes de cada configuracao vem depois e a substituem).
BASE = ["--padrao-memoria", "1", "--ort-option", "arena_extend_strategy=kSameAsRequested"]

LIBERAR_ARENA =["--ort-run-config", "memory.enable_memory_arena_shrinkage=gpu:0"]
PESOS_FORA_DA_ARENA = ["--ort-config", "session.use_device_allocator_for_initializers=1"]
HEURISTICA = ["--ort-option", "cudnn_conv_algo_search=HEURISTIC"]
SEM_WORKSPACE_MAXIMO = ["--ort-option", "cudnn_conv_use_max_workspace=0"]
SEM_PADRAO_DE_MEMORIA = ["--padrao-memoria", "0"]

# (rotulo, descricao, argumentos extras do motor)
POTENCIA_DE_DOIS = ["--ort-option", "arena_extend_strategy=kNextPowerOfTwo"]

CONFIGURACOES = [
    ("padrao", "Configuração atual (kSameAsRequested, busca exaustiva)", []),
    # Mostra quanto a saida varia entre execucoes identicas (escolha de
    # algoritmos do cuDNN), para separar esse efeito do efeito das opcoes.
    ("padrao_repetido", "Configuração atual, repetida", []),
    ("potencia_de_dois", "Arena cresce em potências de dois (padrão do ORT)", POTENCIA_DE_DOIS),
    ("heuristica", "Algoritmo do cuDNN por heurística", HEURISTICA),
    ("sem_workspace_maximo", "Workspace do cuDNN limitado", SEM_WORKSPACE_MAXIMO),
    ("pesos_fora_da_arena", "Pesos alocados fora da arena", PESOS_FORA_DA_ARENA),
    ("sem_padrao_de_memoria", "Sem planejamento de memória", SEM_PADRAO_DE_MEMORIA),
    ("liberar_arena", "Arena liberada após cada execução", LIBERAR_ARENA),
    ("sem_padrao_potencia_de_dois", "Sem planejamento de memória, arena em potências de dois",
     SEM_PADRAO_DE_MEMORIA + POTENCIA_DE_DOIS),
    ("combinado", "Todas as opções acima, exceto potências de dois",
     HEURISTICA + SEM_WORKSPACE_MAXIMO + PESOS_FORA_DA_ARENA + SEM_PADRAO_DE_MEMORIA + LIBERAR_ARENA),
]

argumentos = argparse.ArgumentParser()
argumentos.add_argument("--res", default="1080p")
argumentos.add_argument("--quadros", type=int, default=11, help="quadros de entrada")
argumentos.add_argument("--aquecimento", type=int, default=3, help="interpolacoes descartadas")
argumentos.add_argument("--blocos", type=int, nargs="+", default=[0, 512])
argumentos.add_argument("--configuracoes", nargs="+", help="rotulos a executar (padrao: todos)")
argumentos.add_argument("--csv", default=os.path.join(RAIZ, "resultados", "alocador", "estudo_alocador.csv"))
args = argumentos.parse_args()


def ler_video(caminho):
    captura = cv2.VideoCapture(caminho)
    quadros = []

    while True:
        ok, quadro = captura.read()

        if not ok:
            return quadros

        quadros.append(quadro)


def diferenca_maxima(caminho_a, caminho_b):
    a, b = ler_video(caminho_a), ler_video(caminho_b)

    if len(a) != len(b):
        return float("nan")

    return max(int(numpy.abs(x.astype(int) - y.astype(int)).max()) for x, y in zip(a, b))


def executar(rotulo, bloco, extras, pasta):
    entrada = os.path.join(RAIZ, "dados", "testes", f"gta_{args.res}_15fps.mkv")
    saida = os.path.join(pasta, f"{rotulo}_b{bloco}.mkv")
    resumo = os.path.join(pasta, f"{rotulo}_b{bloco}.csv")

    comando = [
        MOTOR, "--model", MODELOS["blocos" if bloco else "inteiro"],
        "--video", entrada, "--out", saida, "--factor", "2",
        "--max-frames", str(args.quadros), "--warmup", str(args.aquecimento), "--csv", resumo,
    ] + (["--bloco", str(bloco)] if bloco else []) + BASE + extras

    processo = subprocess.run(comando, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if processo.returncode != 0 or not os.path.exists(resumo):
        print(processo.stdout[-1500:], processo.stderr[-1500:])
        return None, saida

    with open(resumo, newline="") as arquivo:
        linha = next(csv.DictReader(arquivo))

    buffers = float(re.search(r"Buffers do motor:\s+([\d.]+)", processo.stdout).group(1))
    linha["buffers_mib"] = buffers
    return linha, saida


def main():
    configuracoes = [c for c in CONFIGURACOES if not args.configuracoes or c[0] in args.configuracoes]
    resultados = []

    with tempfile.TemporaryDirectory() as pasta:
        for bloco in args.blocos:
            referencia = None

            for rotulo, descricao, extras in configuracoes:
                print(f"{args.res} | {'inteiro' if not bloco else f'blocos {bloco}'} | {rotulo} ...", flush=True)
                linha, saida = executar(rotulo, bloco, extras, pasta)

                if linha is None:
                    resultados.append({"resolucao": args.res, "bloco": bloco, "configuracao": rotulo,
                                       "descricao": descricao, "falhou": 1})
                    print("  falhou")
                    continue

                if referencia is None:
                    referencia = saida

                consumo = float(linha["vram_consumo_mib"])
                resultado = {
                    "resolucao": args.res,
                    "bloco": bloco,
                    "configuracao": rotulo,
                    "descricao": descricao,
                    "falhou": 0,
                    "medicoes": linha["medicoes"],
                    "total_ms": linha["total_ms"],
                    "total_desvio_ms": linha["total_desvio_ms"],
                    "inferencia_ms": linha["inferencia_ms"],
                    "vram_consumo_mib": f"{consumo:.0f}",
                    "buffers_motor_mib": f"{linha['buffers_mib']:.0f}",
                    "ort_e_contexto_mib": f"{consumo - linha['buffers_mib']:.0f}",
                    "vram_saturada": linha["vram_saturada"],
                    "dif_max_vs_padrao": diferenca_maxima(referencia, saida),
                }
                resultados.append(resultado)
                print(f"  {float(resultado['total_ms']):8.1f} ms  VRAM {resultado['vram_consumo_mib']:>5} MiB"
                      f"  (motor {resultado['buffers_motor_mib']}, ORT+contexto {resultado['ort_e_contexto_mib']})"
                      f"  dif. max. {resultado['dif_max_vs_padrao']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    novo = not os.path.exists(args.csv)
    campos = ["resolucao", "bloco", "configuracao", "descricao", "falhou", "medicoes", "total_ms",
              "total_desvio_ms", "inferencia_ms", "vram_consumo_mib", "buffers_motor_mib",
              "ort_e_contexto_mib", "vram_saturada", "dif_max_vs_padrao"]

    with open(args.csv, "a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)

        if novo:
            escritor.writeheader()

        escritor.writerows(resultados)

    print(f"\nResultados acrescentados em {args.csv}")


if __name__ == "__main__":
    main()
