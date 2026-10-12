"""
Prepara os videos de teste a partir do trecho 4K do trailer de GTA VI.

Para cada resolucao gera, em dados/testes/, dois arquivos sem perdas (FFV1):

  gta_<res>_30fps.mkv   trecho original (usado no teste de desempenho e
                        como ground truth no teste de qualidade)
  gta_<res>_15fps.mkv   apenas os quadros pares do trecho; ao interpolar 2x,
                        os quadros impares sintetizados podem ser comparados
                        com os quadros reais do arquivo de 30 FPS

Uso:
    python implementacao/utilitarios_video/preparar_testes.py [--inicio 8.0] [--quadros 61]
"""

import argparse
import os

import cv2

# Raiz do projeto (dois niveis acima de implementacao/<pasta>/)
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ORIGEM = os.path.join(RAIZ, "dados", "videos", "GTAVI_Trailer_2_6s_26s.mp4")
DESTINO = os.path.join(RAIZ, "dados", "testes")

RESOLUCOES = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
}


def main():
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument("--inicio", type=float, default=8.0, help="segundo inicial do trecho (8 s a 10 s nao tem cortes de cena)")
    argumentos.add_argument("--quadros", type=int, default=61, help="quantidade de quadros (impar)")
    args = argumentos.parse_args()

    captura = cv2.VideoCapture(ORIGEM)

    if not captura.isOpened():
        raise SystemExit(f"Erro: nao foi possivel abrir {ORIGEM}")

    fps = captura.get(cv2.CAP_PROP_FPS)
    captura.set(cv2.CAP_PROP_POS_FRAMES, int(round(args.inicio * fps)))

    quadros = []

    while len(quadros) < args.quadros:
        ok, quadro = captura.read()

        if not ok:
            break

        quadros.append(quadro)

    captura.release()

    print(f"{len(quadros)} quadros lidos de {os.path.basename(ORIGEM)} a partir de {args.inicio:.1f} s")

    os.makedirs(DESTINO, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*"FFV1")

    for nome, (largura, altura) in RESOLUCOES.items():
        caminho_30 = os.path.join(DESTINO, f"gta_{nome}_30fps.mkv")
        caminho_15 = os.path.join(DESTINO, f"gta_{nome}_15fps.mkv")

        escritor_30 = cv2.VideoWriter(caminho_30, codec, fps, (largura, altura))
        escritor_15 = cv2.VideoWriter(caminho_15, codec, fps / 2, (largura, altura))

        if not escritor_30.isOpened() or not escritor_15.isOpened():
            raise SystemExit("Erro: o OpenCV nao conseguiu criar videos FFV1")

        for indice, quadro in enumerate(quadros):
            if (quadro.shape[1], quadro.shape[0]) != (largura, altura):
                quadro = cv2.resize(quadro, (largura, altura), interpolation=cv2.INTER_AREA)

            escritor_30.write(quadro)

            if indice % 2 == 0:
                escritor_15.write(quadro)

        escritor_30.release()
        escritor_15.release()

        print(f"  {nome}: {os.path.relpath(caminho_30, RAIZ)} e {os.path.basename(caminho_15)}")


if __name__ == "__main__":
    main()
