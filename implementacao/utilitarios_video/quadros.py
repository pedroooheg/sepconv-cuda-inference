import cv2
import os

# Raiz do projeto (dois niveis acima de implementacao/<pasta>/)
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# CONFIGURAÇÕES
# ============================================================

VIDEO_PATH = os.path.join(RAIZ, "resultados", "videos", "gta_side_by_side.mp4")

# Intervalo que você quer extrair
INICIO_SEGUNDOS = 7
FIM_SEGUNDOS = 8

# Pasta onde os frames serão salvos
OUTPUT_DIR = os.path.join(RAIZ, "resultados", "quadros_gta_838_957")

# ============================================================
# ABRIR VÍDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Erro: não foi possível abrir o vídeo.")
    exit()

# Informações do vídeo
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duracao = total_frames / fps

print(f"FPS: {fps}")
print(f"Total de frames: {total_frames}")
print(f"Duração: {duracao:.2f} segundos")

# ============================================================
# CRIAR PASTA
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CALCULAR FRAMES DO INTERVALO
# ============================================================

frame_inicio = int(INICIO_SEGUNDOS * fps)
frame_fim = int(FIM_SEGUNDOS * fps)

print()
print(f"Extraindo de {INICIO_SEGUNDOS}s até {FIM_SEGUNDOS}s")
print(f"Frame inicial: {frame_inicio}")
print(f"Frame final:   {frame_fim}")
print(f"Frames esperados: {frame_fim - frame_inicio}")

# ============================================================
# IR DIRETAMENTE PARA O FRAME INICIAL
# ============================================================

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)

contador = 0

while True:

    frame_atual = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Parar ao chegar no final do intervalo
    if frame_atual >= frame_fim:
        break

    ret, frame = cap.read()

    if not ret:
        print("Erro ao ler frame.")
        break

    # Nome do arquivo
    nome = f"frame_{frame_atual:06d}.png"

    caminho = os.path.join(OUTPUT_DIR, nome)

    # Salvar frame
    cv2.imwrite(caminho, frame)

    contador += 1

# ============================================================
# FINALIZAR
# ============================================================

cap.release()

print()
print("========================================")
print("Extração concluída!")
print(f"Frames salvos: {contador}")
print(f"Pasta: {OUTPUT_DIR}")
print("========================================")