import cv2
import os

# Raiz do projeto (dois niveis acima de implementacao/<pasta>/)
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Arquivo de entrada
INPUT = os.path.join(RAIZ, "resultados", "figuras_tcc", "frame_000842.png")

# Arquivo de saída
OUTPUT = os.path.join(RAIZ, "resultados", "figuras_tcc", "frame_cortado.png")

# Ler imagem
imagem = cv2.imread(INPUT)

if imagem is None:
    print("Erro: não foi possível abrir a imagem.")
    exit()

# Pegar dimensões
altura, largura = imagem.shape[:2]

# Definir onde cortar
# Mantém aproximadamente os 60% superiores da imagem
nova_altura = int(altura * 0.60)

# Fazer o corte
imagem_cortada = imagem[0:nova_altura, 0:largura]

# Salvar
cv2.imwrite(OUTPUT, imagem_cortada)

print("Imagem cortada com sucesso!")
print(f"Original: {largura} x {altura}")
print(f"Cortada:  {largura} x {nova_altura}")
print(f"Salva em: {OUTPUT}")