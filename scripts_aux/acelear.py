import cv2

video_original_path = "./videos/GTAVI_Trailer_2_6s_26s_2k.mp4"
video_interpolado_path = "./gta.mp4"
video_saida_path = "./gta_acelerado.mp4"

# 1. Obter a duração do vídeo original (em segundos)
cap_orig = cv2.VideoCapture(video_original_path)
fps_orig = cap_orig.get(cv2.CAP_PROP_FPS)
total_frames_orig = cap_orig.get(cv2.CAP_PROP_FRAME_COUNT)
duracao_segundos = total_frames_orig / fps_orig
cap_orig.release()

# 2. Ler o vídeo interpolado gerado pelo SepConv
cap_interp = cv2.VideoCapture(video_interpolado_path)
total_frames_interp = cap_interp.get(cv2.CAP_PROP_FRAME_COUNT)
width = int(cap_interp.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_interp.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 3. Calcular o novo FPS necessário para durar exatamente a mesma quantidade de segundos
fps_saida_acelerado = total_frames_interp / duracao_segundos

print(f"--- PARÂMETROS TEMPORAIS ---")
print(f"Duração Original: {duracao_segundos:.2f}s | FPS Original: {fps_orig:.2f}")
print(f"Total de Frames Interpolados: {total_frames_interp}")
print(f"Novo FPS do Vídeo Acelerado: {fps_saida_acelerado:.2f} FPS")

# 4. Gravar o novo MP4 forçando o novo FPS de exibição
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_saida_path, fourcc, fps_saida_acelerado, (width, height))

while cap_interp.isOpened():
    ret, frame = cap_interp.read()
    if not ret:
        break
    out.write(frame)

cap_interp.release()
out.release()

print(f"\nVídeo final acelerado e sincronizado salvo em: {video_saida_path}")