import cv2

video_original = "./videos/GTAVI_Trailer_2_6s_26s_2k.mp4"
video_acelerado = "./gta_acelerado.mp4"
video_saida = "./gta_side_by_side.mp4"

cap_orig = cv2.VideoCapture(video_original)
cap_acel = cv2.VideoCapture(video_acelerado)

# 1. Obter taxas e contagens de quadros
fps_orig = cap_orig.get(cv2.CAP_PROP_FPS)
fps_acel = cap_acel.get(cv2.CAP_PROP_FPS)

# Fator de repetição para sincronizar o vídeo original com a alta taxa do acelerado
repeat_factor = int(round(fps_acel / fps_orig)) if fps_orig > 0 else 4
print(f"FPS Original: {fps_orig:.2f} | FPS Acelerado: {fps_acel:.2f}")
print(f"Fator de repetição do quadro original: {repeat_factor}x")

# Dimensões da janela combinada (1920x540 total)
TARGET_WIDTH = 960
TARGET_HEIGHT = 540
width_combined = TARGET_WIDTH * 2
height_combined = TARGET_HEIGHT

# 2. Configuração do VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_saida, fourcc, fps_acel, (width_combined, height_combined))

ret_orig, frame_o = cap_orig.read()
count_orig_hold = 0

frame_idx = 0
while True:
    ret_acel, frame_a = cap_acel.read()
    if not ret_acel:
        break

    # Avança 1 quadro no vídeo original a cada 'repeat_factor' quadros do vídeo acelerado
    if count_orig_hold % repeat_factor == 0 and count_orig_hold != 0:
        ret_orig, frame_o = cap_orig.read()

    if not ret_orig or frame_o is None:
        break

    count_orig_hold += 1

    # Redimensionar ambas as metades
    frame_o_resized = cv2.resize(frame_o, (TARGET_WIDTH, TARGET_HEIGHT))
    frame_a_resized = cv2.resize(frame_a, (TARGET_WIDTH, TARGET_HEIGHT))

    # Marca d'água / Identificadores visuais
    cv2.putText(frame_o_resized, "Original (30 FPS)", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame_a_resized, "SepConv Interpolado (120 FPS)", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Concatenação horizontal
    combined_frame = cv2.hconcat([frame_o_resized, frame_a_resized])
    out.write(combined_frame)
    frame_idx += 1

cap_orig.release()
cap_acel.release()
out.release()

print(f"Vídeo comparativo sincronizado gerado com sucesso ({frame_idx} quadros): {video_saida}")