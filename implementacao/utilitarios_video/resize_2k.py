import cv2
import os

# Raiz do projeto (dois niveis acima de implementacao/<pasta>/)
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

input_video = os.path.join(RAIZ, "dados", "videos", "GTAVI_Trailer_2_6s_26s.mp4")
output_video = os.path.join(RAIZ, "dados", "videos", "GTAVI_Trailer_2_6s_26s_2k.mp4")

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)

new_width = 2560
new_height = 1440

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (new_width, new_height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    out.write(frame)

cap.release()
out.release()

print(f"Vídeo salvo em: {output_video}")
print(f"Resolução: {new_width}x{new_height}")