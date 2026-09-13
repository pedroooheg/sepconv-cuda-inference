import cv2

input_video = "videos/Lucia_Caminos_Video_Clip.mp4"
output_video = "videos/Lucia_1280.mp4"

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

new_width = 1280
new_height = int(height * (new_width / width))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (new_width, new_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (new_width, new_height))
    out.write(frame)

cap.release()
out.release()

print(f"Salvo em: {output_video}")
print(f"Nova resolução: {new_width}x{new_height}")