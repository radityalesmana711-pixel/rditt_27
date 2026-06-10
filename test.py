import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("models/best.pt")

# Buka kamera
cap = cv2.VideoCapture(0)

# Jika kamera tidak terdeteksi
if not cap.isOpened():

    # Membuat gambar putih untuk popup
    popup = np.ones((200, 600, 3), dtype=np.uint8) * 255

    # Pesan error
    cv2.putText(
        popup,
        "Kamera tidak Terdeteksi!",
        (70, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        popup,
        "Tolong Cek Koneksi Webcam Kamu",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    cv2.imshow("Error", popup)

    # Popup tampil 5 detik
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    exit()

print("Kamera berhasil Terdeteksi!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca Frame!")
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("Pendeteksi Kantuk", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()