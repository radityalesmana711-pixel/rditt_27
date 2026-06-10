import cv2
import os

# Path dataset
TRAIN_IMAGES = "data/images/train"
VAL_IMAGES = "data/images/val"

TRAIN_LABELS = "data/labels/train"
VAL_LABELS = "data/labels/val"

# Buat folder labels jika belum bikin
os.makedirs(TRAIN_LABELS, exist_ok=True)
os.makedirs(VAL_LABELS, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Fungsi convert ke format YOLO

def convert_to_yolo(size, box):
    img_w, img_h = size
    x, y, w, h = box

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h

    width = w / img_w
    height = h / img_h

    return x_center, y_center, width, height

# Fungsi auto annotation

def annotate_folder(images_path, labels_path):

    for filename in os.listdir(images_path):

        if filename.endswith((".png", ".jpg", ".jpeg")):

            image_path = os.path.join(images_path, filename)

            image = cv2.imread(image_path)

            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20)
            )

            img_h, img_w = image.shape[:2]

            # Tentukan class otomatis
            if "awake" in filename.lower():
                class_id = 0

            elif "drowsy" in filename.lower():
                class_id = 1

            else:
                continue

            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(labels_path, label_filename)

            with open(label_path, "w") as f:

                if len(faces) == 0:
                    print(f"Wajah tidak terdeteksi: {filename}")
                    continue

                for (x, y, w, h) in faces:

                    x_center, y_center, width, height = convert_to_yolo(
                        (img_w, img_h),
                        (x, y, w, h)
                    )

                    f.write(
                        f"{class_id} {x_center} {y_center} {width} {height}\n"
                    )

            print(f"Berhasil annotate: {filename}")

# Ini Kodingan buat jalanin annotation otomatis yagesyaa..
print("Memulai auto annotation train...")
annotate_folder(TRAIN_IMAGES, TRAIN_LABELS)

print("Memulai auto annotation val...")
annotate_folder(VAL_IMAGES, VAL_LABELS)

print("Selesai auto annotation!")