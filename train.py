from ultralytics import YOLO

# Load model YOLOv8 nano
model = YOLO("yolov8n.pt")

# Training
model.train(
    data="/content/data.yaml",
    epochs=40,
    imgsz=640,
    batch=8,
    project="/content/drive/MyDrive/runs",
    name="train"
)