# YOLOv8 
from ultralytics import YOLO

# === ŚCIEŻKI ===
data_yaml = 'data.yaml'  # plik konfiguracyjny YOLO
weights = 'yolov8n.pt'   # pretrenowany model do fine-tuningu

# === TRENING ===
model = YOLO(weights)  # załaduj pretrenowany model
model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=8,
    name="yolo_cells_detector_hybrid",
    project="runs",
    pretrained=True
)

