# YOLOv11
from ultralytics import YOLO

data_yaml = 'yolo_models/data.yaml' 
weights = 'yolo11s.pt'   

model = YOLO(weights)  
model.train(
    data=data_yaml,
    epochs=100,
    patience=20, 
    imgsz=768, 
    batch=16,
    optimizer='Adam', 
    lr0=0.001, 
    degrees=10,  
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    name="16.0.001.augmentacja15_test22_adam", 
    project="yolo_models/models",
    pretrained=True
)

