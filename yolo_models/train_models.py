# YOLOv8 
from ultralytics import YOLO

data_yaml = 'yolo_models/data.yaml' 
weights = 'yolov8n.pt'   

model = YOLO(weights)  
model.train(
    data=data_yaml,
    epochs=100,
    patience=20, # number of epochs with no improvement before stopping
    imgsz=768, # size of input images, bigger because the cells are small and dense 
    batch=16,
    optimizer='Adam', # optimizer to use
    lr0=0.001, # initial learning rate
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    name="yolo_detector_2107_100_20_16_768",
    project="yolo_models/models",
    pretrained=True
)

