from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

model.train(
    data="dataset.yaml",
    epochs=40,
    imgsz=416,
    batch=8,
    name="mask_detector",
    patience=7
)
