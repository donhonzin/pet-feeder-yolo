from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11m.pt")

# Start training
model.train(
    data="dataset_custom.yaml",
    imgsz=640,
    batch=16,
    epochs=100,
    workers=0,
    device=0   
)
