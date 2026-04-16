from ultralytics import YOLO

model = YOLO("yolo11_custom.pt")

metrics = model.val(data="dataset_custom.yaml", split="test")