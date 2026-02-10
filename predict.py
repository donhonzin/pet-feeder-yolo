from ultralytics import YOLO


model = YOLO("yolo11_custom.pt")

model.predict("2.mp4", show=True, save=True, conf=0.7, classes=[5,7])