from ultralytics import YOLO


model = YOLO("yolo11_custom.pt")

model.predict("1.mp4", show=True, save=True, conf=0.7, classes=[0,1,2])