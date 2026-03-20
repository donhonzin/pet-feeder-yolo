from ultralytics import YOLO

model = YOLO("C:/Users/Xuxuzim/Desktop/yolov3/runs/detect/train/weights/best.pt")
model.export(format="ncnn")