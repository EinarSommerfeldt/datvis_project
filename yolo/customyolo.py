from custom_model import YOLO_custom
from ultralytics import YOLO

# Load a model
model = YOLO_custom('/cluster/home/einarjso/datvis_project/runs/detect/train3/weights/epoch32.pt')  # load an official model
#model = YOLO('yolov8n.pt')

model.train(data='/cluster/home/einarjso/datvis_project/yolo/rdd2022.yaml', epochs=100, imgsz=640, save_period=1, workers=4)
