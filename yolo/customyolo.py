from custom_model import YOLO_custom
from ultralytics import YOLO

# Load a model
model = YOLO_custom('yolov8n.pt')  # load an official model
#model = YOLO('yolov8n.pt')

model.train(data='C:/Users/einar/Desktop/datvis_project/yolo/rdd2022.yaml', epochs=100, imgsz=640)