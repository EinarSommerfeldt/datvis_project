from ultralytics import YOLO

RDD2022 = "C:/Users/einar/Desktop/datvis_project/RDD2022_dataset/RDD2022_released_through_CRDDC2022/RDD2022/"
filepath = RDD2022 + "Norway/train/images/"

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
#model = YOLO('path/to/best.pt')  # load a custom model

model.train(data='C:/Users/einar/Desktop/datvis_project/yolo/test.yaml', epochs=100, imgsz=640)