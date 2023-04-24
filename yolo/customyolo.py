from custom_model import YOLO_custom

# Load a model
model = YOLO_custom('yolov8n.pt')  # load an official model

model.train(data='C:/Users/einar/Desktop/datvis_project/yolo/test.yaml', epochs=100, imgsz=640)