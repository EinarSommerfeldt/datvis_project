from custom_model import YOLO_custom
from ultralytics import YOLO

# Load a model
model = YOLO_custom('/content/gdrive/MyDrive/repos/datvis_project/train/weights/32ep_fullset.pt') 

model.train(data='/content/gdrive/MyDrive/repos/datvis_project/train/colab.yaml', epochs=30, imgsz=640, save_period=1, workers=4)
