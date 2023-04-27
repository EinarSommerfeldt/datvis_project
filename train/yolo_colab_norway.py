from custom_model import YOLO_custom
from ultralytics import YOLO

# Load a model
model = YOLO_custom('/content/gdrive/MyDrive/repos/datvis_project/train/weights/33ep_fullset.pt') 

model.train(data='/content/gdrive/MyDrive/repos/datvis_project/train/colab_norway.yaml', epochs=10, imgsz=1280, save_period=1, workers=8)
