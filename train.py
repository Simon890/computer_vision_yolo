from ultralytics import YOLO
import torch
from googledriver import download_folder

URL_DATASET = "https://drive.google.com/drive/folders/1kJQ1aiFHODO-TxE4LkJ3_kePeMZUJvCO?usp=sharing"
print("Descargando dataset de google drive... Si la descarga no puede completarse entonces descargarlo manualmente")
download_folder(URL_DATASET)
print("Dataset descargado!")

torch.backends.cudnn.enabled = False

model = YOLO("yolov8s.pt")

model.train(data="dataset/dataset.yaml", epochs=20, imgsz=640, batch=16, lr0=0.001)
metrics = model.val()
path = model.export(format="onnx")
