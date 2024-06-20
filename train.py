from ultralytics import YOLO
import torch

torch.backends.cudnn.enabled = False

model = YOLO("yolov8s.pt")

model.train(data="dataset/dataset.yaml", epochs=20, imgsz=640, batch=16, lr0=0.001)
metrics = model.val()
path = model.export(format="onnx")
