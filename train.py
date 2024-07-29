from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import pandas as pd
import json
from googledriver import download_folder

URL_DATASET = "https://drive.google.com/drive/folders/1kJQ1aiFHODO-TxE4LkJ3_kePeMZUJvCO?usp=sharing"
print("Descargando dataset de google drive... Si la descarga no puede completarse entonces descargarlo manualmente")
download_folder(URL_DATASET)
print("Dataset descargado!")

torch.backends.cudnn.enabled = False

model = YOLO("yolov8s.pt")

model.train(data="dataset2/dataset.yaml", epochs=40, imgsz=640, batch=16, lr0=0.002, save_json=True)
results = model.val(save_json=True)

with open("metricas", "w") as output:
    output.write(json.dumps(results.results_dict))

jsonDataPorClase = {}
for i in range(len(results.names)):
    try:
        data = results.class_result(i)
        className = results.names[i]
        jsonDataPorClase[className] = {
            "precision": data[0],
            "recall": data[1],
            "map50": data[2],
            "map5095": data[3]
        }
    except:
        pass

with open("m_por_clase.json", "w") as file:
    file.write(json.dumps(jsonDataPorClase))
