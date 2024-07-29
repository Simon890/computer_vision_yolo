import os
import shutil

def limpiar_dataset(dir):
    INPUT_DIR = "./dataset/images/" + dir
    LABEL_DIR = "./dataset/labels/" + dir
    OUT_DIR = "./dataset2/images/" + dir
    LABEL_OUT_DIR = "./dataset2/labels/" + dir
    imgsEliminar = []
    for imgPath in os.listdir(INPUT_DIR):
        labelNombre = imgPath.replace(".jpg", ".txt").replace(".png", ".txt")
        labelPath = LABEL_DIR + "/" + labelNombre
        if not os.path.exists(labelPath): 
            imgsEliminar.append(imgPath)
            continue
        
        shutil.copy2(INPUT_DIR + "/" + imgPath, OUT_DIR + "/" + imgPath)
        with open(labelPath, "r") as file, open(LABEL_OUT_DIR + "/" + labelNombre, "w") as outFile:
            for line in file:
                partes = line.strip().split()
                classId = None
                xCentro = None
                yCentro = None
                width = None
                height = None
                
                if(len(partes) == 0):
                    imgsEliminar.append(imgPath)
                    break
                
                if len(partes) == 5:
                    classId = int(partes[0])
                    xCentro = float(partes[1])
                    yCentro = float(partes[2])
                    width = float(partes[3])
                    height = float(partes[4])
                elif len(partes) == 9:
                    classId = int(partes[0])
                    x1 = float(partes[1])
                    y1 = float(partes[2])
                    x2 = float(partes[3])
                    y2 = float(partes[4])
                    x3 = float(partes[5])
                    y3 = float(partes[6])
                    x4 = float(partes[7])
                    y4 = float(partes[8])
                    xCentro = (x1 + x2 + x3 + x4) / 4
                    yCentro = (y1 + y2 + y3 + y4) / 4
                    width = max(x2, x3) - min(x1, x4)
                    height = max(y3, y4) - min(y1, y2)
                
                outFile.write(f"{classId} {xCentro} {yCentro} {width} {height}\n")


limpiar_dataset("train")
limpiar_dataset("val")
# for imgPath in imgsEliminar:
#     labelNombre = imgPath.replace(".jpg", ".txt").replace(".png", ".txt")
#     labelPath = LABEL_DIR + "/" + labelNombre
#     if os.path.exists(labelPath):
#         os.remove(labelPath)