import albumentations as A
import os
import cv2

def cargar_etiquetas(labelPath):
    with open(labelPath, "r") as file:
        labels = []
        for line in file:
            data = list(map(float, line.strip().split()))
            
            if(len(data) > 5): return None
            class_id, x_center, y_center, width, height = data
            if((x_center) < 0 or (y_center) < 0 or (width) < 0 or (height) < 0): 
                return None
            labels.append((int(class_id), [x_center, y_center, width, height]))
    return labels

transformar = A.Compose([
    A.Resize(640, 640),
    # Ajustes de color
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.6),
    A.RGBShift(p=0.3),
    A.RandomGamma(p=0.2),
    A.CLAHE(p=0.4),
    
    # Transformaciones geométricas
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
    
    # Distorsiones y otras transformaciones
    A.Perspective(p=0.7),
    A.GridDistortion(p=0.4),
    A.GaussNoise(p=0.2),
    A.MotionBlur(p=0.2),
    A.MedianBlur(p=0.2),
    A.Blur(p=0.2),
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.1))

INPUT_DIR = "dataset2/images/train"
LABEL_DIR = "dataset2/labels/train"
imgsToRemove = []

for imgNombre in os.listdir(INPUT_DIR):
    try:
        imgPath = os.path.join(INPUT_DIR, imgNombre)
        labelNombre = imgNombre.replace(".jpg", ".txt").replace(".png", ".txt")
        labelPath = os.path.join(LABEL_DIR, labelNombre)
        
        img = cv2.imread(imgPath)
        
        bboxesLabels = cargar_etiquetas(labelPath)
        
        bboxes = []
        classLabels = []
        
        for bBoxlabel in bboxesLabels:
            bboxes.append(bBoxlabel[1])
            classLabels.append(bBoxlabel[0])

        augmented = transformar(image=img, bboxes=bboxes, class_labels=classLabels)
        augmentedImg = augmented["image"]
        augmentedBboxes = augmented["bboxes"]
        
        #Generar labels:
        labelContenido = ""
        for i in range(len(augmentedBboxes)):
            label = augmented["class_labels"][i]
            bxs = augmented["bboxes"][i]
            labelContenido += f"{label} " + " ".join(map(str, bxs)) + "\n"

        augmentedImgPath = INPUT_DIR + "/" + f"augmented_{imgNombre}"
        augmentedTxtPath = LABEL_DIR + "/" + f"augmented_{labelNombre}"
        cv2.imwrite(augmentedImgPath, augmentedImg)
        with open(augmentedTxtPath, "w") as output:
            output.write(labelContenido)
    except ValueError as e:
        print("Error procesando " + imgPath)
    