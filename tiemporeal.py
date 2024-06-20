import cv2
from ultralytics import YOLO

title_win = "Deteccion de cartas"
trackbar_name = "Confianza"
slider_max = 100
certeza = 25

def on_trackbar(val):
    global certeza
    certeza = val / 100

def escribir_texto(img, texto):
    return cv2.putText(img, texto, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

cv2.namedWindow(title_win)
cv2.createTrackbar(trackbar_name, title_win, certeza, slider_max, on_trackbar)
cap = cv2.VideoCapture(0)
model = YOLO("./runs/detect/train/weights/best.pt")
while True:
    ret, frame = cap.read()
    if frame is None: break
    
    results = model.predict(source=frame, conf=certeza, show_labels=True, show_conf=True, imgsz=640, verbose=False)
    frame_predict = None
    
    cantCartas = len(results[0].boxes)
    result = results[0]
    
    if cantCartas > 0:
        frame_predict = result.plot()
    
    if cantCartas > 3:
        frame_predict = escribir_texto(frame_predict, "Sobra(n) " + str(cantCartas - 3) + " carta(s)")
    elif cantCartas < 3:
        frame_predict = escribir_texto(frame_predict, "Falta(n) " + str(3 - cantCartas) + " carta(s)")
    else:
        totalNums = []
        tipos = []
        hayEspecial = False
        for box in result.boxes:
            clase = int(box.cls[0])
            claseNombre = model.names[clase]
            numero = int(claseNombre[:-1])
            tipo = claseNombre[-1]
            if tipo in tipos:
                totalNums.append(numero)
            tipos.append(tipo)
            if numero == 8 or numero == 9 or claseNombre == "J":
                frame_predict = escribir_texto(frame_predict, "8, 9 y Comodines no pueden ser usados")
                hayEspecial = True
                break
            
        if not hayEspecial:
            total = 0
            sumarVeinte = False
            for num in totalNums:
                if num in [10, 11, 12]:
                    sumarVeinte = True
                    continue
                total += num
            if sumarVeinte:
                total += 20
            if total >= 28:
                frame_predict = escribir_texto(frame_predict, "Envido!")
            
    if frame_predict is None:
        cv2.imshow(title_win, frame)
    else: 
        cv2.imshow(title_win, frame_predict)
    
    #Salir al presionar la letra q
    key = cv2.waitKey(30)
    if key == ord("q") or key == 27: break
