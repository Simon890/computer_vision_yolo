import PIL.Image as Image
import gradio as gr

from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")


def predecirCartas(img, conf_threshold, iou_threshold):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im


iface = gr.Interface(
    fn=predecirCartas,
    inputs=[
        gr.Image(type="pil", label="Subir imagen"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Umbral de confianza"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="Umbral IoU")
    ],
    outputs=gr.Image(type="pil", label="Resultado"),
    title="Simón Revello - Trabajo Práctico - Computer Vision",
    description="Detección de cartas"
)

if __name__ == '__main__':
    iface.launch()