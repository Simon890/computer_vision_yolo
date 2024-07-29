-----------------------------------------------------------
ATENCION!!!
-----------------------------------------------------------
 Una vez subido al drive, deben ajustar la ruta en el archivo data/eval/dataset.yaml en la clave "path" para apunte al directorio padre de ese archivo, como por ejemplo:
 
path: /content/drive/MyDrive/TUIA/CV/2024_01/final/data/eval/

 Dentro de la ruta del ejemplo, se deben encontrar los directorios images y lables, y los archivos dataset.yaml y gt_envido.json

 De lo contrario, Fiftyone no podrá importar el dataset.
