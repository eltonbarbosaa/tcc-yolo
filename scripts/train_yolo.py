from google.colab import drive
import zipfile
import os
import shutil
from ultralytics import YOLO
from google.colab import files

# ----------------------------
# CONFIGURAÇÃO DO TREINO
# ----------------------------
modelo_escolhido = 'v8'  # Opções: 'v8', 'v9', 'v10'
epochs = 30
imgsz = 640

# Caminhos
zip_path = '/content/drive/My Drive/TCC.zip'  # Zip do dataset
unzip_dir = '/content'
pesos_base = {
    'v8': '/content/TCC/pesos_base/yolov8n.pt',
    'v9': '/content/TCC/pesos_base/yolov9n.pt',
    'v10': '/content/TCC/pesos_base/yolov10n.pt'
}

# ----------------------------
# Montar Google Drive e extrair dataset
# ----------------------------
drive.mount('/content/drive')

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_dir)

dataset_dir = '/content/TCC/imagens'
os.listdir(dataset_dir)

# Instalar ultralytics
!pip install -q ultralytics

# ----------------------------
# Inicializar o modelo selecionado
# ----------------------------
modelo = YOLO(pesos_base[modelo_escolhido])

# Treinar
results = modelo.train(
    data=os.path.join(dataset_dir, 'data.yaml'),
    epochs=epochs,
    imgsz=imgsz,
    project='/content/TCC/resultados',
    name=f'yolov{modelo_escolhido}',
    exist_ok=True
)

# ----------------------------
# Compactar resultados para download
# ----------------------------
resultados_path = f'/content/TCC/resultados/yolov{modelo_escolhido}'
zip_path = f'/content/yolov{modelo_escolhido}_resultados.zip'
shutil.make_archive(f'/content/yolov{modelo_escolhido}_resultados', 'zip', resultados_path)

files.download(zip_path)
