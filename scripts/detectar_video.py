import cv2
import random
from datetime import datetime
from ultralytics import YOLO

# Carrega o modelo YOLO treinado
modelo = YOLO(r'C:\Users\elton\OneDrive\Área de Trabalho\TCC\deteccao_invasao_alerta-main\best.pt')

# Dicionário para armazenar cores únicas por classe
cores_classes = {}

def cor_por_classe(nome_classe):
    if nome_classe not in cores_classes:
        cores_classes[nome_classe] = tuple(random.randint(0, 255) for _ in range(3))
    return cores_classes[nome_classe]

# Abre o vídeo
video = cv2.VideoCapture(r'C:\Users\elton\OneDrive\Área de Trabalho\TCC\deteccao_invasao_alerta-main\ex06.mp4')

# Nome do arquivo de log
caminho_log = r'C:\Users\elton\OneDrive\Área de Trabalho\TCC\deteccao_invasao_alerta-main\log_deteccoes.txt'

# Função para registrar no log
def registrar_log(nome, confianca):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    linha = f"[{timestamp}] Objeto detectado: {nome} ({confianca*100:.1f}%)\n"
    with open(caminho_log, 'a', encoding='utf-8') as f:
        f.write(linha)

# Loop de detecção
while True:
    check, img = video.read()
    if not check or img is None:
        print("Fim do vídeo ou erro ao ler frame.")
        break

    resultado = modelo.predict(img, verbose=False)

    for obj in resultado:
        nomes = obj.names
        for item in obj.boxes:
            x1, y1, x2, y2 = map(int, item.xyxy[0])
            cls = int(item.cls[0])
            nomeClasse = nomes[cls]
            conf = round(float(item.conf[0]), 2)
            texto = f'{nomeClasse} - {conf}'

            # Desenho
            cor = cor_por_classe(nomeClasse)
            cv2.rectangle(img, (x1, y1), (x2, y2), cor, 3)
            cv2.putText(img, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

            # Salva no log
            registrar_log(nomeClasse, conf)

    cv2.imshow('Detecção de Objetos', img)

    if cv2.waitKey(1) == 27:  # ESC para sair
        break

video.release()
cv2.destroyAllWindows()
