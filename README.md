# Sistema de Vigilância Baseado em IA para Prevenção de Crimes com Armas e Disfarces

Este repositório contém os resultados de treinamento dos modelos **YOLOv8, YOLOv9 e YOLOv10** e o script principal para **detecção de objetos em vídeo**. O dataset completo não está incluído no repositório e pode ser baixado através do Google Drive.

---

## Estrutura do Repositório

```
tcc-yolo/
├── scripts/                 
│   ├── detectar_video.py    
│   └── train_yolo_colab.py
├── models/              
│   ├── yolov8/              
│   │   ├── best.pt
│   │   └── last.pt
│   ├── yolov9/              
│   │   ├── best.pt
│   │   └── last.pt
│   └── yolov10/             
│       ├── best.pt
│       └── last.pt
├── results/              
│   ├── yolov8/              
│   │   └── resultados dos do treinamento
│   ├── yolov9/              
│   │   └── resultados dos do treinamento
│   └── yolov10/             
│       └── resultados dos do treinamento              
├── docs/                    
├── logs/                    
├── links_dataset.txt        
└── README.md                
```

---

## Dataset

O **dataset completo** está hospedado no Google Drive devido ao seu tamanho. Para baixá-lo, consulte o arquivo `links_dataset.txt`.

O dataset inclui imagens divididas em:

* `train/` – conjunto de treinamento
* `valid/` – conjunto de validação
* `test/` – conjunto de teste

Técnicas de **data augmentation** aplicadas:

* Giro e espelhamento (horizontal/vertical)
* Ajuste de brilho, contraste e saturação
* Inserção de ruídos aleatórios (Gaussian Noise)
* Corte aleatório (random crop)
* Zoom e rotação leve (affine transform)

---

## Requisitos

* Python 3.10 ou superior
* Bibliotecas principais:

```bash
pip install ultralytics opencv-python numpy
```

* Recomendado: GPU para processamento em tempo real

---

## Executando a Aplicação

1. Baixe os pesos desejados (`best.pt` ou `last.pt`) de `models/yolov8/`, `yolov9/` ou `yolov10/`.
2. Coloque o vídeo de teste na pasta `videos/` (crie a pasta se não existir) e ajuste o caminho no script `scripts/detectar_video.py`.
3. Execute o script:

```bash
python scripts/detectar_video.py
```

### O que o script faz:

* Processa o vídeo frame a frame
* Desenha **bounding boxes** com cores únicas por classe
* Salva todas as detecções em `logs/log_deteccoes.txt`
* Exibe o vídeo em tempo real (pressione `ESC` para sair)

> Observação: Por padrão, o script carrega o modelo YOLOv8 (`best.pt`). Para usar YOLOv9 ou YOLOv10, altere o caminho do modelo dentro do script.

---

## Treinamento no Google Colab

Para treinar qualquer modelo (v8, v9 ou v10), utilize o script `train_yolo_colab.py`:

1. Suba o arquivo `.zip` do dataset no Google Drive.
2. Abra o Colab e execute o script.
3. Escolha o modelo através da variável `modelo_escolhido` (`v8`, `v9` ou `v10`).
4. O script treina o modelo, salva os resultados em `/content/TCC/resultados/yolov{modelo}` e gera um arquivo `.zip` para download.

---

## Contato

* Autor: Elton Barbosa
* GitHub: [https://github.com/eltonbarbosaa/tcc-yolo](https://github.com/eltonbarbosaa/tcc-yolo)
* E-mail: [elton.baarbosa@gmail.com](mailto:elton.baarbosa@gmail.com)
