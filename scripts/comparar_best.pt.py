# === 1. Bibliotecas ===
import zipfile, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# === 2. Upload dos arquivos .zip ===
from google.colab import files
uploaded = files.upload()  # Envie os 3 arquivos zip: yolov8_resultados.zip, yolov9_resultados.zip, yolov10_resultados.zip

# === 3. Extração dos arquivos ===
def extrair_zip(nome_arquivo):
    nome_pasta = nome_arquivo.replace('.zip', '')
    with zipfile.ZipFile(nome_arquivo, 'r') as zip_ref:
        zip_ref.extractall(nome_pasta)
    return nome_pasta

caminhos = [extrair_zip(nome) for nome in uploaded.keys()]

# === 4. Leitura dos arquivos results.csv ===
def carregar_resultados(pasta):
    caminho_csv = os.path.join(pasta, 'results.csv')
    return pd.read_csv(caminho_csv).tail(1)  # última linha (resultados finais)

# Carrega em um dicionário
dados = {
    'YOLOv8': carregar_resultados(caminhos[0]),
    'YOLOv9': carregar_resultados(caminhos[1]),
    'YOLOv10': carregar_resultados(caminhos[2])
}

# === 5. Criação de DataFrame comparativo ===
comparativo = pd.DataFrame({
    'Modelo': list(dados.keys()),
    'Precisão': [dados[m]['metrics/precision(B)'].values[0] for m in dados],
    'Revocação': [dados[m]['metrics/recall(B)'].values[0] for m in dados],
    'mAP@0.5': [dados[m]['metrics/mAP50(B)'].values[0] for m in dados],
    'mAP@0.5:0.95': [dados[m]['metrics/mAP50-95(B)'].values[0] for m in dados],
    'Box Loss': [dados[m]['val/box_loss'].values[0] for m in dados],
    'Cls Loss': [dados[m]['val/cls_loss'].values[0] for m in dados],
    'DFL Loss': [dados[m]['val/dfl_loss'].values[0] for m in dados],
    'Tempo (s)': [dados[m]['time'].values[0] for m in dados]
})

# Exibir a tabela
print("📊 Comparativo entre modelos YOLO:\n")
display(comparativo)

# === 6. Gráficos ===
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Análise Comparativa de Desempenho dos Modelos YOLO", fontsize=18)

# Precisão e Revocação
sns.barplot(x='Modelo', y='value', hue='variable',
            data=pd.melt(comparativo[['Modelo', 'Precisão', 'Revocação']], ['Modelo']),
            ax=axs[0, 0])
axs[0, 0].set_title('Precisão vs Revocação')
axs[0, 0].set_ylim(0, 1)

# mAP
sns.barplot(x='Modelo', y='value', hue='variable',
            data=pd.melt(comparativo[['Modelo', 'mAP@0.5', 'mAP@0.5:0.95']], ['Modelo']),
            ax=axs[0, 1])
axs[0, 1].set_title('mAP Comparativo')
axs[0, 1].set_ylim(0, 1)

# Perdas
sns.barplot(x='Modelo', y='value', hue='variable',
            data=pd.melt(comparativo[['Modelo', 'Box Loss', 'Cls Loss', 'DFL Loss']], ['Modelo']),
            ax=axs[1, 0])
axs[1, 0].set_title('Perdas de Validação')

# Tempo
sns.barplot(x='Modelo', y='Tempo (s)', data=comparativo, ax=axs[1, 1], color='orange')
axs[1, 1].set_title('Tempo de Treinamento (segundos)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# === 7. Conclusão preliminar ===
melhor_mAP = comparativo.sort_values(by='mAP@0.5', ascending=False).iloc[0]
print(f"\n🏁 Conclusão preliminar:\nO modelo com melhor desempenho em mAP@0.5 foi o **{melhor_mAP['Modelo']}** "
      f"com {melhor_mAP['mAP@0.5']:.4f} de acurácia.\n")

print("Análise adicional pode ser feita com base na precisão, revocação, perdas e tempo de execução.")
