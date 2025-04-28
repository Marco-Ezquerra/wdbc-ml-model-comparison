import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# === Configuracion de rutas ===
BASE_PATH = 'C:/Users/Usuario/Desktop/curso IA'
DATA_PATH = os.path.join(BASE_PATH, 'wdbc.data')
OUTPUT_PATH = os.path.join(BASE_PATH, 'graficas')
RESULTS_FILE = os.path.join(BASE_PATH, 'resultados.csv')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# === Carga de datos ===
try:
    datos = pd.read_csv(DATA_PATH, header=None)
except FileNotFoundError:
    raise FileNotFoundError(f"El archivo no se encontró en {DATA_PATH}")


datos.columns = ['ID', 'Diagnostico'] + [f'Caracteristica_{i}' for i in range(1, 31)]
datos = datos.drop(columns=['ID'])
datos['Diagnostico'] = datos['Diagnostico'].map({'M': 1, 'B': 0})


X = datos.drop(columns=['Diagnostico'])
y = datos['Diagnostico']

# Normalizacion  imporante !!
scaler = StandardScaler()
X = scaler.fit_transform(X)

#  resultados 
resultados = []

# === Entrenamiento y evaluacion ===
for tamaño_entrenamiento in tqdm(range(10, 91, 1), desc='Procesando porcentajes'):
    # División de datos
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, train_size=tamaño_entrenamiento/100, random_state=42)

    # Regresion Logistica
    modelo_logistica = LogisticRegression(max_iter=10000)
    modelo_logistica.fit(X_entrenamiento, y_entrenamiento)
    precision_entrenamiento_logistica = accuracy_score(y_entrenamiento, modelo_logistica.predict(X_entrenamiento))
    precision_prueba_logistica = accuracy_score(y_prueba, modelo_logistica.predict(X_prueba))

    # buscando el mejor k
    mejor_k = 1
    mejor_precision_entrenamiento_knn = 0
    mejor_precision_prueba_knn = 0
    for k in range(1, 21):
        modelo_knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        modelo_knn.fit(X_entrenamiento, y_entrenamiento)
        precision_entrenamiento = accuracy_score(y_entrenamiento, modelo_knn.predict(X_entrenamiento))
        precision_prueba = accuracy_score(y_prueba, modelo_knn.predict(X_prueba))
        if precision_prueba > mejor_precision_prueba_knn:
            mejor_precision_entrenamiento_knn = precision_entrenamiento
            mejor_precision_prueba_knn = precision_prueba
            mejor_k = k

    # SVM lineal
    modelo_svm = SVC(kernel='linear')
    modelo_svm.fit(X_entrenamiento, y_entrenamiento)
    precision_entrenamiento_svm = accuracy_score(y_entrenamiento, modelo_svm.predict(X_entrenamiento))
    precision_prueba_svm = accuracy_score(y_prueba, modelo_svm.predict(X_prueba))

    # Guardar resultados de esta iteracion
    resultados.append({
        'tamaño_entrenamiento': tamaño_entrenamiento,
        'precision_entrenamiento_logistica': precision_entrenamiento_logistica,
        'precision_prueba_logistica': precision_prueba_logistica,
        'precision_entrenamiento_knn': mejor_precision_entrenamiento_knn,
        'precision_prueba_knn': mejor_precision_prueba_knn,
        'mejor_k': mejor_k,
        'precision_entrenamiento_svm': precision_entrenamiento_svm,
        'precision_prueba_svm': precision_prueba_svm
    })

# Convertir resultados a DataFrame
df_resultados = pd.DataFrame(resultados)


df_resultados.to_csv(RESULTS_FILE, index=False)

# === Graficas ===

# Precision  prueba
plt.figure(figsize=(12, 8))
plt.plot(df_resultados['tamaño_entrenamiento'], df_resultados['precision_prueba_logistica'], label='Regresion Logistica (Prueba)', marker='o')
plt.plot(df_resultados['tamaño_entrenamiento'], df_resultados['precision_prueba_knn'], label='k-NN (Prueba, Mejor k)', marker='s')
plt.plot(df_resultados['tamaño_entrenamiento'], df_resultados['precision_prueba_svm'], label='SVM (Prueba)', marker='^')
plt.xlabel('Porcentaje de Datos de Entrenamiento (%)')
plt.ylabel('Precision')
plt.title('Precision en el Conjunto de Prueba')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_PATH, 'precision_vs_entrenamiento_prueba.png'))
plt.show()

# Precision entrenamiento
plt.figure(figsize=(12, 8))
plt.plot(df_resultados['tamaño_entrenamiento'], df_resultados['precision_entrenamiento_logistica'], label='Regresion Logistica (Entrenamiento)', linestyle='--', marker='o')
plt.plot(df_resultados['tamaño_entrenamiento'], df_resultados['precision_entrenamiento_knn'], label='k-NN (Entrenamiento, Mejor k)', linestyle='--', marker='s')
plt.plot(df_resultados['tamaño_entrenamiento'], df_resultados['precision_entrenamiento_svm'], label='SVM (Entrenamiento)', linestyle='--', marker='^')
plt.xlabel('Porcentaje de Datos de Entrenamiento (%)')
plt.ylabel('Precision')
plt.title('Precision en el Conjunto de Entrenamiento')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_PATH, 'precision_vs_entrenamiento_entrenamiento.png'))
plt.show()

# Mejor valor de k
plt.figure(figsize=(12, 8))
plt.plot(df_resultados['tamaño_entrenamiento'], df_resultados['mejor_k'], label='Mejor k en k-NN', marker='o')
plt.xlabel('Porcentaje de Datos de Entrenamiento (%)')
plt.ylabel('Valor de k')
plt.title('Evolucion del Mejor k segun tamaño de entrenamiento')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_PATH, 'mejor_k_vs_entrenamiento.png'))
plt.show()
