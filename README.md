# wdbc-ml-model-comparison


# Comparativa de Modelos de Machine Learning: WDBC Dataset

Este proyecto analiza el rendimiento de tres modelos de Machine Learning sobre el conjunto de datos **WDBC (Wisconsin Diagnostic Breast Cancer)**, variando el porcentaje de datos de entrenamiento desde el 10% hasta el 90%. Se evalúan tanto las precisiones de entrenamiento como de prueba, además de optimizar el hiperparámetro `k` en el clasificador k-NN.

Además, se introducen fundamentos teóricos de cada modelo empleado basados en el manual "Introducción a la Inteligencia Artificial" (Universidad de Salamanca, 2021).

---

## Contenidos

- **Carga y preprocesamiento de datos**
- **Normalización de características**
- **Entrenamiento y evaluación de modelos:**
  - Regresión Logística
  - k-Nearest Neighbors (k-NN)
  - Máquinas de Vectores de Soporte (SVM) con kernel lineal
- **Optimización del hiperparámetro `k` en k-NN**
- **Análisis del rendimiento en función del tamaño de entrenamiento**
- **Visualización de resultados**
- **Exportación de resultados en CSV**

---

## Fundamentos Teóricos

### Regresión Logística
La regresión logística es un modelo supervisado utilizado para problemas de clasificación binaria. Estima la probabilidad de que una instancia pertenezca a una clase específica usando la función logística (sigmoide), convirtiendo cualquier valor real en un rango de 0 a 1. Se optimiza mediante técnicas de descenso de gradiente.

### k-Nearest Neighbors (k-NN)
k-NN es un algoritmo de clasificación supervisado basado en instancias. La clasificación de una nueva muestra se determina observando las etiquetas de sus "k" vecinos más cercanos en el espacio de características. No realiza un aprendizaje explícito: simplemente almacena el dataset completo. La elección de un buen `k` es crítica para su rendimiento.

### Máquinas de Vectores de Soporte (SVM)
SVM es un método supervisado que busca encontrar el hiperplano que mejor separa las clases en el espacio de características, maximizando el margen entre ellas. En este proyecto, se emplea una SVM con kernel lineal adecuada para problemas donde las clases son aproximadamente separables linealmente.

---

## Estructura del proyecto

- **`BASE_PATH`**: carpeta donde están los datos y donde se guardan las gráficas y el CSV de resultados.
- **`DATA_PATH`**: ruta del archivo de datos `wdbc.data`.
- **`OUTPUT_PATH`**: carpeta para las gráficas generadas.
- **`RESULTS_FILE`**: archivo CSV donde se almacenan los resultados.

---

## Flujo de trabajo

### 1. Carga de datos

- Se lee el archivo `wdbc.data`, se asignan nombres a las columnas, y se transforma la variable objetivo `Diagnostico` en valores binarios (1: Maligno, 0: Benigno).

### 2. Preprocesamiento

- Se eliminan columnas irrelevantes (ID).
- Se normalizan las características con `StandardScaler` para mejorar el rendimiento de k-NN y SVM.

### 3. Entrenamiento y evaluación de modelos

Para cada porcentaje de entrenamiento entre 10% y 90%:

- Se divide el conjunto de datos en entrenamiento y prueba.
- Se entrenan los siguientes modelos:
  - **Regresión Logística**
  - **k-NN**: se buscan los mejores valores de `k` de 1 a 20, seleccionando el que maximiza la precisión en prueba.
  - **SVM lineal**
- Se calcula la precisión en entrenamiento y en prueba para cada modelo.
- Se almacena el mejor valor de `k` en k-NN para cada porcentaje.

La ejecución se monitoriza con una barra de progreso (`tqdm`).

### 4. Exportación de resultados

Se genera un archivo `resultados.csv` que contiene:

| Tamaño Entrenamiento | Precisión Entrenamiento Logística | Precisión Prueba Logística | Precisión Entrenamiento k-NN | Precisión Prueba k-NN | Mejor k | Precisión Entrenamiento SVM | Precisión Prueba SVM |
| :------------------: | :------------------------------: | :------------------------: | :--------------------------: | :--------------------: | :-----: | :--------------------------: | :------------------: |

### 5. Visualización de resultados

Se generan varias gráficas, guardadas en la carpeta de salida:

- **Precisión en conjunto de prueba** para los tres modelos.
- **Precisión en conjunto de entrenamiento** para los tres modelos.
- **Comparativa de entrenamiento vs prueba** para cada modelo.
- **Evolución del mejor valor de `k`** en función del tamaño del conjunto de entrenamiento.

---

## Requisitos

- Python 3.x
- Bibliotecas:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `tqdm`

Instalación de dependencias:
```bash
pip install numpy pandas scikit-learn matplotlib tqdm
```

---

## Estructura de carpetas esperada

```
/curso IA
    |-- wdbc.data
    |-- script.py (código principal)
    |-- graficas/ (se genera automáticamente)
    |-- resultados.csv
```

---

## Comentarios finales

- **Normalización** de características es crítica para asegurar que los modelos trabajen correctamente.
- El **valor óptimo de `k`** en k-NN puede cambiar según el tamaño del conjunto de entrenamiento.
- Comparar la precisión entre entrenamiento y prueba permite detectar **overfitting** o **underfitting**.

Este proyecto proporciona una base sólida para experimentar con otros modelos (como Random Forests o Redes Neuronales) y análisis de hiperparámetros.

Basado en fundamentos teóricos del curso que hice  "Introducción a la Inteligencia Artificial: Aprendizaje Automático y Redes Neuronales" (USAL, 2021).

---

> **Autor:** [Marco Ezquerra Ruano]  
> **Fecha:** Abril 2025
