# Prototipo de IA para Detección de Anomalías en Cuartos Fríos

## Descripción del Proyecto
Este repositorio documenta el desarrollo de un **modelo de Inteligencia Artificial (IA)** diseñado para el **Sistema de Monitoreo de Temperaturas en Cuartos Fríos de Grupo Detpon**.

El objetivo principal es analizar en tiempo real los **patrones de temperatura y humedad** y detectar **anomalías** que puedan indicar fallos en equipos de refrigeración o errores humanos (por ejemplo, una puerta abierta), contribuyendo a la **reducción de pérdidas de producto**.

El modelo implementado es un **Autoencoder basado en Redes Neuronales Recurrentes (LSTM)** desarrollado y entrenado en **Python** mediante **Jupyter Notebook**.

---

## Parte 1: Configuración del Entorno de Desarrollo

### 1. Estructura del Proyecto
Crea una carpeta raíz para el proyecto, por ejemplo:


### 2. Crear y Activar Entorno Virtual

```bash
# Crear entorno virtual
python -m venv ia_env

# Activar entorno (Windows)
.\ia_env\Scripts\activate

# Activar entorno (Linux/Mac)
source ia_env/bin/activate

Instalación de Librerías Necesarias
pip install jupyter
pip install requests
pip install pandas
pip install tensorflow
pip install scikit-learn
pip install matplotlib
pip install joblib

Iniciar el Entorno de Jupyter
jupyter notebook

Abrir el archivo:

modelo_autoencoder_frio.ipynb

Parte 2: Desarrollo del Modelo

Cada celda del notebook realiza una función específica dentro del proceso de entrenamiento y validación del modelo.

Celda	Descripción	Función Principal
1	Importación de bibliotecas	Carga las librerías necesarias (pandas, tensorflow, sklearn, etc.)
2	Carga de datos desde API	Obtiene las lecturas de temperatura/humedad desde https://drover.detpon.com/api/lecturas
2.5	Limpieza y filtrado	Prepara los datos: selecciona un solo cuarto frío, filtra lecturas normales y columnas relevantes
3	Preprocesamiento	Normaliza los datos (MinMaxScaler) y crea secuencias temporales (TIME_STEPS = 6)
4	Arquitectura del modelo	Construye un Autoencoder LSTM con capas Encoder, Decoder y TimeDistributed
5	Entrenamiento	Entrena el modelo para reconstruir los patrones normales de comportamiento térmico
6	Umbral de anomalía	Calcula el error de reconstrucción y define el límite superior (percentil 95)
7	Validación	Evalúa el modelo con datos anómalos y grafica el error frente al umbral
8	Guardado de artefactos	Guarda el modelo y el scaler entrenados para su implementación posterior



Parte 4: Resultados del Entrenamiento

El modelo aprende a reconstruir comportamientos normales de temperatura y humedad.

Un umbral estadístico (percentil 95) separa los valores normales de los anómalos.

Las gráficas de validación muestran cómo los errores de reconstrucción aumentan durante una anomalía (puerta abierta, fallo de compresor, etc.).

Parte 5: Artefactos Finales

El entrenamiento genera tres elementos clave:

Archivo	Descripción
modelo_autoencoder_frio_REAL.h5	Contiene la arquitectura y los pesos del modelo entrenado (el "cerebro" de la IA).
scaler_frio_REAL.gz	Contiene el objeto MinMaxScaler ajustado para normalizar futuras lecturas.
umbral.json (opcional)	Guarda el valor numérico del umbral de anomalía calculado.

Parte 6: Créditos y Autoría

📍 Proyecto: Sistema de Monitoreo de Temperaturas en Cuartos Fríos
👨‍💻 Autor: Equipo de IA – Grupo Detpon
🏗️ Desarrollado con: Python, TensorFlow, Keras, Pandas, Scikit-learn
📅 Versión: 1.0 (Prototipo validado en entorno Jupyter)
