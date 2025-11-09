import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Ubicación de nuestros modelos
BASE_DIR = os.path.dirname(__file__)
MODEL_REGISTRY_PATH = os.path.join(BASE_DIR, 'model_registry')
TIME_STEPS = 6  # ¡Debe ser el mismo que usaste para entrenar!

model_cache = {}

def get_model_for_cuarto(cuarto_id):
    """
    Carga (o obtiene de caché) el modelo, scaler y umbral para un cuarto_id.
    """
    cuarto_key = str(cuarto_id)
    
    if cuarto_key in model_cache:
        return model_cache[cuarto_key]

    # El modelo no está en caché, hay que cargarlo
    cuarto_path = os.path.join(MODEL_REGISTRY_PATH, f'cuarto_{cuarto_key}')
    model_path = os.path.join(cuarto_path, 'modelo_autoencoder_frio_REAL.h5')
    scaler_path = os.path.join(cuarto_path, 'scaler_frio_REAL.g')
    umbral_path = os.path.join(cuarto_path, 'umbral.txt')

    if not all([os.path.exists(p) for p in [model_path, scaler_path, umbral_path]]):
        # Si no tenemos un modelo para este cuarto, no podemos predecir
        return None

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        with open(umbral_path, 'r') as f:
            umbral = float(f.read().strip())
        
        # Guardar en caché para la próxima vez
        model_cache[cuarto_key] = (model, scaler, umbral)
        return model_cache[cuarto_key]
        
    except Exception as e:
        print(f"Error cargando modelo para cuarto {cuarto_key}: {e}")
        return None

def crear_secuencias(datos, time_steps=TIME_STEPS):
    # Función de ayuda para crear la secuencia
    # (Asume que 'datos' ya está escalado y es un array 2D de (6, 2))
    if len(datos) != time_steps:
        return None
    return np.array([datos]) # Forma (1, 6, 2)

@app.route('/detectar', methods=['POST'])
def detectar_anomalia():
    data = request.get_json()

    if not data or 'cuarto_id' not in data or 'lecturas' not in data:
        return jsonify({'error': 'JSON inválido. Se requiere "cuarto_id" y "lecturas".'}), 400

    cuarto_id = data['cuarto_id']
    lecturas = data['lecturas'] # Lista de 6 lecturas

    if len(lecturas) != TIME_STEPS:
        return jsonify({'error': f'Se esperan {TIME_STEPS} lecturas, pero se recibieron {len(lecturas)}.'}), 400

    # 1. Cargar el modelo, scaler y umbral correctos
    model_assets = get_model_for_cuarto(cuarto_id)
    if model_assets is None:
        return jsonify({'error': f'No se encontró un modelo de IA entrenado para el cuarto_id {cuarto_id}.'}), 404
    
    model, scaler, UMBRAL = model_assets

    try:
        # 2. Preparar los datos (convertir [{'temp': 5, 'hum': 80}, ...] a [[5, 80], ...])
        # ¡IMPORTANTE! El orden debe ser el mismo del entrenamiento (ej. temp, hum)
        lecturas_np = np.array([[float(l['temperatura_c']), float(l['humedad_pct'])] for l in lecturas])
        
        # 3. Escalar los datos
        datos_scaled = scaler.transform(lecturas_np)
        
        # 4. Crear la secuencia para la predicción
        secuencia = crear_secuencias(datos_scaled, TIME_STEPS)
        if secuencia is None:
             return jsonify({'error': 'Falló la creación de la secuencia.'}), 500

        # 5. Predecir y calcular error
        pred = model.predict(secuencia)
        loss = np.mean(np.abs(pred - secuencia))

        # 6. Decidir si es anomalía
        es_anomalia = bool(loss > UMBRAL) # bool() para que sea JSON serializable

        return jsonify({
            'anomalia': es_anomalia,
            'cuarto_id': cuarto_id,
            'error_reconstruccion': loss,
            'umbral': UMBRAL
        })

    except Exception as e:
        return jsonify({'error': f'Error durante la predicción: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    # Un endpoint simple para saber que el servicio está vivo
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    # '0.0.0.0' es necesario para que sea accesible fuera de localhost
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))