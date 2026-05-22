import csv
import os
from functools import lru_cache

import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'hour.csv')
FEATURE_NAMES = ['hr', 'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed']
WEATHER_LABELS = {
    '1': 'Clear',
    '2': 'Mist',
    '3': 'Light rain',
    '4': 'Heavy rain/snow',
}
SEASON_LABELS = {
    '1': 'Dry period',
    '2': 'Moderate rainfall period',
    '3': 'High rainfall period',
    '4': 'Variable weather period',
}


def load_dataset():
    with open(DATA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


@lru_cache(maxsize=1)
def get_feature_means():
    rows = load_dataset()
    values = {name: [] for name in FEATURE_NAMES}
    for row in rows:
        for name in FEATURE_NAMES:
            values[name].append(float(row[name]))
    return {name: float(np.mean(values[name])) for name in FEATURE_NAMES}


def _build_feature_vector(row):
    return np.array([
        float(row['hr']),
        float(row['season']),
        float(row['yr']),
        float(row['mnth']),
        float(row['holiday']),
        float(row['weekday']),
        float(row['workingday']),
        float(row['weathersit']),
        float(row['temp']),
        float(row['hum']),
        float(row['windspeed']),
    ], dtype=float)


def train_linear_model(rows=None):
    if rows is None:
        rows = load_dataset()
    X = np.vstack([_build_feature_vector(row) for row in rows])
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    y = np.array([float(row['cnt']) for row in rows], dtype=float)
    weights = np.linalg.pinv(X).dot(y)
    return weights


@lru_cache(maxsize=1)
def get_trained_weights():
    return train_linear_model()


def predict_with_weights(feature_vector, weights):
    inputs = np.hstack(([1.0], feature_vector))
    return float(np.dot(weights, inputs))


def predecir_ocupacion(clima, hora, zona):
    clima = float(clima)
    hora = float(hora)
    zona = float(zona)

    means = get_feature_means()
    feature_vector = np.array([
        hora,
        zona,
        means['yr'],
        means['mnth'],
        means['holiday'],
        means['weekday'],
        means['workingday'],
        clima,
        means['temp'],
        means['hum'],
        means['windspeed'],
    ], dtype=float)

    weights = get_trained_weights()
    prediction = predict_with_weights(feature_vector, weights)
    return round(prediction, 2)


def crear_datos_evaluacion(limit=5):
    rows = load_dataset()
    samples = []
    for row in rows[:limit]:
        samples.append({
            'clima_val': float(row['weathersit']),
            'hora_val': float(row['hr']),
            'zona_val': float(row['season']),
            'clima': WEATHER_LABELS.get(row['weathersit'], row['weathersit']),
            'hora': f"{int(row['hr']):02d}:00",
            'zona': SEASON_LABELS.get(row['season'], f"Season {row['season']}"),
            'real': float(row['cnt']),
        })
    return samples


@lru_cache(maxsize=4)
def calcular_validacion_cruzada(k=5):
    rows = load_dataset()
    total = len(rows)
    indices = np.arange(total)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    fold_size = total // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else total
        test_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

        train_rows = [rows[idx] for idx in train_indices]
        test_rows = [rows[idx] for idx in test_indices]

        weights = train_linear_model(train_rows)
        predictions = []
        for row in test_rows:
            vector = _build_feature_vector(row)
            pred = predict_with_weights(vector, weights)
            predictions.append({'real': float(row['cnt']), 'pred': float(pred)})

        reales = [item['real'] for item in predictions]
        preds = [item['pred'] for item in predictions]
        mae = float(np.mean(np.abs(np.array(reales) - np.array(preds))))
        rmse = float(np.sqrt(np.mean((np.array(reales) - np.array(preds)) ** 2)))
        promedio = float(np.mean(reales))
        ss_res = float(np.sum((np.array(reales) - np.array(preds)) ** 2))
        ss_tot = float(np.sum((np.array(reales) - promedio) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

        folds.append({
            'fold': i + 1,
            'train_size': len(train_rows),
            'test_size': len(test_rows),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 2),
        })

    resumen = {
        'mae_mean': round(float(np.mean([f['mae'] for f in folds])), 2),
        'mae_std': round(float(np.std([f['mae'] for f in folds], ddof=0)), 2),
        'rmse_mean': round(float(np.mean([f['rmse'] for f in folds])), 2),
        'r2_mean': round(float(np.mean([f['r2'] for f in folds])), 2),
        'r2_std': round(float(np.std([f['r2'] for f in folds], ddof=0)), 2),
        'r2_pct': int(round(float(np.mean([f['r2'] for f in folds])) * 100)),
    }
    return {'folds': folds, 'resumen': resumen}
