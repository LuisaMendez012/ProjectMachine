import os
from functools import lru_cache
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import load_dataset, _build_feature_vector
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import FEATURE_NAMES
from model import get_feature_means
import requests
from datetime import datetime




def _build_matrix(rows):
    X = np.vstack([_build_feature_vector(row) for row in rows])
    y = np.array([float(row['cnt']) for row in rows], dtype=float)
    return X, y


@lru_cache(maxsize=1)
def train_random_forest(n_estimators=100, random_state=42, max_depth=None):
    rows = load_dataset()
    X, y = _build_matrix(rows)
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
    rf.fit(X, y)
    return rf

print("🔥 Entrenando modelo una sola vez...")
modelo_global = train_random_forest(n_estimators=20)
print("✅ Modelo listo")


def predict_rf(clima, hora, zona):
    global modelo_global

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

    pred = modelo_global.predict(feature_vector.reshape(1, -1))[0]
    return float(round(float(pred), 2))



def predict_auto(lugar, zona):
    from datetime import datetime
    import requests

    hora = datetime.now().hour

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={lugar}&appid=9fff9f0d02f0876e9b331bb4433f2d07&units=metric"
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            print("API error:", response.text)
            clima = 1
        else:
            data = response.json()

            if "weather" not in data:
                clima = 1
            else:
                weather_main = data["weather"][0]["main"].lower()

                if "rain" in weather_main:
                    clima = 3
                elif "cloud" in weather_main:
                    clima = 2
                else:
                    clima = 1

    except Exception as e:
        print("Error clima:", e)
        clima = 1

    return predict_rf(clima, hora, zona)


def cross_validate_rf(k=5, n_estimators=100, random_state=42, max_depth=None):
    rows = load_dataset()
    X, y = _build_matrix(rows)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    maes = []
    mses = []
    rmses = []
    r2s = []
    mapes = []

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(np.mean((y_test - preds) ** 2))
        mape = np.mean(np.abs((y_test - preds) / np.where(y_test != 0, y_test, 1))) * 100
        r2 = r2_score(y_test, preds)
        folds.append({'fold': i + 1, 'train_size': len(train_idx), 'test_size': len(test_idx), 'mae': round(float(mae), 2), 'mse': round(float(np.mean((y_test - preds) ** 2)), 2), 'rmse': round(float(rmse), 2), 'r2': round(float(r2), 2), 'mape': round(float(mape), 2)})
        maes.append(mae)
        mses.append(float(np.mean((y_test - preds) ** 2)) )
        rmses.append(rmse)
        r2s.append(r2)
        mapes.append(mape)

    resumen = {
        'mae_mean': round(float(np.mean(maes)), 2),
        'mse_mean': round(float(np.mean(mses)), 2),
        'rmse_mean': round(float(np.mean(rmses)), 2),
        'r2_mean': round(float(np.mean(r2s)), 2),
        'mape_mean': round(float(np.mean(mapes)), 2),
        'r2_std': round(float(np.std(r2s, ddof=0)), 2),
        'r2_pct': int(round(float(np.mean(r2s)) * 100)),
    }
    return {'folds': folds, 'resumen': resumen}


def plot_feature_importances(model=None, save_path=None):
    if model is None:
        model = train_random_forest()
    importances = model.feature_importances_
    names = FEATURE_NAMES
    indices = np.argsort(importances)[::-1]

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'static', 'images', 'rf_importances.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.title('Random Forest feature importances')
    plt.bar([names[i] for i in indices], importances[indices])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return os.path.join('static', 'images', 'rf_importances.png')


def plot_prediction_vs_actual(actuals, preds, save_path=None):
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'static', 'images', 'prediction_vs_actual.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 4))
    indices = np.arange(len(actuals)) + 1
    plt.plot(indices, actuals, label='Actual', marker='o')
    plt.plot(indices, preds, label='Predicted', marker='o')
    plt.title('Prediction vs Actual Occupancy')
    plt.xlabel('Sample index')
    plt.ylabel('Occupancy count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return os.path.join('static', 'images', 'prediction_vs_actual.png')


def plot_model_comparison(cv_linear, cv_rf, save_path=None):
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'static', 'images', 'model_comparison.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = ['MAE', 'RMSE', 'R²']
    linear_values = [cv_linear['resumen']['mae_mean'], cv_linear['resumen']['rmse_mean'], cv_linear['resumen']['r2_mean']]
    rf_values = [cv_rf['resumen']['mae_mean'], cv_rf['resumen']['rmse_mean'], cv_rf['resumen']['r2_mean']]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - width/2, linear_values, width, label='Linear')
    plt.bar(x + width/2, rf_values, width, label='Random Forest')
    plt.xticks(x, labels)
    plt.title('Model Comparison: Cross-validated Metrics')
    plt.ylabel('Metric value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return os.path.join('static', 'images', 'model_comparison.png')


def plot_metrics_summary(m, save_path=None):
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'static', 'images', 'metrics_summary.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = ['MAE', 'MSE', 'RMSE', 'R²']
    values = [m['mae'], m['mse'], m['rmse'], m['r2']]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values, color=['#38bdf8', '#0ea5e9', '#22c55e', '#f59e0b'])
    plt.title('Evaluation Metrics Summary')
    plt.ylabel('Value')
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(value), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return os.path.join('static', 'images', 'metrics_summary.png')
