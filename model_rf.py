import os
from functools import lru_cache
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import load_dataset, _build_feature_vector


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


def predict_rf(clima, hora, zona, model=None):
    clima = float(clima)
    hora = float(hora)
    zona = float(zona)
    # Build feature vector consistent with linear model
    from model import get_feature_means
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

    if model is None:
        model = train_random_forest()

    pred = model.predict(feature_vector.reshape(1, -1))[0]
    return float(round(float(pred), 2))


def cross_validate_rf(k=5, n_estimators=100, random_state=42, max_depth=None):
    rows = load_dataset()
    X, y = _build_matrix(rows)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    maes = []
    rmses = []
    r2s = []

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        folds.append({'fold': i + 1, 'train_size': len(train_idx), 'test_size': len(test_idx), 'mae': round(float(mae), 2), 'rmse': round(float(rmse), 2), 'r2': round(float(r2), 2)})
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

    resumen = {
        'mae_mean': round(float(np.mean(maes)), 2),
        'mae_std': round(float(np.std(maes, ddof=0)), 2),
        'rmse_mean': round(float(np.mean(rmses)), 2),
        'r2_mean': round(float(np.mean(r2s)), 2),
        'r2_std': round(float(np.std(r2s, ddof=0)), 2),
        'r2_pct': int(round(float(np.mean(r2s)) * 100)),
    }
    return {'folds': folds, 'resumen': resumen}
