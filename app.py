from flask import Flask, render_template, request, redirect
from model import predecir_ocupacion, crear_datos_evaluacion, calcular_validacion_cruzada
from model_rl import train_q_learning, get_q_table
from model_rf import (
    train_random_forest,
    predict_rf,
    cross_validate_rf,
    plot_feature_importances,
    plot_prediction_vs_actual,
    plot_model_comparison,
    plot_metrics_summary,
    predict_auto,
)
from model_kmeans import run_kmeans, plot_cluster_counts
import os
import math
import statistics

app = Flask(__name__)

# ==============================
# METRICS
# ==============================
def calcular_metricas(dataset):
    reales = [item['real'] for item in dataset]
    predichos = [item['pred'] for item in dataset]
    n = len(dataset)
    mae = sum(abs(r - p) for r, p in zip(reales, predichos)) / n
    mse = sum((r - p) ** 2 for r, p in zip(reales, predichos)) / n
    rmse = math.sqrt(mse)
    mape = sum(abs((r - p) / r) for r, p in zip(reales, predichos) if r != 0) / n * 100
    promedio = sum(reales) / n
    ss_res = sum((r - p) ** 2 for r, p in zip(reales, predichos))
    ss_tot = sum((r - promedio) ** 2 for r in reales)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return {
        'mae': round(mae, 2),
        'mse': round(mse, 2),
        'rmse': round(rmse, 2),
        'r2': round(r2, 2),
        'mape': round(mape, 2),
        'r2_pct': int(round(r2 * 100)),
    }


def seleccionar_mejor_model(cv_linear, rf_cv):
    best = {
        'name': 'Linear Regression',
        'model_key': 'linear',
        'metrics': cv_linear.get('resumen', {}),
    }
    if isinstance(rf_cv, dict) and 'error' not in rf_cv:
        if rf_cv['resumen']['r2_mean'] >= cv_linear['resumen']['r2_mean']:
            best = {
                'name': 'Random Forest',
                'model_key': 'rf',
                'metrics': rf_cv.get('resumen', {}),
            }
    return best


# ==============================
# HOME
# ==============================
@app.route('/')
def fase1():
    return render_template('fase1.html', active_page='home')


# ==============================
# ABOUT
# ==============================
@app.route('/about')
def about():
    return render_template('about.html', active_page='about')


# ==============================
# PHASE 2 (UPDATED 🔥)
# ==============================
@app.route('/modelo', methods=['GET', 'POST'])
def fase2():
    resultado = None

    if request.method == 'POST':
        try:
            lugar = request.form.get('lugar')
            zona = request.form.get('zona')

            print('DEBUG FASE2:', lugar, zona)

            if lugar and zona:
                resultado = predict_auto(lugar, float(zona))
        except Exception as e:
            print('ERROR FASE2:', str(e))

    return render_template('fase2.html', resultado=resultado, active_page='model')


# ==============================
# FASE 3
# ==============================
@app.route('/evaluacion')
@app.route('/evaluation')
@app.route('/fase3')
def fase3():
    base_samples = crear_datos_evaluacion()

    for item in base_samples:
        item['pred'] = predecir_ocupacion(item['clima_val'], item['hora_val'], item['zona_val'])

    cv = calcular_validacion_cruzada(k=5)

    try:
        rf_cv = cross_validate_rf(k=5)
    except Exception as e:
        rf_cv = {'error': str(e)}

    best_model = seleccionar_mejor_model(cv, rf_cv)

    return render_template(
        'fase3.html',
        cv=cv,
        rf_cv=rf_cv,
        best_model=best_model,
        active_page='evaluation',
    )


# ==============================
# 🔥 PREDICTION (FIXED)
# ==============================
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    error = None
    resultado = None

    if request.method == 'POST':
        try:
            lugar = request.form.get('lugar')
            zona = request.form.get('zona')

            print('DEBUG INPUT:', lugar, zona)

            if not lugar or not zona:
                error = 'Missing data'
            else:
                resultado = predict_auto(lugar, float(zona))
                print('RESULT:', resultado)
        except Exception as e:
            error = str(e)
            print('ERROR:', error)

    return render_template('prediction.html', resultado=resultado, error=error)


# ==============================
# RF TRAIN
# ==============================
@app.route('/rf/train')
def rf_train():
    train_random_forest()
    return 'RF ready'


# ==============================
# RL
# ==============================
@app.route('/rl')
def rl_menu():
    return render_template('rl.html')


@app.route('/rl/train')
def rl_train():
    train_q_learning()
    return 'Trained'


@app.route('/rl/results')
def rl_results():
    q = get_q_table()
    return render_template('rl_results.html', q=q)


# ==============================
# RUN
# ==============================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

