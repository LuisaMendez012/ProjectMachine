from flask import Flask, render_template, request, redirect
from model import predecir_ocupacion, crear_datos_evaluacion, calcular_validacion_cruzada
from model_rl import train_q_learning, get_q_table
from model_rf import train_random_forest, predict_rf, cross_validate_rf, plot_feature_importances, plot_prediction_vs_actual, plot_model_comparison, plot_metrics_summary
from model_kmeans import run_kmeans, plot_cluster_counts
import os
import math
import statistics

app = Flask(__name__)

# Compute regression metrics for a dataset of actual and predicted values
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
        'reason': 'Linear Regression is the reference model used for baseline comparison.',
        'model_key': 'linear',
        'metrics': cv_linear.get('resumen', {}),
    }
    if isinstance(rf_cv, dict) and 'error' not in rf_cv:
        if rf_cv['resumen']['r2_mean'] >= cv_linear['resumen']['r2_mean'] and rf_cv['resumen']['rmse_mean'] <= cv_linear['resumen']['rmse_mean']:
            best = {
                'name': 'Random Forest',
                'reason': 'Random Forest achieves stronger cross-validated metrics and is selected as the preferred production model.',
                'model_key': 'rf',
                'metrics': rf_cv.get('resumen', {}),
            }
    return best


# -------- PHASE 1: BUSINESS UNDERSTANDING --------
@app.route('/')
def fase1():
    return render_template('fase1.html', active_page='home')

# -------- PHASE 2: MODEL BUILDING --------
@app.route('/modelo', methods=['GET', 'POST'])
def fase2():
    resultado = None
 
    if request.method == 'POST':
        clima = float(request.form['clima'])
        hora  = float(request.form['hora'])
        zona  = float(request.form['zona'])
        resultado = predecir_ocupacion(clima, hora, zona)
 
    return render_template('fase2.html', resultado=resultado, active_page='model')
 
# -------- PHASE 3: EVALUATION --------
@app.route('/evaluacion')
@app.route('/evaluation')
@app.route('/fase3')
def fase3():
    base_samples = crear_datos_evaluacion()
    for item in base_samples:
        item['pred'] = predecir_ocupacion(item['clima_val'], item['hora_val'], item['zona_val'])
        item['error'] = round(abs(item['real'] - item['pred']), 2)

    m = calcular_metricas(base_samples)
    m['samples'] = [
        {
            'clima': item['clima'],
            'hora': item['hora'],
            'zona': item['zona'],
            'real': item['real'],
            'pred': item['pred'],
            'error': item['error'],
        }
        for item in base_samples
    ]

    cv = calcular_validacion_cruzada(k=5)

    # Random Forest cross-validation (kept separate from linear CV)
    try:
        rf_cv = cross_validate_rf(k=5)
    except Exception as e:
        rf_cv = {'error': str(e)}

    page_errors = []

    # K-Means clustering analysis and visualization
    try:
        kmeans_data = run_kmeans(n_clusters=4)
        kmeans_plot = plot_cluster_counts(kmeans_data['counts'])
    except Exception as e:
        kmeans_data = {'error': str(e)}
        kmeans_plot = None
        page_errors.append('K-Means analysis failed: ' + str(e))

    try:
        prediction_plot = plot_prediction_vs_actual([item['real'] for item in base_samples], [item['pred'] for item in base_samples])
    except Exception as e:
        prediction_plot = None
        page_errors.append('Prediction plot generation failed: ' + str(e))

    try:
        metrics_plot = plot_metrics_summary(m)
    except Exception as e:
        metrics_plot = None
        page_errors.append('Metrics summary chart failed: ' + str(e))

    try:
        comparison_plot = plot_model_comparison(cv, rf_cv)
    except Exception as e:
        comparison_plot = None
        page_errors.append('Model comparison chart failed: ' + str(e))

    riesgos = [
        {
            'nivel': 'Alto',
            'nombre': 'Limited data volume',
            'descripcion': 'The model is evaluated on a small sample of real cases.',
            'indicador': 'Few observations for generalization',
            'recomendacion': 'Collect more historical data before using the model for operational decisions.',
        },
        {
            'nivel': 'Medio',
            'nombre': 'Weather sensitivity',
            'descripcion': 'Predictions vary depending on weather conditions and location.',
            'indicador': 'Moderate MAPE and fold-to-fold variation',
            'recomendacion': 'Validate with additional weather and traffic conditions.',
        },
        {
            'nivel': 'Bajo',
            'nombre': 'Climatic period variation',
            'descripcion': 'Occupancy behavior may change across different climatic periods (e.g., dry vs. rainy).',
            'indicador': 'Future shifts in usage patterns',
            'recomendacion': 'Retrain the model with recent data periodically.',
        },
    ]

    reales = [item['real'] for item in base_samples]
    promedio_real = sum(reales) / len(reales)
    baseline_metrics = calcular_metricas([
        {'real': real, 'pred': promedio_real} for real in reales
    ])
    mb = baseline_metrics

    best_model = seleccionar_mejor_model(cv, rf_cv)

    return render_template(
        'fase3.html',
        m=m,
        cv=cv,
        rf_cv=rf_cv,
        kmeans=kmeans_data,
        kmeans_plot=kmeans_plot,
        prediction_plot=prediction_plot,
        metrics_plot=metrics_plot,
        comparison_plot=comparison_plot,
        riesgos=riesgos,
        mb=mb,
        best_model=best_model,
        page_errors=page_errors,
        active_page='evaluation'
    )


def validar_entrada_prediccion(clima, hora, zona):
    errores = []
    valores = {'clima': clima, 'hora': hora, 'zona': zona}

    if clima.strip() == '' or hora.strip() == '' or zona.strip() == '':
        errores.append('Todos los campos son obligatorios.')
        return None, None, None, valores, errores

    try:
        clima_val = float(clima)
        hora_val = float(hora)
        zona_val = float(zona)
    except ValueError:
        errores.append('Ingrese valores numéricos válidos para clima, hora y periodo climático.')
        return None, None, None, valores, errores

    if not clima_val.is_integer() or clima_val < 1 or clima_val > 4:
        errores.append('La categoría de clima debe ser un número entero entre 1 y 4.')
    if not zona_val.is_integer() or zona_val < 1 or zona_val > 4:
        errores.append('El periodo climático debe ser un número entero entre 1 y 4.')
    if not hora_val.is_integer() or hora_val < 0 or hora_val > 23:
        errores.append('La hora debe ser un número entero entre 0 y 23.')

    if errores:
        return None, None, None, valores, errores

    return int(clima_val), int(hora_val), int(zona_val), valores, errores


@app.route('/prediction', methods=['GET', 'POST'])
@app.route('/prediction-system', methods=['GET', 'POST'])
def prediction():
    error = None
    resultado = None
    reliability = None
    input_values = {'clima': '', 'hora': '', 'zona': ''}

    cv = calcular_validacion_cruzada(k=5)
    try:
        rf_cv = cross_validate_rf(k=5)
    except Exception as e:
        rf_cv = {'error': str(e)}

    best_model = seleccionar_mejor_model(cv, rf_cv)

    if request.method == 'POST':
        clima = request.form.get('clima', '')
        hora = request.form.get('hora', '')
        zona = request.form.get('zona', '')
        clima_val, hora_val, zona_val, input_values, errores = validar_entrada_prediccion(clima, hora, zona)

        if errores:
            error = ' '.join(errores)
        else:
            try:
                if best_model['model_key'] == 'rf' and 'error' not in rf_cv:
                    # Use the cached Random Forest model for inference rather than retraining on every request.
                    model = train_random_forest()
                    resultado = predict_rf(clima_val, hora_val, zona_val, model=model)
                else:
                    resultado = predecir_ocupacion(clima_val, hora_val, zona_val)

                reliability = {
                    'mae': best_model['metrics'].get('mae_mean', 'N/A'),
                    'mse': best_model['metrics'].get('mse_mean', 'N/A'),
                    'rmse': best_model['metrics'].get('rmse_mean', 'N/A'),
                    'mape': best_model['metrics'].get('mape_mean', 'N/A'),
                    'r2': best_model['metrics'].get('r2_mean', 'N/A'),
                }
            except Exception as e:
                error = 'Prediction failed: ' + str(e)

    return render_template(
        'prediction.html',
        resultado=resultado,
        error=error,
        reliability=reliability,
        input_values=input_values,
        best_model=best_model,
        active_page='prediction'
    )


@app.route('/rf/train')
def rf_train():
    # Trigger training (cached by lru_cache in model_rf)
    n_estimators = int(request.args.get('n_estimators', 100))
    max_depth = request.args.get('max_depth')
    max_depth = int(max_depth) if max_depth is not None and max_depth != '' else None
    train_random_forest(n_estimators=n_estimators, max_depth=max_depth)
    # create feature importance plot
    try:
        plot_feature_importances()
    except Exception:
        pass
    return "Random Forest training completed. <a href='/evaluacion'>Back to evaluation</a>"


@app.route('/modelo/rf', methods=['GET', 'POST'])
def modelo_rf():
    resultado = None
    if request.method == 'POST':
        clima = float(request.form['clima'])
        hora = float(request.form['hora'])
        zona = float(request.form['zona'])
        # allow optional hyperparams for a quick train+predict
        n_estimators = int(request.form.get('n_estimators', 100))
        max_depth = request.form.get('max_depth')
        max_depth = int(max_depth) if max_depth not in (None, '') else None
        # force a model with provided params
        model = train_random_forest(n_estimators=n_estimators, max_depth=max_depth)
        resultado = predict_rf(clima, hora, zona, model=model)
    return render_template('fase2.html', resultado_rf=resultado, active_page='model')


@app.route('/kmeans/run')
def kmeans_run():
    k = int(request.args.get('k', 4))
    try:
        run_kmeans(n_clusters=k)
    except Exception:
        pass
    return redirect('/evaluacion')

@app.route('/entendimiento')
@app.route('/home')
def fase1_alias():
    return render_template('fase1.html', active_page='home')
 
# -------- REINFORCEMENT LEARNING --------
@app.route('/rl')
def rl_menu():
    return render_template('rl.html', active_page='rl')
 
@app.route('/rl/concepts')
def rl_concepts():
    return render_template('rl_concepts.html', active_page='rl')
 
@app.route('/rl/agent')
def rl_agent():
    return render_template('rl_agent.html', active_page='rl')
 
@app.route('/rl/train')
def rl_train():
    train_q_learning()
    return "Training completed. <a href='/rl/results'>See results</a>"
 
@app.route('/rl/results')
def rl_results():
    q = get_q_table()
    return render_template('rl_results.html', q=q, active_page='rl')
 
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
 