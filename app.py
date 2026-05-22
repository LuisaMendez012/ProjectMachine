from flask import Flask, render_template, request
from model import predecir_ocupacion
from model_rl import train_q_learning, get_q_table
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


# Build a small sample dataset for evaluation and example display
def crear_datos_evaluacion():
    return [
        {'clima_val': 1.0, 'hora_val': 8.0, 'zona_val': 1.0, 'clima': 'Sunny', 'hora': '08:00', 'zona': 'A', 'real': 3.2},
        {'clima_val': 0.7, 'hora_val': 12.0, 'zona_val': 2.0, 'clima': 'Nublado', 'hora': '12:00', 'zona': 'B', 'real': 4.2},
        {'clima_val': 0.3, 'hora_val': 17.0, 'zona_val': 3.0, 'clima': 'Lluvia', 'hora': '17:00', 'zona': 'C', 'real': 5.6},
        {'clima_val': 0.7, 'hora_val': 20.0, 'zona_val': 1.0, 'clima': 'Nublado', 'hora': '20:00', 'zona': 'A', 'real': 4.4},
        {'clima_val': 1.0, 'hora_val': 22.0, 'zona_val': 2.0, 'clima': 'Soleado', 'hora': '22:00', 'zona': 'B', 'real': 5.1},
    ]


# Perform simple K-fold cross-validation using the sample dataset
def calcular_validacion_cruzada(dataset, k=5):
    fold_size = max(1, len(dataset) // k)
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size
        test = dataset[start:end] if i < k - 1 else dataset[start:]
        train = [x for j, x in enumerate(dataset) if j < start or j >= end]
        if not test:
            continue
        fold_preds = []
        for item in test:
            pred = predecir_ocupacion(item['clima_val'], item['hora_val'], item['zona_val'])
            fold_preds.append({'real': item['real'], 'pred': pred})
        metrics = calcular_metricas(fold_preds)
        folds.append({
            'fold': i + 1,
            'train_size': len(train),
            'test_size': len(test),
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'r2': metrics['r2'],
        })
    resumen = {
        'mae_mean': round(statistics.mean([f['mae'] for f in folds]), 2),
        'mae_std': round(statistics.pstdev([f['mae'] for f in folds]), 2),
        'rmse_mean': round(statistics.mean([f['rmse'] for f in folds]), 2),
        'r2_mean': round(statistics.mean([f['r2'] for f in folds]), 2),
        'r2_std': round(statistics.pstdev([f['r2'] for f in folds]), 2),
        'r2_pct': int(round(statistics.mean([f['r2'] for f in folds]) * 100)),
    }
    return {'folds': folds, 'resumen': resumen}


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

    cv = calcular_validacion_cruzada(base_samples, k=5)

    riesgos = [
        {
            'nivel': 'Alto',
            'nombre': 'Datos limitados',
            'descripcion': 'Se evalúa el modelo con una muestra pequeña de casos reales.',
            'indicador': 'Pocas observaciones para generalizar',
            'recomendacion': 'Recolectar más datos históricos antes de producir decisiones comerciales.',
        },
        {
            'nivel': 'Medio',
            'nombre': 'Sensibilidad al clima',
            'descripcion': 'Las predicciones cambian según la condición meteorológica y la zona.',
            'indicador': 'MAPE moderado y variación entre pliegues',
            'recomendacion': 'Validar con nuevas condiciones meteorológicas y de tráfico.',
        },
        {
            'nivel': 'Bajo',
            'nombre': 'Evolución estacional',
            'descripcion': 'El comportamiento de ocupación puede variar según la temporada.',
            'indicador': 'Cambios futuros en el patrón de uso',
            'recomendacion': 'Reentrenar el modelo con datos recientes cada cierto tiempo.',
        },
    ]

    reales = [item['real'] for item in base_samples]
    promedio_real = sum(reales) / len(reales)
    baseline_metrics = calcular_metricas([
        {'real': real, 'pred': promedio_real} for real in reales
    ])
    mb = {'r2': baseline_metrics['r2'], 'rmse': baseline_metrics['rmse'], 'mae': baseline_metrics['mae']}

    return render_template('fase3.html', m=m, cv=cv, riesgos=riesgos, mb=mb, active_page='evaluation')

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
 