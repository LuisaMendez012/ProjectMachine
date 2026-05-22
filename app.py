from flask import Flask, render_template, request
from model import predecir_ocupacion
from model_rl import train_q_learning, get_q_table
import os

app = Flask(__name__)

# -------- FASE 1: ENTENDIMIENTO --------
@app.route('/')
def fase1():
    return render_template('fase1.html')

# -------- FASE 2: MODELO --------
@app.route('/modelo', methods=['GET', 'POST'])
def fase2():
    resultado = None
 
    if request.method == 'POST':
        clima = float(request.form['clima'])
        hora  = float(request.form['hora'])
        zona  = float(request.form['zona'])
        resultado = predecir_ocupacion(clima, hora, zona)
 
    return render_template('fase2.html', resultado=resultado)
 
# -------- RL --------
@app.route('/rl')
def rl_menu():
    return render_template('rl.html')
 
@app.route('/rl/concepts')
def rl_concepts():
    return render_template('rl_concepts.html')
 
@app.route('/rl/agent')
def rl_agent():
    return render_template('rl_agent.html')
 
@app.route('/rl/train')
def rl_train():
    train_q_learning()
    return "Training completed. <a href='/rl/results'>See results</a>"
 
@app.route('/rl/results')
def rl_results():
    q = get_q_table()
    return render_template('rl_results.html', q=q)
 
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
 