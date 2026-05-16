from flask import Flask, render_template, request
from model import predecir_ocupacion

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
        hora = float(request.form['hora'])
        zona = float(request.form['zona'])

        resultado = predecir_ocupacion(clima, hora, zona)

    return render_template('fase2.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)