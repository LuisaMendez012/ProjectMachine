def predecir_ocupacion(clima, hora, zona):
    # Pesos simulados (como tu circuito)
    w1 = 0.5
    w2 = 0.3
    w3 = 0.2

    # Bias
    b = 0.1

    # Modelo lineal
    resultado = (clima * w1) + (hora * w2) + (zona * w3) + b

    return round(resultado, 2)