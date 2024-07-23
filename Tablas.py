import numpy as np
import pandas as pd

# Función 1: f(x) = cos(x) - x
def f1(x):
    return np.cos(x) - x

# Derivada de f1(x) para Newton-Raphson
def df1(x):
    return -np.sin(x) - 1

# Función g para Punto Fijo de f1
def g1(x):
    return np.cos(x)

# Función 2: f(x) = e^x - x^2 + 4
def f2(x):
    return np.exp(x) - x**2 + 4

# Derivada de f2(x) para Newton-Raphson
def df2(x):
    return np.exp(x) - 2*x

# Función g para Punto Fijo de f2
# Use a safe evaluation for g2 to prevent overflow in exp(x)
def g2(x):
    try:
        result = np.sqrt(np.exp(x) + 4)
        return result if np.isfinite(result) else x  # Return x if result is not finite
    except OverflowError:
        return x  # Return x if overflow occurs

# Método de Bisección
def biseccion(f, a, b, tol):
    iteraciones = 0
    while (b - a) / 2 > tol:
        iteraciones += 1
        c = (a + b) / 2
        if f(c) == 0:
            return c, iteraciones
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2, iteraciones

# Método de Falsa Posición
def falsa_posicion(f, a, b, tol):
    iteraciones = 0
    while True:
        iteraciones += 1
        c = b - (f(b) * (a - b)) / (f(a) - f(b))
        if abs(f(c)) < tol or abs(b - a) < tol:
            return c, iteraciones
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

# Método de Newton-Raphson
def newton_raphson(f, df, x0, tol):
    iteraciones = 0
    while True:
        iteraciones += 1
        x1 = x0 - f(x0) / df(x0)
        if abs(f(x1)) < tol:
            return x1, iteraciones
        x0 = x1

# Método de la Secante
def secante(f, x0, x1, tol):
    iteraciones = 0
    while True:
        iteraciones += 1
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(f(x2)) < tol:
            return x2, iteraciones
        x0, x1 = x1, x2

# Método del Punto Fijo
def punto_fijo(g, x0, tol, max_iter=1000):
    iteraciones = 0
    while iteraciones < max_iter:
        iteraciones += 1
        x1 = g(x0)
        if abs(x1 - x0) < tol:
            return x1, iteraciones
        x0 = x1
    return x0, iteraciones  # Return the best guess if max_iter is reached

# Tolerancia
tol = 1e-5

# Calcular las raíces y el número de iteraciones para cada método
# Función 1: f(x) = cos(x) - x
raiz_biseccion1, iter_biseccion1 = biseccion(f1, 0, 1, tol)
raiz_falsa_pos1, iter_falsa_pos1 = falsa_posicion(f1, 0, 1, tol)
raiz_newton1, iter_newton1 = newton_raphson(f1, df1, 0.5, tol)
raiz_secante1, iter_secante1 = secante(f1, 0, 1, tol)
raiz_punto_fijo1, iter_punto_fijo1 = punto_fijo(g1, 0.5, tol)

# Función 2: f(x) = e^x - x^2 + 4
raiz_biseccion2, iter_biseccion2 = biseccion(f2, -3, -2, tol)
raiz_falsa_pos2, iter_falsa_pos2 = falsa_posicion(f2, -3, -2, tol)
raiz_newton2, iter_newton2 = newton_raphson(f2, df2, -2.5, tol)
raiz_secante2, iter_secante2 = secante(f2, -3, -2, tol)
raiz_punto_fijo2, iter_punto_fijo2 = punto_fijo(g2, -2.5, tol)

# Crear la tabla de comparación
data = {
    "Método": ["Bisección", "Falsa Posición", "Newton-Raphson", "Secante", "Punto Fijo"],
    "Raíz f1(x)": [raiz_biseccion1, raiz_falsa_pos1, raiz_newton1, raiz_secante1, raiz_punto_fijo1],
    "Iteraciones f1(x)": [iter_biseccion1, iter_falsa_pos1, iter_newton1, iter_secante1, iter_punto_fijo1],
    "Raíz f2(x)": [raiz_biseccion2, raiz_falsa_pos2, raiz_newton2, raiz_secante2, raiz_punto_fijo2],
    "Iteraciones f2(x)": [iter_biseccion2, iter_falsa_pos2, iter_newton2, iter_secante2, iter_punto_fijo2]
}

# Convertir a DataFrame
tabla_comparacion = pd.DataFrame(data)
print(tabla_comparacion)

