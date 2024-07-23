import numpy as np

# Función 1: f(x) = cos(x) - x
def f1(x):
    return np.cos(x) - x

# Implementación del Método de Bisección
def biseccion(f, a, b, tol):
    iteraciones = 0
    while (b - a) / 2 > tol:
        iteraciones += 1
        c = (a + b) / 2
        if f(c) == 0:
            return c, iteraciones  # Se encontró la raíz exacta
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2, iteraciones

# Aplicación para f1(x) = cos(x) - x
raiz_biseccion_f1, iter_biseccion_f1 = biseccion(f1, 0, 1, 1e-5)
print(f"Raíz (Bisección, f1): {raiz_biseccion_f1}, Iteraciones: {iter_biseccion_f1}")
# Implementación del Método de Falsa Posición
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

# Aplicación para f1(x) = cos(x) - x
raiz_falsa_pos_f1, iter_falsa_pos_f1 = falsa_posicion(f1, 0, 1, 1e-5)
print(f"Raíz (Falsa Posición, f1): {raiz_falsa_pos_f1}, Iteraciones: {iter_falsa_pos_f1}")

# Derivada de la función f1
def df1(x):
    return -np.sin(x) - 1

# Implementación del Método de Newton-Raphson
def newton_raphson(f, df, x0, tol):
    iteraciones = 0
    while True:
        iteraciones += 1
        x1 = x0 - f(x0) / df(x0)
        if abs(f(x1)) < tol:
            return x1, iteraciones
        x0 = x1

# Aplicación para f1(x) = cos(x) - x
raiz_newton_f1, iter_newton_f1 = newton_raphson(f1, df1, 0.5, 1e-5)
print(f"Raíz (Newton-Raphson, f1): {raiz_newton_f1}, Iteraciones: {iter_newton_f1}")

# Implementación del Método de la Secante
def secante(f, x0, x1, tol):
    iteraciones = 0
    while True:
        iteraciones += 1
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(f(x2)) < tol:
            return x2, iteraciones
        x0, x1 = x1, x2

# Aplicación para f1(x) = cos(x) - x
raiz_secante_f1, iter_secante_f1 = secante(f1, 0, 1, 1e-5)
print(f"Raíz (Secante, f1): {raiz_secante_f1}, Iteraciones: {iter_secante_f1}")

# Función de iteración del punto fijo para f1
def g1(x):
    return np.cos(x)

# Implementación del Método del Punto Fijo
def punto_fijo(g, x0, tol):
    iteraciones = 0
    while True:
        iteraciones += 1
        x1 = g(x0)
        if abs(x1 - x0) < tol:
            return x1, iteraciones
        x0 = x1

# Aplicación para g1(x) = cos(x)
raiz_punto_fijo_f1, iter_punto_fijo_f1 = punto_fijo(g1, 0.5, 1e-5)
print(f"Raíz (Punto Fijo, f1): {raiz_punto_fijo_f1}, Iteraciones: {iter_punto_fijo_f1}")
