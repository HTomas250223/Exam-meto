import numpy as np

# Función 2: f(x) = e^x - x^2 + 4
def f2(x):
    return np.exp(x) - x**2 + 4

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

# Aplicación para f2(x) = e^x - x^2 + 4
raiz_biseccion_f2, iter_biseccion_f2 = biseccion(f2, -3, -2, 1e-5)
print(f"Raíz (Bisección, f2): {raiz_biseccion_f2}, Iteraciones: {iter_biseccion_f2}")

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

# Aplicación para f2(x) = e^x - x^2 + 4
raiz_falsa_pos_f2, iter_falsa_pos_f2 = falsa_posicion(f2, -3, -2, 1e-5)
print(f"Raíz (Falsa Posición, f2): {raiz_falsa_pos_f2}, Iteraciones: {iter_falsa_pos_f2}")

# Derivada de la función f2
def df2(x):
    return np.exp(x) - 2*x

# Implementación del Método de Newton-Raphson
def newton_raphson(f, df, x0, tol):
    iteraciones = 0
    while True:
        iteraciones += 1
        x1 = x0 - f(x0) / df(x0)
        if abs(f(x1)) < tol:
            return x1, iteraciones
        x0 = x1

# Aplicación para f2(x) = e^x - x^2 + 4
raiz_newton_f2, iter_newton_f2 = newton_raphson(f2, df2, -2.5, 1e-5)
print(f"Raíz (Newton-Raphson, f2): {raiz_newton_f2}, Iteraciones: {iter_newton_f2}")

# Implementación del Método de la Secante
def secante(f, x0, x1, tol):
    iteraciones = 0
    while True:
        iteraciones += 1
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(f(x2)) < tol:
            return x2, iteraciones
        x0, x1 = x1, x2

# Aplicación para f2(x) = e^x - x^2 + 4
raiz_secante_f2, iter_secante_f2 = secante(f2, -3, -2, 1e-5)
print(f"Raíz (Secante, f2): {raiz_secante_f2}, Iteraciones: {iter_secante_f2}")

# Función de iteración del punto fijo para f2
def g2(x):
    return np.log(x**2 - 4)

# Función de iteración del punto fijo para f2
def g2(x):
    return np.log(x**2 - 4)

# Implementación del Método del Punto Fijo
def punto_fijo(g, x0, tol):
    iteraciones = 0
    while True:
        iteraciones += 1
        x1 = g(x0)
        if abs(x1 - x0) < tol:
            return x1, iteraciones
        x0 = x1

# Aplicación para g2(x) = ln(x^2 - 4)
raiz_punto_fijo_f2, iter_punto_fijo_f2 = punto_fijo(g2, -2.5, 1e-5)
print(f"Raíz (Punto Fijo, f2): {raiz_punto_fijo_f2}, Iteraciones: {iter_punto_fijo_f2}")

