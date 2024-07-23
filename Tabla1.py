import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
results = {
    "Método": [],
    "Raíz f1(x)": [],
    "Iteraciones f1(x)": [],
    "Raíz f2(x)": [],
    "Iteraciones f2(x)": []
}

# Métodos que no requieren derivada
methods_no_derivative = [
    ("Bisección", biseccion, f1, 0, 1, tol, f2, -3, -2),
    ("Falsa Posición", falsa_posicion, f1, 0, 1, tol, f2, -3, -2)
]

# Métodos que requieren derivada
methods_with_derivative = [
    ("Newton-Raphson", newton_raphson, f1, df1, 0.5, tol, f2, df2, -2.5, tol),
    ("Secante", secante, f1, 0, 1, tol, f2, -3, -2, tol)
]

# Ejecutar métodos sin derivada
for method_name, method_func, f1_func, a1, b1, tol, f2_func, a2, b2 in methods_no_derivative:
    raiz_f1, iter_f1 = method_func(f1_func, a1, b1, tol)
    raiz_f2, iter_f2 = method_func(f2_func, a2, b2, tol)
    
    results["Método"].append(method_name)
    results["Raíz f1(x)"].append(raiz_f1)
    results["Iteraciones f1(x)"].append(iter_f1)
    results["Raíz f2(x)"].append(raiz_f2)
    results["Iteraciones f2(x)"].append(iter_f2)

# Ejecutar métodos con derivada
for method_name, method_func, f1_func, df1_func, x0_1, tol, f2_func, df2_func, x0_2, tol_2 in methods_with_derivative:
    raiz_f1, iter_f1 = method_func(f1_func, df1_func, x0_1, tol)
    raiz_f2, iter_f2 = method_func(f2_func, df2_func, x0_2, tol_2)
    
    results["Método"].append(method_name)
    results["Raíz f1(x)"].append(raiz_f1)
    results["Iteraciones f1(x)"].append(iter_f1)
    results["Raíz f2(x)"].append(raiz_f2)
    results["Iteraciones f2(x)"].append(iter_f2)

# Ejecutar Punto Fijo
raiz_punto_fijo1, iter_punto_fijo1 = punto_fijo(g1, 0.5, tol)
raiz_punto_fijo2, iter_punto_fijo2 = punto_fijo(g2, -2.5, tol)

results["Método"].append("Punto Fijo")
results["Raíz f1(x)"].append(raiz_punto_fijo1)
results["Iteraciones f1(x)"].append(iter_punto_fijo1)
results["Raíz f2(x)"].append(raiz_punto_fijo2)
results["Iteraciones f2(x)"].append(iter_punto_fijo2)

# Convertir a DataFrame
tabla_comparacion = pd.DataFrame(results)

# Crear la tabla de comparación y guardarla como imagen
plt.figure(figsize=(12, 6))
sns.heatmap(tabla_comparacion.set_index("Método"), annot=True, fmt=".6f", cmap="YlGnBu", linewidths=.5)
plt.title("Comparación de Métodos Numéricos")
plt.savefig("tabla_comparacion.png", bbox_inches='tight')
plt.show()
