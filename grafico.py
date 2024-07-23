import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definir las funciones
def f1(x):
    return np.cos(x) - x

def df1(x):
    return -np.sin(x) - 1

def g1(x):
    return np.cos(x)

def f2(x):
    return np.exp(x) - x**2 + 4

def df2(x):
    return np.exp(x) - 2*x

def g2(x):
    try:
        result = np.sqrt(np.exp(x) + 4)
        return result if np.isfinite(result) else x
    except OverflowError:
        return x

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

def newton_raphson(f, df, x0, tol):
    iteraciones = 0
    while True:
        iteraciones += 1
        x1 = x0 - f(x0) / df(x0)
        if abs(f(x1)) < tol:
            return x1, iteraciones
        x0 = x1

def secante(f, x0, x1, tol):
    iteraciones = 0
    while True:
        iteraciones += 1
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(f(x2)) < tol:
            return x2, iteraciones
        x0, x1 = x1, x2

def punto_fijo(g, x0, tol, max_iter=1000):
    iteraciones = 0
    while iteraciones < max_iter:
        iteraciones += 1
        x1 = g(x0)
        if abs(x1 - x0) < tol:
            return x1, iteraciones
        x0 = x1
    return x0, iteraciones

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

# Convertir a DataFrame
tabla_comparacion = pd.DataFrame(results)
print(tabla_comparacion)

# Graficar
x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-4, 2, 400)

f1_values = f1(x1)
f2_values = f2(x2)

plt.figure(figsize=(14, 10))

# Graficar f1(x)
plt.subplot(2, 1, 1)
plt.plot(x1, f1_values, label='f1(x) = cos(x) - x', color='blue')
for method_name, raiz_f1, _ in zip(results["Método"], results["Raíz f1(x)"], results["Iteraciones f1(x)"]):
    plt.plot(raiz_f1, f1(raiz_f1), 'ro', label=f'{method_name} Raíz f1(x)')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.title('Gráfico de f1(x) con las raíces encontradas')
plt.xlabel('x')
plt.ylabel('f1(x)')
plt.legend()
plt.grid(True)

# Graficar f2(x)
plt.subplot(2, 1, 2)
plt.plot(x2, f2_values, label='f2(x) = e^x - x^2 + 4', color='green')
for method_name, raiz_f2, _ in zip(results["Método"], results["Raíz f2(x)"], results["Iteraciones f2(x)"]):
    plt.plot(raiz_f2, f2(raiz_f2), 'ro', label=f'{method_name} Raíz f2(x)')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.title('Gráfico de f2(x) con las raíces encontradas')
plt.xlabel('x')
plt.ylabel('f2(x)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
