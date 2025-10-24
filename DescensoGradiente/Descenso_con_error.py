"""
Codigo para la regresión lineal mediante el metodo de descenso del gradiente con criterio de convergencia.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cost_function(X, y, theta):
    m = y.size
    error = np.dot(X, theta.T) - y
    cost = 1/(2*m) * np.dot(error.T, error)
    return cost, error

def gradient_descent(X, y, theta, alpha, max_iters, tolerance=1e-6, min_cost_change=1e-8):
    """
    Descenso del gradiente con criterio de convergencia

    Parameters:
    - X: matriz de características
    - y: vector objetivo
    - theta: parámetros iniciales
    - alpha: tasa de aprendizaje
    - max_iters: máximo número de iteraciones
    - tolerance: error aceptable para la convergencia
    - min_cost_change: cambio mínimo en el costo para considerar convergencia
    """
    cost_array = []
    m = y.size
    previous_cost = float('inf')

    for i in range(max_iters):
        cost, error = cost_function(X, y, theta)
        cost_array.append(cost)

        # Verificar criterios de convergencia
        if i > 0:
            cost_change = abs(previous_cost - cost)

            # Criterio 1: Cambio en el costo muy pequeño
            if cost_change < min_cost_change:
                print(f'Convergencia alcanzada: cambio en costo ({cost_change:.2e}) < {min_cost_change}')
                break

            # Criterio 2: Costo por debajo del error aceptable
            if cost < tolerance:
                print(f'Convergencia alcanzada: costo ({cost:.2e}) < tolerancia ({tolerance})')
                break

        # Actualizar parámetros
        theta = theta - (alpha * (1/m) * np.dot(X.T, error))
        previous_cost = cost

        # Mostrar progreso cada 100 iteraciones
        if i % 100 == 0:
            print(f'Iteración {i}: Costo = {cost:.6f}')

    return theta, np.array(cost_array)

def plotChart(cost_num):
    """
    Grafica la evolución del costo
    """
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(cost_num)), cost_num, 'r-', linewidth=2)
    ax.set_xlabel('Iteraciones')
    ax.set_ylabel('Costo (J)')
    ax.set_title('Evolución del Costo vs Iteraciones')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def run():
    # Leer datos del dataset
    data = pd.read_csv('dataset.csv')

    # Se extraen los valores independientes en X y el dependiente en Y.
    X = data[['indice_tiempo', 'superficie_sembrada_mani_ha', 'superficie_cosechada_mani_ha', 'produccion_mani_t']]
    y = data['rendimiento_mani_kgxha']

    # Normalizar características
    X_normalized = (X - X.mean()) / X.std()

    # Añadir columna de unos para el término de bias
    X_final = np.c_[np.ones(X_normalized.shape[0]), X_normalized]

    # Set hyperparameters
    alpha = 0.01
    max_iterations = 10000  # Máximo más alto para permitir convergencia
    tolerance = 1e-4       # Error aceptable
    min_cost_change = 1e-8 # Cambio mínimo en costo para convergencia

    # Initialize Theta Values to 0
    theta = np.zeros(X_final.shape[1])
    initial_cost, _ = cost_function(X_final, y, theta)

    print('=' * 60)
    print('REGRESIÓN LINEAL - DESCENSO DEL GRADIENTE')
    print('=' * 60)
    print(f'Parámetros iniciales theta: {theta}')
    print(f'Costo inicial: {initial_cost:.6f}')
    print(f'Tasa de aprendizaje (alpha): {alpha}')
    print(f'Error aceptable (tolerancia): {tolerance}')
    print(f'Máximo de iteraciones: {max_iterations}')
    print('-' * 60)

    # Run Gradient Descent
    theta_final, cost_history = gradient_descent(
        X_final, y, theta, alpha, max_iterations, tolerance, min_cost_change
    )

    print('-' * 60)
    print(f'Iteraciones ejecutadas: {len(cost_history)}')
    print(f'Parámetros finales theta: {theta_final}')

    final_cost, _ = cost_function(X_final, y, theta_final)
    print(f'Costo final: {final_cost:.6f}')
    print(f'Reducción del costo: {((initial_cost - final_cost)/initial_cost * 100):.2f}%')
    print('=' * 60)

    # Display cost chart
    plotChart(cost_history)

    # Mostrar métricas adicionales
    print('\nMÉTRICAS ADICIONALES:')
    print('-' * 30)

    # Predicciones
    predictions = np.dot(X_final, theta_final.T)

    # Error cuadrático medio (MSE)
    mse = np.mean((predictions - y) ** 2)
    print(f'Error Cuadrático Medio (MSE): {mse:.4f}')

    # R-cuadrado
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f'R-cuadrado (R²): {r_squared:.4f}')

if __name__ == "__main__":
    run()
