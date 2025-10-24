"""
Codigo para la regresión lineal mediante el metodo de descenso del gradiente
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cost_function(X, y, theta):
    """
    Función de costo f(θ) para regresión lineal
    """
    m = y.size
    error = np.dot(X, theta.T) - y
    cost = 1/(2*m) * np.dot(error.T, error)
    return cost, error

def gradient_descent(X, y, theta, eta, max_iters, epsilon):
    """
    Descenso del gradiente con condición de corte estricta: |f(xₖ₊₁) - f(xₖ)| < ε

    Parametros:
    - X: matriz de características (punto inicial x₀ implícito en theta)
    - y: vector objetivo
    - theta: parámetros iniciales (x₀)
    - eta: tamaño del paso (η)
    - max_iters: máximo número de iteraciones
    - epsilon: tolerancia (ε)
    """
    cost_history = []
    m = y.size

    # Calcular costo inicial f(x₀)
    current_cost, error = cost_function(X, y, theta)
    cost_history.append(current_cost)

    print(f'Iteración 0: f(θ) = {current_cost:.6f}')

    for i in range(1, max_iters + 1):
        # Guardar costo anterior f(xₖ)
        previous_cost = current_cost

        # Calcular siguiente punto según función de actualización: xₜ₊₁ = xₜ - η∇f(xₜ)
        gradient = (1/m) * np.dot(X.T, error)
        theta = theta - eta * gradient

        # Calcular nuevo costo f(xₖ₊₁)
        current_cost, error = cost_function(X, y, theta)
        cost_history.append(current_cost)

        # Calcular diferencia |f(xₖ₊₁) - f(xₖ)|
        cost_difference = abs(current_cost - previous_cost)

        print(f'Iteración {i}: f(θ) = {current_cost:.6f}, |Δf| = {cost_difference:.2e}')

        # Verificar condición de corte: |f(xₖ₊₁) - f(xₖ)| < ε
        if cost_difference < epsilon:
            print(f'✓ Condición de corte alcanzada: |f(xₖ₊₁) - f(xₖ)| = {cost_difference:.2e} < ε = {epsilon}')
            break

    return theta, np.array(cost_history)


def run():
    # 1. Leer datos del dataset
    data = pd.read_csv('dataset.csv')

    # 2. Extraer variables independientes (X) y dependiente (y)
    X = data[['indice_tiempo', 'superficie_sembrada_mani_ha', 'superficie_cosechada_mani_ha', 'produccion_mani_t']]
    y = data['rendimiento_mani_kgxha']

    # 3. Normalizar características
    X_normalized = (X - X.mean()) / X.std()

    # 4. Añadir columna de unos para el término de bias
    X_final = np.c_[np.ones(X_normalized.shape[0]), X_normalized]

    # 5. Parámetros del algoritmo (según la especificación)
    eta = 0.01          # η - tamaño del paso
    epsilon = 1e-6      # ε - tolerancia
    max_iters = 10000   # máximo número de iteraciones

    # 6. Inicializar parámetros (x₀)
    theta = np.zeros(X_final.shape[1])

    # 7. Calcular costo inicial f(x₀)
    initial_cost, _ = cost_function(X_final, y, theta)

    print('=' * 70)
    print('DESCENSO DEL GRADIENTE - CONDICIÓN DE CORTE: |f(xₖ₊₁) - f(xₖ)| < ε')
    print('=' * 70)
    print(f'Punto inicial θ₀: {theta}')
    print(f'Costo inicial f(θ₀): {initial_cost:.6f}')
    print(f'Tamaño del paso (η): {eta}')
    print(f'Tolerancia (ε): {epsilon}')
    print(f'Máximo de iteraciones: {max_iters}')
    print('-' * 70)

    # 8. Ejecutar Descenso del Gradiente
    theta_final, cost_history = gradient_descent(
        X_final, y, theta, eta, max_iters, epsilon
    )

    print('-' * 70)
    print('RESULTADOS FINALES:')
    print('-' * 70)
    print(f'Iteraciones ejecutadas: {len(cost_history) - 1}')
    print(f'Parámetros finales θ: {theta_final}')

    final_cost, _ = cost_function(X_final, y, theta_final)
    print(f'Costo final f(θ): {final_cost:.6f}')

    # Verificar condición de corte final
    if len(cost_history) > 1:
        final_difference = abs(cost_history[-1] - cost_history[-2])
        print(f'Diferencia final |Δf|: {final_difference:.2e}')
        print(f'Condición |Δf| < ε: {final_difference:.2e} < {epsilon} = {final_difference < epsilon}')

    print('=' * 70)

    # Predicciones finales
    predictions = np.dot(X_final, theta_final.T)

    # Error cuadrático medio
    mse = np.mean((predictions - y) ** 2)
    print(f'Error Cuadrático Medio (MSE): {mse:.4f}')

    # R-cuadrado
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f'Coeficiente R²: {r_squared:.4f}')

if __name__ == "__main__":
    run()
