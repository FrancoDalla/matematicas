"""
Este vibe coding fue tremendo. Deepseek robo el código que había robado yo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TU CÓDIGO ORIGINAL
def cost_function(X, y, theta):
    m = y.size
    error = np.dot(X, theta.T) - y
    cost = 1/(2*m) * np.dot(error.T, error)
    return cost, error

def gradient_descent(X, y, theta, alpha, iters):
    cost_array = np.zeros(iters)
    m = y.size
    for i in range(iters):
        cost, error = cost_function(X, y, theta)
        theta = theta - (alpha * (1/m) * np.dot(X.T, error))
        cost_array[i] = cost
    return theta, cost_array

def plotChart(iterations, cost_num):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost_num, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs Iterations')
    plt.style.use('fivethirtyeight')
    plt.show()

# CÓDIGO DE VALIDACIÓN
def gradient_descent_validation(X, y, alpha=0.01, iterations=1000):
    """Implementación estándar para validación"""
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    
    for i in range(iterations):
        # Calcular gradiente
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        
        # Actualizar theta
        theta = theta - alpha * gradient
        
        # Calcular costo
        cost = (1/(2*m)) * np.sum(errors**2)
        cost_history.append(cost)
        
    return theta, cost_history

def validate_with_reference():
    # Usa los mismos datos que en tu implementación
    data = pd.read_csv('dataset.csv')
    X = data[['indice_tiempo', 'superficie_sembrada_mani_ha', 
              'superficie_cosechada_mani_ha', 'produccion_mani_t']]
    y = data['rendimiento_mani_kgxha']
    
    # Normalizar
    X = (X - X.mean()) / X.std()
    X = np.c_[np.ones(X.shape[0]), X]
    
    # Ejecutar ambas implementaciones
    theta_yours, cost_yours = gradient_descent(X, y, np.zeros(X.shape[1]), 0.01, 1000)
    theta_ref, cost_ref = gradient_descent_validation(X, y, 0.01, 1000)
    
    print("=" * 60)
    print("VALIDACIÓN DE RESULTADOS")
    print("=" * 60)
    print("Tus thetas:", theta_yours)
    print("Thetas referencia:", theta_ref)
    print("Diferencia:", np.abs(theta_yours - theta_ref))
    print("Diferencia máxima:", np.max(np.abs(theta_yours - theta_ref)))
    print("Diferencia promedio:", np.mean(np.abs(theta_yours - theta_ref)))
    
    # Comparar costos finales
    final_cost_yours = cost_yours[-1]
    final_cost_ref = cost_ref[-1]
    print("\nCOMPARACIÓN DE COSTOS:")
    print(f"Tu costo final: {final_cost_yours:.6f}")
    print(f"Costo referencia: {final_cost_ref:.6f}")
    print(f"Diferencia: {abs(final_cost_yours - final_cost_ref):.6f}")

def calcular_R2(theta_final, X, y):

    # Predicciones de tu modelo
    y_pred = X.dot(theta_final)

    # Cálculo del R²
    ss_res = np.sum((y - y_pred) ** 2)  # Suma de cuadrados de residuos
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Suma de cuadrados total
    r_squared = 1 - (ss_res / ss_tot)

    print(f'R² = {r_squared:.4f}')
    print(f'El modelo explica el {r_squared*100:.1f}% de la varianza')

def run():
    # Import data
    data = pd.read_csv('dataset.csv')

    # Extract data into X and y
    X = data[['indice_tiempo', 'superficie_sembrada_mani_ha', 'superficie_cosechada_mani_ha', 'produccion_mani_t']]
    y = data['rendimiento_mani_kgxha']

    # Normalize our features
    X = (X - X.mean()) / X.std()

    # Add a 1 column to the start to allow vectorized gradient descent
    X = np.c_[np.ones(X.shape[0]), X] 

    # Set hyperparameters
    alpha = 0.01
    iterations = 1000

    # Initialize Theta Values to 0
    theta = np.zeros(X.shape[1])
    initial_cost, _ = cost_function(X, y, theta)

    print('With initial theta values of {0}, cost error is {1}'.format(theta, initial_cost))

    # Run Gradient Descent
    theta, cost_num = gradient_descent(X, y, theta, alpha, iterations)

    # Display cost chart
    plotChart(iterations, cost_num)

    final_cost, _ = cost_function(X, y, theta)

    print('With final theta values of {0}, cost error is {1}'.format(theta, final_cost))
    
    # Ejecutar validación
    validate_with_reference()

    calcular_R2(theta, X, y)

if __name__ == "__main__":
    run()