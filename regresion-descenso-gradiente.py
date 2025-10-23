import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MultipleLinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iter=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.coef_ = None
        self.intercept_ = None
        self.loss_history = []

    def fit(self, X, y):
        """
        X: matriz de características (n_samples, n_features)
        y: vector objetivo (n_samples,)
        """
        # Agregar columna de unos para el intercepto
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

        # Inicializar parámetros
        n_samples, n_features = X_with_intercept.shape
        self.theta = np.zeros(n_features)  # [intercept, coef1, coef2, ...]

        for i in range(self.n_iter):
            # Calcular predicciones
            y_pred = X_with_intercept.dot(self.theta)

            # Calcular error
            error = y_pred - y

            # Calcular gradiente
            gradient = (1 / n_samples) * X_with_intercept.T.dot(error)

            # Guardar pérdida (MSE)
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            # Actualizar parámetros
            new_theta = self.theta - self.learning_rate * gradient

            # Verificar convergencia
            if np.linalg.norm(new_theta - self.theta) < self.tolerance:
                print(f"Convergió en la iteración {i+1}")
                break

            self.theta = new_theta

        # Separar intercept y coeficientes
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

        return self

    def predict(self, X):
        """Realizar predicciones"""
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        return X_with_intercept.dot(self.theta)

    def get_params(self):
        """Obtener parámetros del modelo"""
        return {
            'intercept': self.intercept_,
            'coefficients': self.coef_
        }

# EJEMPLO DE USO CON DATOS SIMULADOS (reemplaza con tus datos reales)
def ejemplo_completo():
    # Generar datos de ejemplo (reemplaza con tu dataset real)
    np.random.seed(42)
    n_samples = 100

    # Variables predictoras
    año = np.random.randint(2000, 2020, n_samples)
    superficie_sembrada = np.random.uniform(100, 1000, n_samples)
    superficie_cosechada = superficie_sembrada * np.random.uniform(0.8, 1.0, n_samples)
    produccion = superficie_cosechada * np.random.uniform(2, 5, n_samples)

    # Variable dependiente (Rendimiento)
    rendimiento = (10 +
                  0.5 * año +
                  0.8 * superficie_sembrada +
                  1.2 * superficie_cosechada +
                  0.3 * produccion +
                  np.random.normal(0, 10, n_samples))

    # Crear matriz de características
    X = np.column_stack([año, superficie_sembrada, superficie_cosechada, produccion])
    y = rendimiento

    # Estandarizar características (opcional, ayuda con la convergencia)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standardized = (X - X_mean) / X_std

    # Entrenar modelo con descenso de gradiente
    print("=== DESCESO DE GRADIENTE ===")
    model_gd = MultipleLinearRegressionGD(learning_rate=0.01, n_iter=5000)
    model_gd.fit(X_standardized, y)

    # Obtener parámetros
    params = model_gd.get_params()
    print(f"Intercepto (β₀): {params['intercept']:.4f}")
    for i, coef in enumerate(params['coefficients']):
        print(f"Coeficiente β{i+1}: {coef:.4f}")

    # Graficar evolución de la pérdida
    plt.figure(figsize=(10, 6))
    plt.plot(model_gd.loss_history)
    plt.title('Evolución de la Función de Pérdida')
    plt.xlabel('Iteración')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.show()

    # Calcular métricas
    y_pred_gd = model_gd.predict(X_standardized)
    mse_gd = np.mean((y - y_pred_gd) ** 2)
    r2_gd = 1 - np.sum((y - y_pred_gd) ** 2) / np.sum((y - np.mean(y)) ** 2)

    print(f"\nMétricas del Modelo:")
    print(f"MSE: {mse_gd:.4f}")
    print(f"R²: {r2_gd:.4f}")

    return model_gd, X, y

# FUNCIÓN PARA TUS DATOS REALES
def usar_con_tus_datos():
    """
    Reemplaza esta función con la carga de tus datos reales
    """
    # CARGAR TUS DATOS AQUÍ (ejemplo con CSV)
    # df = pd.read_csv('tu_archivo.csv')
    # X = df[['Año', 'SuperficieSembrada', 'SuperficieCosechada', 'Produccion']].values
    # y = df['Rendimiento'].values

    # Por ahora usamos datos de ejemplo
    return ejemplo_completo()

# COMPARACIÓN CON MÍNIMOS CUADRADOS
def comparar_con_minimos_cuadrados(X, y):
    """Comparar con solución por mínimos cuadrados"""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    print("\n=== MÍNIMOS CUADRADOS ===")
    model_ols = LinearRegression()
    model_ols.fit(X, y)

    print(f"Intercepto: {model_ols.intercept_:.4f}")
    for i, coef in enumerate(model_ols.coef_):
        print(f"Coeficiente β{i+1}: {coef:.4f}")

    y_pred_ols = model_ols.predict(X)
    mse_ols = mean_squared_error(y, y_pred_ols)
    r2_ols = r2_score(y, y_pred_ols)

    print(f"\nMétricas Mínimos Cuadrados:")
    print(f"MSE: {mse_ols:.4f}")
    print(f"R²: {r2_ols:.4f}")

    return model_ols

if __name__ == "__main__":
    # Ejecutar ejemplo completo
    model_gd, X, y = usar_con_tus_datos()

    # Comparar con mínimos cuadrados
    model_ols = comparar_con_minimos_cuadrados(X, y)
