import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RegresionLinealMultivariable:
    def __init__(self, tasa_aprendizaje=0.01, iteraciones=1000):
        """
        Inicializa el modelo de regresión lineal multivariable
        
        Parámetros:
        tasa_aprendizaje (float): Tasa de aprendizaje para el descenso de gradiente
        iteraciones (int): Número de iteraciones para el entrenamiento
        """
        self.tasa_aprendizaje = tasa_aprendizaje
        self.iteraciones = iteraciones
        self.pesos = None
        self.sesgo = None
        self.historial_costos = []
    
    def normalizar_caracteristicas(self, X):
        """Normaliza las características para mejorar la convergencia"""
        self.media = np.mean(X, axis=0)
        self.desviacion = np.std(X, axis=0)
        # Evitar división por cero
        self.desviacion = np.where(self.desviacion == 0, 1, self.desviacion)
        return (X - self.media) / self.desviacion
    
    def hipotesis(self, X):
        """Calcula la hipótesis h(x) = X * pesos + sesgo"""
        return np.dot(X, self.pesos) + self.sesgo
    
    def calcular_costo(self, X, y):
        """Calcula el costo (error cuadrático medio)"""
        m = len(y)
        predicciones = self.hipotesis(X)
        error = predicciones - y
        costo = (1/(2*m)) * np.sum(error**2)
        return costo
    
    def descenso_gradiente(self, X, y):
        """Implementa el algoritmo de descenso de gradiente"""
        m, n = X.shape
        self.pesos = np.zeros(n)
        self.sesgo = 0
        
        for i in range(self.iteraciones):
            # Calcular predicciones
            predicciones = self.hipotesis(X)
            
            # Calcular errores
            error = predicciones - y
            
            # Calcular gradientes
            gradiente_pesos = (1/m) * np.dot(X.T, error)
            gradiente_sesgo = (1/m) * np.sum(error)
            
            # Actualizar parámetros
            self.pesos -= self.tasa_aprendizaje * gradiente_pesos
            self.sesgo -= self.tasa_aprendizaje * gradiente_sesgo
            
            # Guardar historial de costos
            costo = self.calcular_costo(X, y)
            self.historial_costos.append(costo)
            
            if i % 100 == 0:
                print(f"Iteración {i}: Costo = {costo:.6f}")
    
    def entrenar(self, X, y, normalizar=True):
        """
        Entrena el modelo con los datos proporcionados
        
        Parámetros:
        X (array): Variables independientes
        y (array): Variable dependiente
        normalizar (bool): Si normalizar las características
        """
        # Convertir a arrays numpy si es necesario
        X = np.array(X)
        y = np.array(y)
        
        # Normalizar características si se solicita
        if normalizar:
            X_normalizado = self.normalizar_caracteristicas(X)
        else:
            X_normalizado = X
            self.media = np.zeros(X.shape[1])
            self.desviacion = np.ones(X.shape[1])
        
        # Aplicar descenso de gradiente
        self.descenso_gradiente(X_normalizado, y)
        
        print("\nEntrenamiento completado!")
        print(f"Pesos finales: {self.pesos}")
        print(f"Sesgo final: {self.sesgo:.6f}")
        print(f"Costo final: {self.historial_costos[-1]:.6f}")
    
    def predecir(self, X):
        """
        Realiza predicciones con el modelo entrenado
        
        Parámetros:
        X (array): Variables independientes para predecir
        
        Retorna:
        array: Predicciones del modelo
        """
        X = np.array(X)
        X_normalizado = (X - self.media) / self.desviacion
        return self.hipotesis(X_normalizado)
    
    def coeficiente_determinacion(self, X, y):
        """
        Calcula el coeficiente de determinación R²
        
        Parámetros:
        X (array): Variables independientes
        y (array): Valores reales
        
        Retorna:
        float: Coeficiente R²
        """
        y_pred = self.predecir(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def graficar_convergencia(self):
        """Grafica la convergencia del costo durante el entrenamiento"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.iteraciones), self.historial_costos)
        plt.title('Convergencia del Descenso de Gradiente')
        plt.xlabel('Iteraciones')
        plt.ylabel('Costo')
        plt.grid(True)
        plt.show()
    
    def mostrar_ecuacion(self, nombres_variables):
        """
        Muestra la ecuación de regresión final
        
        Parámetros:
        nombres_variables (list): Lista con nombres de las variables independientes
        """
        print("\n" + "="*50)
        print("ECUACIÓN DE REGRESIÓN LINEAL FINAL")
        print("="*50)
        
        ecuacion = f"rendimiento_mani_kgxha = {self.sesgo:.6f}"
        
        for i, (peso, variable) in enumerate(zip(self.pesos, nombres_variables)):
            signo = " + " if peso >= 0 else " - "
            valor_absoluto = abs(peso)
            ecuacion += f"{signo}{valor_absoluto:.6f}*{variable}"
        
        print(ecuacion)
        print("="*50)

def main():
    """
    Función principal que carga los datos y entrena el modelo
    """
    try:
        # Cargar datos desde el archivo CSV
        print("Cargando datos desde data.csv...")
        datos = pd.read_csv('dataset.csv')
        
        # Mostrar información básica del dataset
        print(f"Dataset cargado: {datos.shape[0]} filas, {datos.shape[1]} columnas")
        print("\nPrimeras 5 filas del dataset:")
        print(datos.head())
        
        print("\nNombres de columnas disponibles:")
        print(datos.columns.tolist())
        
        # Verificar que las columnas requeridas existan
        columnas_requeridas = [
            'indice_tiempo', 
            'superficie_sembrada_mani_ha', 
            'superficie_cosechada_mani_ha', 
            'produccion_mani_t', 
            'rendimiento_mani_kgxha'
        ]
        
        for columna in columnas_requeridas:
            if columna not in datos.columns:
                print(f"ERROR: La columna '{columna}' no existe en el dataset")
                print(f"Columnas disponibles: {datos.columns.tolist()}")
                return
        
        # Separar variables independientes y dependiente
        print("\nSeparando variables...")
        X = datos[['indice_tiempo', 'superficie_sembrada_mani_ha', 
                  'superficie_cosechada_mani_ha', 'produccion_mani_t']]
        y = datos['rendimiento_mani_kgxha']
        
        # Mostrar estadísticas descriptivas
        print("\nEstadísticas descriptivas de las variables independientes:")
        print(X.describe())
        
        print("\nEstadísticas descriptivas de la variable dependiente:")
        print(f"Rendimiento_mani_kgxha - Media: {y.mean():.2f}, Desviación: {y.std():.2f}")
        
        # Crear y entrenar el modelo
        print("\nIniciando entrenamiento del modelo...")
        modelo = RegresionLinealMultivariable(tasa_aprendizaje=0.01, iteraciones=1000)
        modelo.entrenar(X, y)
        
        # Mostrar métricas de evaluación
        r2 = modelo.coeficiente_determinacion(X, y)
        print(f"\nMÉTRICAS DE EVALUACIÓN:")
        print(f"Coeficiente de determinación R²: {r2:.4f}")
        
        # Mostrar la ecuación final
        nombres_variables = [
            'indice_tiempo', 
            'superficie_sembrada_mani_ha', 
            'superficie_cosechada_mani_ha', 
            'produccion_mani_t'
        ]
        modelo.mostrar_ecuacion(nombres_variables)
        
        # Graficar convergencia
        print("\nGenerando gráfica de convergencia...")
        modelo.graficar_convergencia()
        
        # Hacer algunas predicciones de ejemplo con los primeros datos
        print("\nPREDICCIONES DE EJEMPLO:")
        print("Primeras 5 observaciones reales vs predichas:")
        X_ejemplo = X.head().values
        y_real = y.head().values
        y_pred = modelo.predecir(X_ejemplo)
        
        for i in range(len(X_ejemplo)):
            print(f"Obs {i+1}: Real = {y_real[i]:.2f}, Predicho = {y_pred[i]:.2f}, Error = {abs(y_real[i]-y_pred[i]):.2f}")
        
        return modelo, X, y
        
    except FileNotFoundError:
        print("ERROR: No se encontró el archivo 'data.csv' en el directorio actual")
        print("Asegúrate de que el archivo esté en la misma carpeta que este script")
    except Exception as e:
        print(f"ERROR: Ocurrió un problema al procesar los datos: {str(e)}")
        print("Verifica que el archivo CSV tenga el formato correcto")

if __name__ == "__main__":
    modelo, X, y = main()