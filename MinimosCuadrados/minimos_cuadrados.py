import numpy as np

import pandas as pd

# se lee el archivo que contiene todo nuestro dataset
df = pd.read_csv("../dataset.csv")

# en el eje X nos quedaran las varialbles independientes 
# en el eje Y nos quedara nuestra variable dependiente 

X = df[["indice_tiempo", "superficie_sembrada_mani_ha", "superficie_cosechada_mani_ha", "produccion_mani_t"]].values
y = df["rendimiento_mani_kgxha"].values

# agregar columna de 1s para el termino independiente b0 
# es decir lo transforma en una matriz 
X_design = np.column_stack((np.ones(X.shape[0]), X))


# calcula la inversa de la matriz generada 
# X_design.T @ X_design calcula X' * X
# linalg.inv calcula la inversa  (X' * X)^−1
# X_design.T @ y X' * Y
# luego multiplica todo (X' * X)^−1 X * Y
b = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y

# Mostramos los resultados 
print("\nCoeficientes del modelo:")
print(f"b0 (rendimiento): {b[0]}")   
print(f"b1 (Año): {b[1]}")
print(f"b2 (SuperficieSembrada): {b[2]}")
print(f"b3 (SuperficieCosechada): {b[3]}")
print(f"b4 (Producción): {b[4]}")

# predicciones / estimacion del modelo 
# Multiplica la matriz X (con la columna de 1s) por el vector de coeficientes b
# esto nos sirve para conseguir los rendimientos del modelo 
y_pred = X_design @ b
print("\nPredicciones del modelo:", y_pred)

# R^2 (bondad del ajuste )
# R^2 mide que tan bien el modelo explica la variabilidad del rendimiento
# Cuanto mas cerca este de 1, mejor ajusta el modelo

ss_total = np.sum((y - np.mean(y))**2)  # Suma total de cuadrados
ss_res = np.sum((y - y_pred)**2)        # Suma de residuos al cuadrado
r2 = 1 - (ss_res / ss_total)
print(f"\nR² del modelo: {r2:.4f}")