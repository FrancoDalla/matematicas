import pandas as pd

"""
Calcular

Sxy = Σ(xi*yi) - (Σxi * Σyi)/n
"""
cadena_x  = "indice_tiempo"
cadena_y = "rendimiento_mani_kgxha"

# Cargar el dataset
data = pd.read_csv('dataset.csv')

# Definir las variables
x = data[cadena_x].values  # Variable independiente
y = data[cadena_y].values       # Variable dependiente

def calcular_sxy(x, y):

    n = len(x)
    
    sum_xy = sum(x[i] * y[i] for i in range(n))
    
    # Calcular Σxi y Σyi
    sum_x = sum(x)
    sum_y = sum(y)
    
    sxy = sum_xy - (sum_x * sum_y) / n
    
    return sxy, sum_xy, sum_x, sum_y, n

# Calcular Sxy
sxy, sum_xy, sum_x, sum_y, n = calcular_sxy(x, y)

# Mostrar resultados
print("FÓRMULA: Sxy = Σ(xi*yi) - (Σxi * Σyi)/n")
print(f"Calculo para x {cadena_x} e y {cadena_y}")
print("=" * 50)
print(f"sxy = {sxy}")
