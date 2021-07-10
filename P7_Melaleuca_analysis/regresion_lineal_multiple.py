
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm  # Ejecucion de regresiones
import seaborn as sns


sns.set()

data = pd.read_csv('DatosMelaleucaProductosV2.csv')

data.drop(data.columns[0], axis=1, inplace=True)

print(data)  # imprime el archivo

print("\nEstadística Descriptiva:")
print(data.describe())

# Definir la variable dependiente y las variable independientes
y = data['Product_Points']
x1 = np.log(data['Product_Price'])
x2 = np.log(data['Preferent_Price'])
xn = np.log(data[['Product_Price', 'Preferent_Price']])

# Gráficas de dispersión para demostrar linealidad:
plt.scatter(x1, y)
plt.xlabel('Product_Price (Log)', fontsize=20)
plt.ylabel('Product_Points', fontsize=20)
plt.show()

plt.scatter(x2, y)
plt.xlabel('Preferent_Price (Log)', fontsize=20)
plt.ylabel('Product_Points', fontsize=20)
plt.show()

# Se requiere el termino constante B0 en donde X0 = 1, statsmodel usa el metodo
x = sm.add_constant(xn)


# Declarar una variable resultados que contendra la salida de la regresión de
# minimos cuadrados ordinarios
resultados = sm.OLS(y,x).fit()

#Se usa el metodo FIT que aplica una tecnica de estimacion especifica para que
#el modelo ajuste
print(resultados.summary())

# Buscar heteroscedasticity por medio del test de White y el test de Breusch-Pagan:
white_test = het_white(resultados.resid,  resultados.model.exog)

# bp_test = het_breuschpagan(resultados.resid, [x1, x2])

labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']

test_results = dict(zip(labels, white_test))

print(test_results)

p = test_results.get('F-Test p-value')

print(p)

if p < 0.05:

    print("The model is heteroskedastic")


vif = pd.DataFrame()
vif["variables"] = data.columns
vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

print()
print(vif)
