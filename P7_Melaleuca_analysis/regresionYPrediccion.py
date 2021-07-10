
from statsmodels.stats.diagnostic import het_white

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import seaborn as sns

sns.set()

data = pd.read_csv("DatosMelaleucaProductosCorrected.csv")

data.drop(data.columns[0], axis=1, inplace=True)

print(data.describe())

y = data['Product_Points']
x1 = np.log(data['Product_Price'])

plt.scatter(x1, y)
plt.xlabel('Product_Price (Log)', fontsize=20)
plt.ylabel('Product_Points', fontsize=20)
plt.show()

x = sm.add_constant(x1)
print(x)

resultados = sm.OLS(y, x).fit()

print(resultados.summary())

'''plt.scatter(x1, y)
yhat = 1.5139 + 0.0181*x1
fig = plt.plot(x1, yhat, lw=4, c='orange', label='linea de regresion')
plt.xlabel('Product_Price', fontsize=20)
plt.ylabel('Product_Points', fontsize=20)
plt.show()
'''

# Buscar heteroscedasticity por medio del test de White:
white_test = het_white(resultados.resid,  resultados.model.exog)

labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']

test_results = dict(zip(labels, white_test))

print(test_results)

p = test_results.get('F-Test p-value')

print(p)

if p < 0.05:

    print("The model is heteroskedastic")

'''
while True:

    xp = int(input("\nPrecio para predecir puntos:"))

    print(1.5139 + 0.0181*xp)

    d = input("Desea predecir otro valor? (si/no)")

    if d == "no" or d == "No" or d == "NO":
        break
'''
