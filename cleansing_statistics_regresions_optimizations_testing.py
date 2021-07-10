
from statsmodels.stats.diagnostic import het_white
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import seaborn as sns


def plot_lin_reg(x, y, model, name, title):
    # plot and save:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x, y, alpha=0.5, color='orange')
    fig.suptitle(title)
    fig.tight_layout(pad=2)
    ax.grid(True)

    x_pred = np.linspace(x.min(), x.max(), 50)
    x_predM = sm.add_constant(x_pred)
    y_pred = model.predict(x_predM)
    ax.plot(x_pred, y_pred, '-', color='blue', linewidth=2)

    fig.savefig(name, dpi=150)


sns.set()

data = pd.read_csv("CarDekho_cardata.csv")

# Eliminar filas con valores null:
data = data.dropna()

# Quitar unidades de columnas con datos numéricos:
data['mileage'] = data['mileage'].str.extract('(\d+)', expand=False)
data['mileage'] = data['mileage'].astype(int)

data['engine'] = data['engine'].str.extract('(\d+)', expand=False)
data['engine'] = data['engine'].astype(int)

data['max_power'] = data['max_power'].str.extract('(\d+)', expand=False)
data['max_power'] = data['max_power'].astype(int)

data['torque'] = data['torque'].str.extract('(\d+)', expand=False)
data['torque'] = data['torque'].astype(int)

# Separar la marca del modelo en columnas distintas:
data['manufacturer'] = np.nan
data['model'] = np.nan

data['manufacturer'] = data['name'].str.split(" ", 1, expand=True)
data['model'] = data['name']

data.drop(data.columns[0], axis=1, inplace=True)

print(data.columns)

# Describir los datos:
pd.set_option('display.max_columns', 20)
print(data.describe(include='all'))

# Comprobación de linealidad y variables escogidas:
# Variable dependiente:
price = data['selling_price']

# Variables independientes:
year = data['year']
mileage = data['km_driven']
fuel_efficiency = data['mileage']
engine_size = data['engine']
max_hp = data['max_power']
torque = data['torque']
seats = data['seats']

# Variables categóricas:
seller_type = data['seller_type']
owner = data['owner']
transmission = data['transmission']
manufacturer = data['manufacturer']
model = data['model']
fuel_type = data['fuel']

# Linealidad:
plt.scatter(price, year)  # (x, y)
plt.ylabel('Year of production', fontsize=20)
plt.xlabel('Vehicle Price', fontsize=20)
plt.show()

plt.scatter(price, mileage)  # (x, y)
plt.ylabel('Mileage', fontsize=20)
plt.xlabel('Vehicle Price', fontsize=20)
plt.show()

plt.scatter(price, fuel_efficiency)  # (x, y)
plt.ylabel('Fuel efficiency (km/l)', fontsize=20)
plt.xlabel('Vehicle Price', fontsize=20)
plt.show()

plt.scatter(price, engine_size)  # (x, y)
plt.ylabel('Engine Size (CC)', fontsize=20)
plt.xlabel('Vehicle Price', fontsize=20)
plt.show()

plt.scatter(price, max_hp)  # (x, y)
plt.ylabel('Max Horse Power (bhp)', fontsize=20)
plt.xlabel('Vehicle Price', fontsize=20)
plt.show()

plt.scatter(price, torque)  # (x, y)
plt.ylabel('Max Torque (Nm)', fontsize=20)
plt.xlabel('Vehicle Price', fontsize=20)
plt.show()

plt.scatter(price, seats)  # (x, y)
plt.ylabel('Number of Seats', fontsize=20)
plt.xlabel('Vehicle Price', fontsize=20)
plt.show()

# Regresión Lineal Simple:
y = price

# Price-year:

x1 = sm.add_constant(year)
print(x1)

resultados1 = sm.OLS(y, x1).fit()

print(resultados1.summary())

plot_lin_reg(year, y, resultados1, 'price_year_lin.png', 'Price-Year Simple Linear:')

# Kilometraje-precio:
x2 = sm.add_constant(mileage)
print(x2)

resultados2 = sm.OLS(y, x2).fit()

print(resultados2.summary())

plot_lin_reg(mileage, y, resultados2, 'price_mileage_lin.png', 'Price-Mileage Simple Linear:')

# Caballos de fuerza - precio:
x3 = sm.add_constant(max_hp)
print(x3)

resultados3 = sm.OLS(y, x3).fit()

print(resultados3.summary())

plot_lin_reg(max_hp, y, resultados3, 'price_hp_lin.png', 'Price-HorsePower Simple Linear:')

# Torque-precio:
x4 = sm.add_constant(torque)
print(x4)

resultados4 = sm.OLS(y, x4).fit()

print(resultados4.summary())

plot_lin_reg(torque, y, resultados4, 'price_torque_lin.png', 'Price-Torque Simple Linear:')

# Regresión Lineal Múltiple:
# Obtener variables dummy (numéricas) para las variables categóricas:
fuel_d = pd.get_dummies(fuel_type)
transmission_d = pd.get_dummies(transmission, drop_first=True)
seller_d = pd.get_dummies(seller_type, drop_first=True)
manufacturer_d = pd.get_dummies(manufacturer, drop_first=True)
owner_d = pd.get_dummies(owner, drop_first=True)

fuel_d.drop('CNG', axis=1, inplace=True)  # Se elimina porque hay muy pocos registros de este combustible.
data2 = data.copy()
data3 = data.copy()
data2 = pd.concat([data2, fuel_d, transmission_d, seller_d, manufacturer_d, owner_d], axis=1)
data3 = pd.concat([data3, fuel_d, transmission_d, seller_d, owner_d], axis=1)
print(data2.head())

# Eliminamos todas las columnas que no vamos a usar:
data2.drop(['fuel', 'seller_type', 'transmission', 'mileage', 'seats', 'manufacturer', 'model', 'owner'], axis=1, inplace=True)
data3.drop(['fuel', 'seller_type', 'transmission', 'mileage', 'seats', 'manufacturer', 'model', 'owner'], axis=1, inplace=True)
print(data2.head())
print(data2.columns)

# Usamos sklearn para tratar las variables numéricas y categóricas igual:

np.random.seed(0)  # Para tener una muestra realmente aleatoria.
train, test = train_test_split(data2, test_size=0.3, random_state=100)
train3, test3 = train_test_split(data3, test_size=0.3, random_state=100)

numeric_vars = ['selling_price', 'km_driven', 'max_power', 'engine', 'year']

scaler = StandardScaler()  # Para estandarizar las variables entre su min, media y media y max.
train[numeric_vars] = scaler.fit_transform(train[numeric_vars])
test[numeric_vars] = scaler.transform(test[numeric_vars])

train3[numeric_vars] = scaler.fit_transform(train3[numeric_vars])
test3[numeric_vars] = scaler.transform(test3[numeric_vars])

# Dividimos el dataset:
y_train = train.pop('selling_price')
X_train = train

X_train = X_train.drop('Diesel', axis=1)
X_train = X_train.drop('Petrol', axis=1)
X_train = X_train.drop('Maruti', axis=1)
X_train = X_train.drop('Hyundai', axis=1)
X_train = X_train.drop('torque', axis=1)
X_train = X_train.drop('Mahindra', axis=1)
X_train = X_train.drop('Tata', axis=1)
X_train = X_train.drop('Manual', axis=1)
X_train = X_train.drop('Toyota', axis=1)
X_train = X_train.drop('Honda', axis=1)

y_train3 = train3.pop('selling_price')
X_train3 = train3

# Definimos el modelo:
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Definimos las caracter[isticas y la cantidad de caracter[isticas utilizando RFE (Recursive Feature Elimination):
rfe = RFE(linear_model, 15)  # Aquí se definen la cantidad de características a seleccionar.
rfe = rfe.fit(X_train, y_train)

# Característica seleccionadas por RFE:
print(list(zip(X_train.columns, rfe.support_, rfe.ranking_)))

col = X_train.columns[rfe.support_]
print(col)

X_train_rfe = X_train[col]

X_train_rfe = sm.add_constant(X_train_rfe)

# Realizamos la regresión lineal múltiple:
multi_linear_model = sm.OLS(y_train, X_train_rfe).fit()
print("\nSELECTED---------------------------------------------")
print(multi_linear_model.summary())

# Regresión con todas las variables:
linear_model2 = LinearRegression()
linear_model2.fit(X_train, y_train)

rfe2 = RFE(linear_model, 36)
rfe2 = rfe2.fit(X_train, y_train)

col2 = X_train.columns[rfe2.support_]
print(col2)

X_train_rfe2 = X_train[col2]

X_train_rfe2 = sm.add_constant(X_train_rfe2)

multi_linear_model2 = sm.OLS(y_train, X_train_rfe2).fit()
print("\nSELECTED---------------------------------------------")
print(multi_linear_model2.summary())

# Regresión lineal sin considerar marcas:
linear_model3 = LinearRegression()
linear_model3.fit(X_train3, y_train3)

rfe3 = RFE(linear_model, 16)
rfe3 = rfe3.fit(X_train3, y_train3)

col3 = X_train3.columns[rfe3.support_]
print(col3)

X_train_rfe3 = X_train3[col3]

X_train_rfe3 = sm.add_constant(X_train_rfe3)

multi_linear_model3 = sm.OLS(y_train3, X_train_rfe3).fit()
print(multi_linear_model3.summary())

# Regresión lineal sin marcas (Optimizado):
linear_model4 = LinearRegression()
linear_model4.fit(X_train3, y_train3)

rfe4 = RFE(linear_model, 10)
rfe4 = rfe4.fit(X_train3, y_train3)

col4 = X_train3.columns[rfe4.support_]
print(col4)

X_train_rfe4 = X_train3[col4]

X_train_rfe4 = sm.add_constant(X_train_rfe4)

multi_linear_model4 = sm.OLS(y_train3, X_train_rfe4).fit()
print(multi_linear_model4.summary())

# Buscar heteroscedasticity por medio del test de White:
white_test1 = het_white(multi_linear_model.resid,  multi_linear_model.model.exog)  # Modelo 15 variables.
white_test2 = het_white(multi_linear_model2.resid,  multi_linear_model2.model.exog)  # Modelo 46 variables.

labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']

test_results1 = dict(zip(labels, white_test1))
test_results2 = dict(zip(labels, white_test2))

print(white_test1)
print()
print(white_test2)

p1 = test_results1.get('F-Test p-value')
p2 = test_results2.get('F-Test p-value')

print("p1: " + str(p1))
print("p2: " + str(p2))

if p1 < 0.05:

    print("Model 1 is heteroskedastic")


if p2 < 0.05:

    print("Model 2 is heteroskedastic")

print("\nTest No Multicolinealidad Modelo 1 y 2:")
X_train_new = X_train
vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by="VIF", ascending=False)

print(vif)

# Testing:
y_test = test.pop('selling_price')  # Objetivos
X_test = test  # Características

X_test_predict = X_test[X_train.columns]
X_test_predict1 = X_test[X_train_rfe.drop('const', axis=1).columns]
X_test_predict = sm.add_constant(X_test_predict)
X_test_predict1 = sm.add_constant(X_test_predict1)

y_predict1 = multi_linear_model.predict(X_test_predict1)
y_predict2 = multi_linear_model2.predict(X_test_predict)

results1 = pd.DataFrame({'Actual': y_test, "Predicted": y_predict1})
print(results1.head())

results2 = pd.DataFrame({'Actual': y_test, "Predicted": y_predict2})
print(results2.head())

score1 = r2_score(y_test, y_predict1)
score2 = r2_score(y_test, y_predict2)

print("Score of model 1: " + str(score1))
print("Score of model 2: " + str(score2))

