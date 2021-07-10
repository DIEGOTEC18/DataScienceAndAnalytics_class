#Modelo de agrupamientos con clusters
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


sns.set()

# Carga de datos
data = pd.read_csv('3.01. Country clusters.csv')
print(data)

# Latitud y longitud corresponden a los centros geograficos de los paises, es una
# forma de representar la ubicacion
# Convencion: Norte y este son + mientras que oeste y sur son -

# Graficar los datos
plt.scatter(data['Longitude'],data['Latitude'])
# Define los límites de la gráfica (basado en valores máximos de latitud y longitud):
plt.xlim(-150,150)
plt.ylim(-90,90)
plt.show()

# Se utilizaran unicamente Latitude y Longitude
print("\nPrint x: ")
# data.iloc[<array of rows>, <array of columns>]
x = data.iloc[:,1:3]
print(x)

# Se procede al agrupamiento
kmeans = KMeans(3) # El 2 es el numero de agrupamientos que se desean producir
kmeans.fit(x) # Se aplica un agrupamiento KMedias con dos grupos a los datos de entrada de x
print(kmeans.fit(x))

clusters_identif = kmeans.fit_predict(x)
print(clusters_identif) # El resultado es una matriz que contiene los agrupamientos previstos

data_con_clusters = data.copy()
data_con_clusters['Cluster'] = clusters_identif
print(data_con_clusters)

# Graficando, el eje horizontal sera latitud y el eje vertical sera longitud
plt.scatter(data_con_clusters['Longitude'],data_con_clusters['Latitude'],c=data_con_clusters['Cluster'],cmap='rainbow')
plt.xlim(-150,150)
plt.ylim(-90,90)
plt.show()