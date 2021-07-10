#Modelo de agrupamientos con clusters
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


sns.set()

# Carga de datos
data = pd.read_csv('3.01. Country clusters.csv')
print(data)

data_mapped = data.copy()

data_mapped['Language'] = data_mapped['Language'].map({'English': 0, 'French': 1, 'German': 2})

print(data_mapped)

# Se utilizaran unicamente Idioma
print("\nPrint x: ")
# data.iloc[<array of rows>, <array of columns>]
x = data_mapped.iloc[: ,3:4]
x2 = data_mapped.iloc[: ,1:4]
print(x)

# Se procede al agrupamiento
kmeans = KMeans(3) # El 3 es el numero de agrupamientos que se desean producir (3 idiomas)
kmeans2 = KMeans(2)
kmeans.fit(x) # Se aplica un agrupamiento KMedias con tres grupos a los datos de entrada de x
kmeans2.fit(x2)
print(kmeans.fit(x))
print(kmeans2.fit(x2))

clusters_identif = kmeans.fit_predict(x)
clusters_identif2 = kmeans2.fit_predict(x2)
print(clusters_identif) # El resultado es una matriz que contiene los agrupamientos previstos
print(clusters_identif2)

data_con_clusters = data.copy()
data_con_clusters['Cluster'] = clusters_identif
print(data_con_clusters)

data_con_clusters2 = data.copy()
data_con_clusters2['Cluster'] = clusters_identif2
print(data_con_clusters2)

# Graficando, el eje horizontal sera latitud y el eje vertical sera longitud
plt.scatter(data_con_clusters['Longitude'],data_con_clusters['Latitude'],c=data_con_clusters['Cluster'],cmap='rainbow')
plt.xlim(-150,150)
plt.ylim(-90,90)
plt.show()

plt.scatter(data_con_clusters2['Longitude'],data_con_clusters2['Latitude'],c=data_con_clusters2['Cluster'],cmap='rainbow')
plt.xlim(-150,150)
plt.ylim(-90,90)
plt.show()
