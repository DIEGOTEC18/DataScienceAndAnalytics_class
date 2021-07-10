# Segmentacion de mercado
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.cluster import KMeans

# Archivo de entrada
# datos de una tienda, cada renglón es un cliente consu nivel de satisfacción y lealtad de marca
data = pd.read_csv('3.12. Example.csv')
print(data)

# Graficar los datos, se observará que existen dos agrupamientos
plt.scatter(data['Satisfaction'],data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()

# Mejorando lo hallado hasta el momento, se hace lo siguiente:
x = data.copy()
kmeans = KMeans(2)
kmeans.fit(x)
print(kmeans.fit(x))

# Observar el efecto del metodo predict(x)
clusters = x.copy()
clusters['cluster_pred']=kmeans.fit_predict(x)

# Graficar ahora los datos, se observará una línea de corte en el nivel de satisfacción
# a la izquierda de Satisfaction = 6 es una agrupación y a la derecha es el otro grupo
# En este sentido, el algoritmo pudo haber considerado únicamente como característica Satisfaction

plt.scatter(clusters['Satisfaction'],clusters['Loyalty'],c=clusters['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()

#Para evitar que el efecto de una variable sobrepase a la otra, se recurre a la estandarización
#Los valores de satisfacción son mayores que los de loyalty por lo que KMeans
#descartó la lealtad como característica


#Estandarizar variable Satisfaction
from sklearn import preprocessing
x_scaled = preprocessing.scale(x) #scale(x) es un método el cual escala cada variable separadamente
print(x_scaled)

#Para conocer el número de agrupamientos necesarios, se utiliza el método del codo
#Manejaremos el método del codo para hasta 9 agrupamientos (decisión arbitraria)
wcss=[]
for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
print(wcss)

#Graficar el codo (se observa importante 2, 3, 4 y 5 agrupamientos)
plt.plot(range(1,10),wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

#Explorar la solución optima en la selección de número de clústers con datos estandarizados
kmeans_new = KMeans(2) #Este parametro se cambiará a 3, 4 y 5
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new['cluster_pred']= kmeans_new.fit_predict(x_scaled)
print(clusters_new)
plt.show()

#Graficar (aquí se observa que ambas características son tomadas en cuenta)
plt.scatter(clusters_new['Satisfaction'],clusters_new['Loyalty'],c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()



