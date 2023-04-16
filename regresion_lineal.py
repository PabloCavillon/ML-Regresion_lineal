import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

#Generacion de datos para el entrenamiento
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

#Armado del diccionario con los datos de entrenamiento
data = {'n_equipos_afectados': x.flatten(), 'coste': y.flatten()} 
df = pd.DataFrame(data)

#Escalado de los datos del diccionario
df['n_equipos_afectados'] = df['n_equipos_afectados'] * 1000
df['n_equipos_afectados'] = df['n_equipos_afectados'].astype('int')
df['coste'] = df['coste'] * 10000
df['coste'] = df['coste'].astype('int')
  
#Grafico de los pares (X, Y)
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste de incidente")
#plt.show()

#Implementacion y entrenamiento del algoritmo de regresion lineal
lin_reg = LinearRegression()
lin_reg.fit(df['n_equipos_afectados'].values.reshape(-1, 1), df['coste'].values)

print(lin_reg.intercept_)
print(lin_reg.coef_)

#Se toma el valor maximo y minimo para poder predecirlos y de esta manera 
#obtener graficamente la funcion hipotesis
x_min_max = np.array([[df['n_equipos_afectados'].min()], [df['n_equipos_afectados'].max()]])
y_train_pred = lin_reg.predict(x_min_max)

#Se grafican los datos en conjunto con la funcion hipotesis
plt.plot(x_min_max, y_train_pred, "g-")
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste de incidente")
#plt.show()

#Se ingresa un valor deseado para poder predecir cuanto valdria 
x_new = np.array([[1200]]) #Se quire saber cuando costaria si 1200 equipos se dañan
coste = lin_reg.predict(x_new)
print("El coste del incidente seria: ", int(coste[0]), "$")

#Se grafica el conjunto de datos, la funcion hipotesis y se marca el valor consultado 
#en par con su prediccion
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.plot(x_min_max, y_train_pred, "g-")
plt.plot(x_new, coste, "rx")
plt.xlabel('Equipos afectados')
plt.ylabel('Coste del incidente')
plt.show()