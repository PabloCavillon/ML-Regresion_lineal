import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

data = {'n_equipos_afectados': x.flatten(), 'coste': y.flatten()} 
df = pd.DataFrame(data)

df['n_equipos_afectados'] = df['n_equipos_afectados'] * 1000
df['n_equipos_afectados'] = df['n_equipos_afectados'].astype('int')
df['coste'] = df['coste'] * 10000
df['coste'] = df['coste'].astype('int')
  
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste de incidente")

lin_reg = LinearRegression()
lin_reg.fit(df['n_equipos_afectados'].values.reshape(-1, 1), df['coste'].values)
print(lin_reg.intercept_)

print(lin_reg.coef_)

x_min_max = np.array([[df['n_equipos_afectados'].min()], [df['n_equipos_afectados'].max()]])
y_train_pred = lin_reg.predict(x_min_max)

plt.plot(x_min_max, y_train_pred, "g-")
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste de incidente")
#plt.show()

x_new = np.array([[1200]])
coste = lin_reg.predict(x_new)
print("El coste del incidente seria: ", int(coste[0]), "$")

plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.plot(x_min_max, y_train_pred, "g-")
plt.plot(x_new, coste, "rx")
plt.xlabel('Equipos afectados')
plt.ylabel('Coste del incidente')
plt.show()