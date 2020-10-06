# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Análisis de los datos zoométricos de distintas especies de cangrejos

# ## Dataset Cangrejos
# ### Información de *dataset*
#
# Se cuenta con algunas medidas zoométricas de cangrejos, así como el periodo en el que se dio si nacimiento, con las que se pretende clasificar a dichos cangrejos por especie, así como predecir la cantidad de nacimientos correspondientes a los periodos subsecuentes y sus características como especie.

# ### Información de las variables
#
# Contamos con un *dataset* cuyas variables son:
#
# | Nombre | Tipo de dato | Unidad | Descripción | 
# |------|------|------|------|
# | PERIODO | Entero | -- | Peridodo de naciemiento de una colonia de cangrejos |
# | ANCHO | Real | -- | Medida perpendicular al largo |
# | LARGO | Real | -- | La medida más larga de su cuerpo |
# | GROSOR | Real | -- | Medida perpendicular al ancho y el largo |
# | PESO | Real | -- | Masa total del individuo |

# ## Análisis exploratorio de datos
#
# Inicialmente verifiquemos el número de observaciones y variables del *dataset*.

# + _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../Datasets/CRABS.csv')
df.shape
# -

# Observemos le tipo de dato de cada variable en el *dataframe*.

df.info()

# Revisemos las características de los datos, podemos observar del mínimo que existen valores de **ANCHO**, **LARGO** y **PESO** que son negativos, por lo que deberá limpiarse el *dataset*. También mostramos los percentiles 1% y 99% para  revsar si existen valores extremos.

df.describe(percentiles = [0.01, 0.25, 0.50, 0.75, 0.99])

# Obtengamos el número de individuos por periodo, asumamos que el número de individuos que contienen el mismo valor para periodo es el número de individuos nacidos durante dicho periodo.

print(df['PERIODO'].value_counts())

# ## Gráficas para observar el comportamiento de los datos
# ### Grafica de la cuenta de individuos por periodo de nacimiento
#
# Observamos una cuenta creciente del número de individuos por periodo.

ipp =df.groupby('PERIODO').count().plot(legend = False, grid = True)
ipp.set_ylabel('Número de individuos')

# ### Histogramas
#
# Observemos los histogramas para cada una de las características, específicamente el histograma de ancho y largo hacen notar que deberíamos tener al menos dos especies de cangrejos dado que la distribución es bimodal.

for col in ['ANCHO', 'LARGO', 'GROSOR', 'PESO']:
    df[[col]].hist(bins = 11)

# Al menos en el primer modo, podemos observar que tanto el ancho como el largo y el peso se comportan de forma similar.

df[['ANCHO', 'LARGO', 'PESO']].plot.hist(bins=20, alpha=0.5)

# Podemos revisar si existe una correlación entre las variables y notemos que existe una correlación de 0.765276 entre el ancho y el largo, sin embargo, la correlación con el peso es negativa, esto es algo inesperado puesto que al crecer el tamaño del cangrejo esperaríamos un mayor peso pero como veíamos en la gráfica anterior mietras que el ancho y el largo crecen en un segundo modo, el peso sigue disminuyendo. 

df[["PERIODO","ANCHO", "LARGO", "GROSOR", "PESO"]].corr()

# ### Diagramas de dispersión
#
# Realizemos diagramas de dispersión para distintos pares de características con el fin de notar algún agrupamiento. En efecto se notan distintos grupamientos, el más interesante es el de LARGO con PESO puesto que muestra 3 agrupamientos.

# +
import seaborn as sns
sns.set()

plt.figure()

plt.plot(df['ANCHO'], df['LARGO'], '.')
plt.xlabel('ANCHO')
plt.ylabel('LARGO')
# -

plt.plot(df['LARGO'], df['PESO'], '.')
plt.xlabel('LARGO')
plt.ylabel('PESO')

plt.plot(df['GROSOR'], df['PESO'], '.')
plt.xlabel('GROSOR')
plt.ylabel('PESO')

plt.plot(df['GROSOR'], df['LARGO'], '.')
plt.xlabel('GROSOR')
plt.ylabel('LARGO')

# Estos 3 agrupamientos notados anteriormente nos dan la pauta para realizar la gráfica de dispersión en 3 dimensiones, con el fin de observarlos mejor y en todo caso, notar si nos estamos perdiendo de algún cuarto.

# +
import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode()

trace = go.Scatter3d(
    x = df['LARGO'],
    y = df['PESO'],
    z = df['ANCHO'],
    mode = 'markers',
    marker = {
        'size': 2,
        'opacity': 0.8,
    }
)

data = [trace]
plot_figure = go.Figure(data=data)
plot_figure.update_layout(scene = dict(
    xaxis_title='LARGO',
    yaxis_title='PESO', 
    zaxis_title='ANCHO'))

plotly.offline.iplot(plot_figure)
# -

# ## Ingeniería de variables

# ### Limpieza de datos
#
# Removemos los valores negativos que encontramos en el dataset puesto ne no tienen sentido lógico en una medición, supongamos que son magnitudes válidas y que únicamente el error está en el signo por lo que se puede dejar el valor sin signo con la función valor absoluto.

df = df.abs()

# Removemos los valores extremos del *dataset* tomando como referencia el 1.° y 99.° percentiles.

df = df[((df.ANCHO > 2.7) & (df.LARGO > 0.3) & (df.GROSOR > 3.8) & (df.PESO > 1.7)
        & (df.ANCHO < 32.2) & (df.LARGO < 37.7) & (df.GROSOR < 18) & (df.PESO < 36.1))]

# ## Selección de características
#
# Para generar el vector de características, seleccionamos aquellas que muestran con mayor claridad los agrupamientos.

X = df[['PERIODO','ANCHO', 'LARGO', 'PESO']]
X.head()

# ### Prueba del codo
#
# En general, no es recomendable seleccionar de manera manual el valor de $K$ ya que esto puede llevar a resultados poco confiables. En su lugar se puede recurrir a la llamada prueba del codo "Elbow Test", la cual es una gráfica que muestra para cada valor de $k$ la suma de la distancia cuadrada entre entre cada dato y el centroide del cluster correspondiente. Con base en la gráfica se seleccionan el valor mínimo de $k$ que proporciona la mayor varianza.

# +
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

sse = []
for k in range(1, 20):
    scaled_data = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=k).fit(X)
    sse.append(kmeans.inertia_)
    
plt.plot(range(1, 20), sse, '-o')
plt.xlabel('$k$')
plt.ylabel('Suma de las distancias cuadradas')
# -

# # K-Means
# Utilizamos K-means como primer algoritmo de agrupamiento puesto que no tenemos datos etiquetados (aprendizaje no supervisado). Dado que K Means utiliza la noción de distancia es necesario realizar un escalamiento de las variables para normalizar la varianza.

# +
from sklearn.pipeline import make_pipeline

kmeans = make_pipeline(StandardScaler(), KMeans(n_clusters=3))
kmeans.fit(X)
labels = kmeans.predict(X)

trace = go.Scatter3d(
    x = df['LARGO'],
    y = df['PESO'],
    z = df['ANCHO'],
    mode = 'markers',
    marker = {
        'size': 3,
        'opacity': 0.8,
        'color': labels,
    }
)

data = [trace]
plot_figure = go.Figure(data=data)
plot_figure.update_layout(scene = dict(
    xaxis_title='LARGO',
    yaxis_title='PESO', 
    zaxis_title='ANCHO'))

plotly.offline.iplot(plot_figure)
# -

# ## Evaluación
#
# Debido a que no se conoce a priori la clase a la que pertenece cada individuo, es necesario evaluar el desempeño del clasificador utilizando metricas como el coeficiente de Silueta (Silhoutte Coefficient) y el índice de Calinkski-Harabasz, de conocerse las etiquetas podría realizarse la evaluación con el mismo modelo.
#
# El coeficiente de silueta es un valor entre -1 y 1. Un valor de -1 indica una agrupación incorrecta, un valor de 1 indica una agrupación densa. Valores alrededor de cero indican que existe cierto traslape entre los clústers.

# +
from sklearn import metrics
from sklearn.metrics import pairwise_distances

metrics.silhouette_score(X, labels, metric='euclidean')
# -

# Por otra parte el índice Calinski-Harabasz es la razón de la suma de la dispersión inter-cluster y la dispersión intra-cluster, en este caso la dispersión se define como la suma de las distancias al cuadrado. Cabe resaltar que no hay valores de referencia para esta metrica, por lo cual se suele evaluar el índice para diferentes clasificadores y tomar el que tenga el índice más alto. Podemos determinar dicho índice como sigue:

metrics.calinski_harabasz_score(X, labels)

# ## Modelo de mezcla de Gausianas
#
# Este algoritmo, al igual que K-Means puede ser utilizado para agrupar datos no etiquetados, sin embargo, este modelos tiene algunas ventajas sobre K-Means puesto que permite agrupamientos no esféricos.

# +
from sklearn.mixture import GaussianMixture

gmm = make_pipeline(StandardScaler(), GaussianMixture(n_components=3))
gmm.fit(X)
labels = gmm.predict(X)

trace = go.Scatter3d(
    x = df['LARGO'],
    y = df['PESO'],
    z = df['ANCHO'],
    mode = 'markers',
    marker = {
        'size': 3,
        'opacity': 0.8,
        'color': labels,
    }
)

data = [trace]
plot_figure = go.Figure(data=data)
plot_figure.update_layout(scene = dict(
    xaxis_title='LARGO',
    yaxis_title='PESO', 
    zaxis_title='ANCHO'))

plotly.offline.iplot(plot_figure)
# -

# ## Evaluación
#
# Al igual que con K-means, la evaluación del modelo se hace calculando el coeficiente de silueta y el índice de Calinski-Harabasz. Para el índice de silueta se obtiene el siguiente valor:

metrics.silhouette_score(X, labels, metric='euclidean')

# El valor de índice de Calinski-Harabasz es el siguiente:

metrics.calinski_harabasz_score(X, labels)

# Como se observa, los mejores resultados se obtienen usando la mezcla de Gaussianas. 

# ### Etiquetado
#
# A continuación se muestran algunas de las etiquetas asignadas a los datos originales:

X['labels'] = labels
X.head(100)

# A diferencia de los algoritmos de clasificación supervisada, los algoritmos de agrupación no deben ser evaluados usando medidas como el número de errores, o la precisión. En su lugar se prefiere utilizar metricas que midan la separación entre clases, la similitud entre datos que pertenecen a la misma clase, o la diferencia entre datos que pertenecen a clases distintan. Cuando se conoce a priori las clases o etiquetas de cada dato (*ground truth*) es posible utilizar medidas como ARI (*Adjusted Rand Index*) o la información mutua para evaluar la clasificación. Si no se conocen las etiquetas de cada dato entonces se puede evaluar el desempeño utilizando metricas como el coeficiente de silueta (*Silhoutte Coefficient*) y el índice de Calinkski-Harabasz.
# A continuación se muestra la cuenta de individuos que pertenecen a cada especie identificada.

X['labels'].value_counts()

# A la etiqueta 0 le corresponde la especie A, a 1, la especie B y a 2, la especie C. Ahora que los datos se encuentran etiquetados, la siguiente gráfica muestra la cuenta de nacimientos por cada especie. 

# +
bpp = [[], [], []]
for i in range(1, 7):
    period = X[X['PERIODO'] == i].groupby(['labels'])['PERIODO'].count()
    bpp[0].append(period[0])
    bpp[1].append(period[1])
    bpp[2].append(period[2])

plt.figure()

plt.title('Nacimientos por especie')
plt.xlabel('Periodo')
plt.ylabel('Número de nacimientos')

plt.plot(bpp[0], '-o', label = 'Especie A')
plt.plot(bpp[1], '-o', label = 'Especie B')
plt.plot(bpp[2], '-o', label = 'Especie C')

plt.legend()
plt.show()
# -

# Esta es la tabla de datos perteneciente a la gráfica anterior, muestra el número de nacimientos de cada especie por periodo.

X_bpp = pd.DataFrame(np.array(bpp).T, columns=['SA', 'SB', 'SC'])
X_bpp.index = np.arange(1, 7)
X_bpp

# ## Predicción
#
# Para predecir los futuros nacimientos por especie, utilizemos el método de regresión, usemos el método de regresión lineal como primera aproximación.
#
# ### Regresión lineal

# +
from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=True)

lr.fit(np.array(X_bpp.index).reshape(-1, 1), X_bpp['SA'])

X_bpp_test = np.arange(1, 13)
y_bpp_test = lr.predict(X_bpp_test[:, np.newaxis])

plt.scatter(X_bpp.index, X_bpp['SA'], label='Datos reales')
plt.plot(X_bpp_test, y_bpp_test, 'r.-', label='Predicción')

plt.title('Especie A')
plt.xlabel('Periodo')
plt.ylabel('Número de nacimientos')
plt.legend(loc='best')
# -

# Los valores de *m* y *b* para la recta son los mostrados a continuación, respectivamente.

model = np.polyfit(X_bpp.index, X_bpp['SA'], 1)
model


# Por ejemplo, el número de nacimientos esperados para el periodo 7 es de 172 individuos.

predict = np.poly1d(model)
nacimientos = 7
int(predict(nacimientos))

x_lin_reg = range(0, 13)
y_lin_reg = predict(x_lin_reg)
plt.scatter(X_bpp.index, X_bpp['SA'])
plt.plot(x_lin_reg, y_lin_reg, c = 'r')

# Podemos calcular el coeficiente de determinación $R^2$ para esta regresión, el cual nos da la proporción de la varianza en la variable dependiente que puede predecirse mediante la variable independiente. Se muestra a continuación.

from sklearn.metrics import r2_score
r2_score(X_bpp['SA'], predict(X_bpp.index))

# ### Regresión exponencial 
#
# Una mejor forma de predecir el número de nacimientos es asumir que la población de cangrejos crece exponencialmente. Dado que no se conoce el tamaño de la población no es posible conocer la tasa de crecimiento *per capita*. No obstante, se puede asumir que el número de nacimientos también se comporta exponencialmente.

# +
from scipy.optimize import curve_fit

def f(x, a, b):
    return a * b ** x

popt, pcov = curve_fit(f, X_bpp.index, X_bpp['SA'])

plt.scatter(X_bpp.index, X_bpp['SA'], label='Datos reales')
plt.plot(X_bpp_test, f(X_bpp_test, *popt), 'r.-', label='Predicción')
plt.title('Especie A')
plt.xlabel('Periodo')
plt.ylabel('Número de nacimientos')
plt.legend(loc='best')

plt.show()
# -

# El error asociado a los parámetros se calcula como sigue:

perr = np.sqrt(np.diag(pcov))
perr

# ## Características
#
# Para determinar las características de una especie basta con agrupar los datos de acuerdo a las etiquetas obtenidas en la clasificación:

X_A = X[X['labels'] == 0]
X_A.describe()
