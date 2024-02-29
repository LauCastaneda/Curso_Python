#Trabajo final
#Laura Castañeda

#Variable: Condición de empleo
#Filtro: Sexo femenino

#Importación de paquetes y funciones
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from sklearn.model_selection import KFold

#Cargamos la base de datos
datos = pd.read_csv("sample_endi_model_10p.txt", sep=";")
print(datos)

##########################LIMPIEZA DE DATOS############################

# Eliminar filas con valores nulos en la columna "dcronica"
datos = datos[~datos["dcronica"].isna()]
variables = ['n_hijos', 'region', 'sexo', 'condicion_empleo', 'tipo_de_piso']

datos.columns

#Observamos lo que hay dentro de la variable de interés
datos.groupby('condicion_empleo').size()

#Filtramos los datos sólo para mujeres considerando su condición de empleo
#mujer_condicion_empleo = datos[(datos['sexo'] == 'Mujer') ]
#print(mujer_condicion_empleo)
datos_mujer_condicion_empleo = datos[datos['sexo'] == 'Mujer']

#Calcular el conteo de mujeres por cada categoría de la variable 'condicion_empleo'
conteo_condicion_empleo = datos_mujer_condicion_empleo['condicion_empleo'].value_counts()
print(conteo_condicion_empleo)

#Eliminar filas con valores no finitos en las columnas especificadas
columnas_con_nulos = ['dcronica', 'region', 'n_hijos', 'tipo_de_piso', 'espacio_lavado', 'categoria_seguridad_alimentaria', 'quintil', 'categoria_cocina', 'categoria_agua', 'serv_hig', 'condicion_empleo']

#Filtrar los datos para las variables seleccionadas y eliminar filas con valores nulos en esas variables
for i in columnas_con_nulos:
    datos_mujer_condicion_empleo = datos_mujer_condicion_empleo[~datos_mujer_condicion_empleo[i].isna()]

conteo_mujer_por_condicion_empleo = datos_mujer_condicion_empleo.groupby(["sexo", "condicion_empleo"]).size()
print("Conteo de mujeres por categoria 'condicion_empleo':")
print(conteo_mujer_por_condicion_empleo)


###################TRANSFORMACIONES DE VARIABLES#########################

# Definir variables categóricas y numéricas
variables_categoricas = ['region', 'sexo', 'tipo_de_piso', 'condicion_empleo']
variables_numericas = ['n_hijos']

#Crear un transformador para estandarizar las variables numéricas
transformador = StandardScaler()
#Crear una copia de los datos originales
datos_escalados = datos_mujer_condicion_empleo.copy()

#Estandarizar las variables numéricas
datos_escalados[variables_numericas] = transformador.fit_transform(datos_escalados[variables_numericas])

#Convertir las variables categóricas en variables dummy
datos_dummies = pd.get_dummies(datos_escalados, columns=variables_categoricas, drop_first=True)

# Seleccionar las variables predictoras (X) y la variable objetivo (y)
X = datos_dummies[['region_2.0', 'region_3.0',
       'tipo_de_piso_Cemento/Ladrillo', 'tipo_de_piso_Tabla/Caña',
       'tipo_de_piso_Tierra', 'condicion_empleo_Empleada',
       'condicion_empleo_Inactiva']]

y = datos_dummies["dcronica"]

# Definir los pesos asociados a cada observación
weights = datos_dummies['fexp_nino']

#####SEPARACIÓN DE MUESTRAS EN ENTRENAMIENTO (train) Y PRUEBA (test)#####

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Asegurar que todas las variables sean numéricas
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Convertir las variables a tipo entero
variables = X_train.columns
for i in variables:
    X_train[i] = X_train[i].astype(int)
    X_test[i] = X_test[i].astype(int)
y_train = y_train.astype(int)

##############################AJUSTE DEL MODELO#########################

modelo = sm.Logit(y_train, X_train)
result = modelo.fit()
print(result.summary())

#Extraemos los coeficientes y los almacenamos en un DataFrame
coeficientes = result.params
df_coeficientes = pd.DataFrame(coeficientes).reset_index()
df_coeficientes.columns = ['Variable', 'Coeficiente']

#Creamos una tabla pivote para una mejor visualización
df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
df_pivot.reset_index(drop=True, inplace=True)

# Realizamos predicciones en el conjunto de prueba
predictions = result.predict(X_test)
# Convertimos las probabilidades en clases binarias
predictions_class = (predictions > 0.5).astype(int)
# Comparamos las predicciones con los valores reales
np.mean(predictions_class == y_test)

predictions = result.predict(X_train)
# Convertimos las probabilidades en clases binarias
predictions_class = (predictions > 0.5).astype(int)
# Comparamos las predicciones con los valores reales
np.mean(predictions_class == y_train)

#¿Cuál es el valor del parámetro asociado a la variable 
#clave si ejecutamos el modelo solo con el conjunto de entrenamiento 
#y predecimos con el mismo conjunto de entrenamiento?

##Al ejecutar el modelo solo con el conjunto de entrenamiento 
#y predecir con el mismo conjunto de entrenamiento, se puede examinar 
#el coeficiente correspondiente en el resumen del modelo de regresión logística.

#Interpretación

#El coeficiente estimado de la variable condicion_empleo_Empleada es negativo, lo cual indica una relación negativa
#El coeficiente estimado de la variable condicion_empleo_Inactiva es positivo, lo cual indica una relación positiva
#El p-valor de la variable condicion_empleo_Empleada es significativa, sin embargo, la variable condicion_empleo_Inactiva no lo es.

############################VALIDACIÓN CRUZADA##########################

# 100 folds:
kf = KFold(n_splits=50)
accuracy_scores = []
df_params = pd.DataFrame()

for train_index, test_index in kf.split(X_train):

    # aleatorizamos los folds en las partes necesarias:
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    weights_train_fold, weights_test_fold = weights_train.iloc[train_index], weights_train.iloc[test_index]
    
    # Ajustamos un modelo de regresión logística en el pliegue de entrenamiento
    log_reg = sm.Logit(y_train_fold, X_train_fold)
    result_reg = log_reg.fit()
    
    # Extraemos los coeficientes y los organizamos en un DataFrame
    coeficientes = result_reg.params
    df_coeficientes = pd.DataFrame(coeficientes).reset_index()
    df_coeficientes.columns = ['Variable', 'Coeficiente']
    df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
    df_pivot.reset_index(drop=True, inplace=True)
    
    # Realizamos predicciones en el pliegue de prueba
    predictions = result_reg.predict(X_test_fold)
    predictions = (predictions >= 0.5).astype(int)
    
    # Calculamos la precisión del modelo en el pliegue de prueba
    accuracy = accuracy_score(y_test_fold, predictions)
    accuracy_scores.append(accuracy)
    
    # Concatenamos los coeficientes estimados en cada pliegue en un DataFrame
    df_params = pd.concat([df_params, df_pivot], ignore_index=True)

print(f"Precisión promedio de validación cruzada: {np.mean(accuracy_scores)}")


##############VALIDACION CRUZADA: PRECISIÓN DEL MODELO#################

# Calcular la precisión promedio
precision_promedio = np.mean(accuracy_scores)
print(precision_promedio)

#Interpretación
##Cuando se utiliza el conjunto de datos filtrado, la precisión promedio del modelo aumenta a 0.7535833 en comparación con el valor anterior. 
#Por el lado de la distribución de los coeficientes beta en comparación al ejercicio previo, 
# se necesitaría información adicional sobre los valores específicos de los coeficientes para 
#los dos escenarios si se busca determinar si existe un aumento o disminución en cuanto a la distribución y la cantidad de este cambio.

plt.hist(accuracy_scores, bins=30, edgecolor='black')

# Añadir una línea vertical en la precisión promedio
plt.axvline(precision_promedio, color='red', linestyle='dashed', linewidth=2)
# Añadir un texto que muestre la precisión promedio
plt.text(precision_promedio-0.1, plt.ylim()[1]-0.1, f'Precisión promedio: {precision_promedio:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))
plt.title('Histograma de Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frecuencia')
# Ajustar los márgenes
plt.tight_layout()
# Mostrar el histograma
plt.show()

#Se observa que la precisión promedio en este caso es mayor con 0.75

#VALIDACIÓN CRUZADA: EL COMPORTAMIENTO DEL PARÁMETRO ASOCIADO A CONDICION EMPLEO

print(df_params.columns)

#Vamos a crear un histograma para visualizar la distribución de los coeficientes estimados para la variable “n_hijos”:
plt.hist(df_params["condicion_empleo_Empleada"], bins=30, edgecolor='black')
# Añadir una línea vertical en la media de los coeficientes
plt.axvline(np.mean(df_params["condicion_empleo_Empleada"]), color='red', linestyle='dashed', linewidth=2)
# Añadir un texto que muestre la media de los coeficientes
plt.text(np.mean(df_params["condicion_empleo_Empleada"])-0.1, plt.ylim()[1]-0.1, f'Media de los coeficientes: {np.mean(df_params["condicion_empleo_Empleada"]):.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))
plt.title('Histograma de Beta (condicion_empleo_Empleada)')
plt.xlabel('Valor del parámetro')
plt.ylabel('Frecuencia')
# Ajustar los márgenes
plt.tight_layout()
# Mostrar el histograma
plt.show()


#Vamos a crear un histograma para visualizar la distribución de los coeficientes estimados para la variable “n_hijos”:
plt.hist(df_params["condicion_empleo_Inactiva"], bins=30, edgecolor='black')
# Añadir una línea vertical en la media de los coeficientes
plt.axvline(np.mean(df_params["condicion_empleo_Inactiva"]), color='red', linestyle='dashed', linewidth=2)
# Añadir un texto que muestre la media de los coeficientes
plt.text(np.mean(df_params["condicion_empleo_Inactiva"])-0.1, plt.ylim()[1]-0.1, f'Media de los coeficientes: {np.mean(df_params["condicion_empleo_Inactiva"]):.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))
plt.title('Histograma de Beta (condicion_empleo_Inactiva)')
plt.xlabel('Valor del parámetro')
plt.ylabel('Frecuencia')
# Ajustar los márgenes
plt.tight_layout()
# Mostrar el histograma
plt.show()

#Finalmente podemos observar que el coeficiente beta asociado a la variable condicion_empleo_Empleada es negativo (-0.93)
#y para la variable condicion_empleo_Inactiva el coeficiente beta asociado es positivo (0.5) y mayor que el anterior