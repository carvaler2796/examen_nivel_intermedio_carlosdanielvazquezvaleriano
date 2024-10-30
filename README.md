# Examen de Nivel Intermedio

Este repositorio contiene el código del examen de nivel intermedio, que evalúa habilidades en la manipulación de datos, modelado de machine learning, y comprensión de técnicas avanzadas en Python. Los ejercicios hacen uso de `pandas`, `sklearn` y `list comprehensions` para resolver problemas comunes en ciencia de datos e ingeniería de datos.

## Criterios de Evaluación

Las soluciones serán evaluadas en función de los siguientes criterios:

- **Correctitud de las funciones**
- **Eficiencia en el uso de pandas, sklearn y list comprehensions**
- **Calidad del código (buenas prácticas de programación)**
- **Estructura del repositorio y documentación clara**

## Estructura del Repositorio

Este repositorio contiene:

- `examen.py`: Archivo Python con todas las funciones solicitadas.
- `README.md`: Este documento que explica cada ejercicio y cómo ejecutar las pruebas.

## Ejercicios

### Ejercicio 1: Filtrar DataFrame con pandas

#### Función: `filter_dataframe`

- **Descripción**: La función `filter_dataframe` toma un `DataFrame`, el nombre de una columna y un valor de umbral, y devuelve un nuevo `DataFrame` filtrado. El objetivo es mostrar solo las filas en las que los valores de la columna especificada sean mayores que el umbral dado.

- **Instrucciones para realizar las pruebas**:
  - Puedes modificar el contenido del `Dataframe` de `Datos` el cual es una entrada.
  - Puedes modificar el nombre de la `Columna A` el cual es un entrada igual (que en este caso no es muy recomendable para no buggear el código).
  - Puedes cambiar el `Umbral` el cual es una entrada. 
  - Finalmente correr el código y esperar un `Dataframe`de salida ya filtrado mostrando los valores mayores por fila del `Umbral` declarado.


- **Ejemplo de Código**:

  ```python
  #
  #Ejercicio 1
  #

  #Importación de la libería pandas, la cual nos ayudará para la manipulación y el análisis de datos en estructura dataframe
  import pandas as pd

  #definimos la función filter_dataframe, la cual se encarga de filtrar un dataframe en función de los valores de una columna especifica y un umbral declarado
  #es por esto que pide 3 datos, (df: Dataframe que contendrá los datos), (column: la columna a la que le aplicaremos el filtro), (treshold: valor del umbral)
  def filter_dataframe(df, column, threshold):
      
  #El resultado se registra en el nuevo dataframe (filtered_df), para esto filtramos los datos del dataframe primero, luego seleccionamos las filas
  #en las que el valor de la columna especificada es mayor que el valor del umbral, así solo saldrán datos que cumplan la condición
      filtered_df = df[df[column] > threshold] 
      
  #la condición de este tipo de formulas es regresar el dataframe filtrado para poderlo usar en scripts posteriores,
  #donde únicamente se contiene las filas que cumplen con el criterio dado
      return filtered_df

  #Aqui declaramos los dataframe de ejemplo, así probaremos la función, para eso declaramos un diccionario llamado (datos) con dos listas de números de ejemplo,
  #la primera lista "A" es la cantidad de venta de un producto, y la segunda lista "B" el precio pagado según la cantidad de producto comprado
  datos = {'A': [1, 2, 3, 4, 5, 6, 7, 8], 'B': [10, 19.9, 29.8, 39.7, 49.6, 59.5, 69.4, 79.3]}

  #Se convierte el diccionario "datos" en un dataframe y lo guardamos en la variable df, con esto obtenemos la tabla con columnas A y B.
  df = pd.DataFrame(datos)

  #Aquí es donde llamamos a la función "Filter_dataframe" con el dataframe (df), la columna (A) y los datos que pertenezcan a esta columna mayores al umbral
  #que es 4, lo cual se resume a imprimir todas las filas donde el valor de la columna (A) sean mayores que 4.
  #También imprimimos el dataframe original por meros fines visuales de ver como se ve el dataframe original vs el filtrado.
  resultado = filter_dataframe(df, 'A', 4)
  print("DataFrame original:")
  print(df)
  print("\nDataFrame filtrado (A > 4):")
  print(resultado)

### Ejercicio 2: Generar datos para regresión

#### Función: `generate_regression_data`

- **Descripción**: La función `generate_regression_data` simula un conjunto de datos para un problema de regresión, utilizando la librería `Faker` para generar datos aleatorios y combinando características con un valor de ruido añadido. Devuelve un `DataFrame` con las variables independientes y una `Serie` con la variable dependiente.

- **Instrucciones para realizar las pruebas**: 
  - Asegurarse de tener instalada la librería `Faker` además de `pandas`.
  - Se puede modificar el valor de `n_samples` para generar diferentes números de muestras.
  - Esperar como salida un `DataFrame` con dos columnas de características aleatorias (`lista1` y `lista2`) y una `Serie` llamada `Target` que representa la variable dependiente calculada a partir de estas características independientes con un factor de ruido.

- **Ejemplo de Código**:

  ```python
  #
  #Ejercicio 2
  #

  #Tuve que instalar faker ya que no la tenía instalada, a diferencia de pandas, que si.
  !pip install faker

  #Instalación de las librerías requeridas para el ejercicio, en este caso la nueva librería que es Faker para generar datos falsos o aleatorios
  #importamos random para generar valores numéricos aleatorios específicos
  import pandas as pd
  from faker import Faker
  import random

  #Definimos la función para simular conjuntos de datos que serán utilizados en el problema de regresión, requiriendo un entero positivo el cual será:
  #n_samples que será el número de muestras o filas que queremos generar para el conjunto de datos.
  def generate_regression_data(n_samples):
    
  #Creé una instancia de la clase Faker para generar datos falsos
      fake = Faker()
      
  #Inicializo listas vacías para almacenar los datos generados que serán las variables independientes y la lista target para la variable dependiente
      lista1 = [] 
      lista2 = []
      target = []
      
  #Generamos los datos de acuerdo al número de muestras especificado
      for _ in range(n_samples):
          
  #Generamos un valor aleatorio para la primera característica
  #Valores entre 0 y 100
          lista1.append(random.uniform(0, 100))
          
  #Generamos un valor aleatorio para la segunda característica
  #Valores entre 0 y 100       
          lista2.append(random.uniform(0, 100))
          
  #Generamos el valor de la variable dependiente como combinación de las características
  #Agrego un valor de coeficiente de peso para fines prácticos o simulación de un ejemplo de regresión real
  #agregamos el [-1] para acceder al último dato de cada una de las listas que ha sido agregado y que nos permitiría tomar el entero que se asignó a esa fila
  #agrego el random.uniform para añadir ruido aleatorio como simulación de datos reales que no siguen una relación perfecta
          target_value = lista1[-1] * 1.5 + lista2[-1] * 0.5 + random.uniform(-10, 10)
          target.append(target_value)
      
  #Creamos un dataframe con las variables independientes
      df = pd.DataFrame({
          'lista1': lista1,
          'lista2': lista2
      })
      
  #Creamos una serie para la variable dependiente
      target_series = pd.Series(target, name='Target')
      
  #Retornamos el dataframe de características y la serie de la variable dependiente
      return df, target_series

  #Generamos un conjunto de datos de prueba con 10 muestras que es la entrada entera positiva
  df, target = generate_regression_data(10)

  #Imprimimos el dataframe de características o variables independientes
  print("Dataframe de características o variables independientes:")
  print(df)

  #Imprimimos la serie de la variable dependiente
  print("\nSerie de la variable dependiente:")
  print(target)

### Ejercicio 3: Entrenar modelo de regresión múltiple

#### Función: `train_multiple_linear_regression`

- **Descripción**: La función `train_multiple_linear_regression` entrena un modelo de regresión lineal múltiple utilizando un conjunto de datos simulado. La función toma un `DataFrame` con variables independientes y una `Serie` con la variable dependiente y devuelve un modelo entrenado de `LinearRegression`.

- **Instrucciones para realizar las pruebas**:
  - Asegurarse de tener instaladas las librerías `pandas`, `scikit-learn`, y `Faker` para generar datos y realizar la regresión lineal.
  - Se puede modificar el número de muestras (`n_samples`) en la función `generate_regression_data` para generar diferentes conjuntos de datos.
  - Utilizar los datos generados en el ejercicio pasado como `X` (DataFrame de variables independientes) y `y` (Serie de la variable dependiente) para entrenar el modelo.
  - La salida del modelo incluye los coeficientes de cada variable independiente y el intercepto, los cuales representan el peso de cada variable en la predicción de la variable dependiente y el punto de intersección en el eje Y, respectivamente.

- **Ejemplo de Código**:

  ```python
  #
  #Ejercicio 3
  #

  #Primero instalo scikit-learn que en este caso será para realizar el entrenamiento del modelo
  !pip install scikit-learn

  #Llamamos o declaramos la libería nueva de sklearn para la regresión lineal múltiple y las otras librerías del problema 2 para
  #tomar esos datos como los datos simulados
  from sklearn.linear_model import LinearRegression
  import pandas as pd
  from faker import Faker
  import random

  #Definimos la función para simular conjuntos de datos que serán utilizados en el problema de regresión
  #n_samples es un entero positivo que representa el número de muestras que queremos generar
  def generate_regression_data(n_samples):

  #Instancia de la clase Faker para generar datos falsos
      fake = Faker()
      
  #Inicialización de listas vacías para almacenar los datos generados
      lista1 = []  # Primera variable independiente
      lista2 = []  # Segunda variable independiente
      target = []  # Variable dependiente
      
  #Generamos los datos de acuerdo al número de muestras especificado
      for _ in range(n_samples):
          
  #Generación de valores aleatorios para cada característica
          lista1.append(random.uniform(0, 100))  #Valores entre 0 y 100 para lista1
          lista2.append(random.uniform(0, 100))  #Valores entre 0 y 100 para lista2
          
  #Generación del valor de la variable dependiente como una combinación de las características
  #Se asignan pesos y se añade ruido aleatorio para simular datos reales
          target_value = lista1[-1] * 1.5 + lista2[-1] * 0.5 + random.uniform(-10, 10)
          target.append(target_value)
      
  #Creación del DataFrame con las variables independientes
      df = pd.DataFrame({
          'lista1': lista1,
          'lista2': lista2
      })
      
  #Creación de la serie para la variable dependiente
      target_series = pd.Series(target, name='Target')
      
  #Retornamos el dataframe de características y la serie de la variable dependiente
      return df, target_series

  #Generamos un conjunto de datos de prueba con 10 muestras
  df, target = generate_regression_data(10)

  # Imprimimos el dataframe de características
  print("DataFrame de características:")
  print(df)

  # Imprimimos la serie de la variable dependiente
  print("\nSerie de la variable dependiente:")
  print(target)

  #Aquí comienza el código nuevo del problema 3, donde definimos la función "train_multiple_linear_regression"
  def train_multiple_linear_regression(X, y):
     
  #Se crea una instancia del modelo de regresión lineal
      model = LinearRegression()
      
  #Entrenaremos el modelo usando el método ".fit(,)"
      model.fit(X, y)
      
  #Retornamos el modelo entrenado
      return model

  #Llamamos a la función "train_multiple_linear_regression" para entrenar el modelo y usamos "df" como las variables independientes (X) 
  #y "target" como la variable dependiente (y)

  modelo_entrenado = train_multiple_linear_regression(df, target)

  #Imprimimos los coeficientes y el intercepto del modelo, los coeficientes muestran el peso que el modelo asigna a cada variable independiente
  #siendo el intercepto es el punto donde la línea de regresión cruza el eje y, tal cual lo que compone una ecuación lineal.
  print("\nCoeficientes:", modelo_entrenado.coef_)  # Pesos de las variables independientes (lista1 y lista2)
  print("Intersección:", modelo_entrenado.intercept_)  # Valor del intercepto (intersección con el eje Y)

### Ejercicio 4: List comprehension anidado

#### Función: `flatten_list`

- **Descripción**: La función `flatten_list` toma una lista de listas (estructura anidada) y la convierte en una lista plana utilizando una list comprehension anidada, lo que permite simplificar la estructura y obtener todos los elementos en una sola lista.

- **Instrucciones para realizar las pruebas**:
  - Se puede modificar el contenido de `nested_list`, que es la lista de listas de entrada, para probar diferentes conjuntos de datos.
  - El resultado será una lista plana que contiene todos los elementos de las sublistas en el mismo orden en que aparecen originalmente.
  - La entrada es una lista de listas, y la salida es una lista aplanada con todos los elementos en una sola dimensión.

- **Ejemplo de Código**:

  ```python
  #
  #Ejercicio 4
  #

  #Para este ejercio definimos la función flatten_list, donde toma una lista de listas y la convierte en una lista plana
  def flatten_list(nested_list):
   
  #Utilizamos una list comprehension anidada para aplanar la lista
  #Primero recorremos cada sublista en nested_list, luego cada elemento en la sublista
      flat_list = [item for sublist in nested_list for item in sublist]
      
  #Retornamos la lista plana
      return flat_list

  #Definí la lista para hacer la función
  nested_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

  #Llamamos a la función flatten_list y mostramos el resultado impreso
  result = flatten_list(nested_list)
  print("Lista aplanada:", result)

### Ejercicio 5: Agrupar y agregar con pandas

#### Función: `group_and_aggregate`

- **Descripción**: La función `group_and_aggregate` toma un `DataFrame`, agrupa los datos por una columna especificada y calcula la media de otra columna dentro de cada grupo. Esto es útil para obtener promedios categorizados en función de una columna de agrupación.

- **Instrucciones para realizar las pruebas**:
  - Se puede modificar el contenido del `DataFrame` de ejemplo `datos` para probar diferentes conjuntos de datos.
  - Se pueden cambiar los valores de `group_column` y `agg_column` según las columnas que deseas agrupar y calcular la media.
  - La entrada es un `DataFrame` con columnas específicas para agrupar y calcular, y la salida es un `DataFrame` donde cada grupo tiene la media de la columna agregada.

- **Ejemplo de Código**:

  ```python
  #
  #Ejercicio 5
  #

  #Importé pandas para trabajar con dataframes
  import pandas as pd

  #Se define la función group_and_aggregate, esta función agrupa un dataframe (a procesar) por una columna (para agrupar) 
  #y calcula la media de otra columna
  def group_and_aggregate(df, group_column, agg_column):
      
  #Utilizamos el método ".groupby()" para agrupar el DataFrame por la columna especificada en este caso group_column,
  #luego aplicamos ".mean()" en la columna de agregación para obtener el promedio de la nueva columna, 
  #.reset_index() nos ayudará a convertir el índice actual en una columna regular 
      
      grouped_df = df.groupby(group_column)[agg_column].mean().reset_index()
      
  #Retornamos el DataFrame agrupado y con la media calculada
      return grouped_df


  #Aquí creé un DataFrame de ejemplo
  datos = {
      'Categoria': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C'],
      'Precio': [10, 15, 10, 20, 30, 25, 5, 15, 20]
  }
  df = pd.DataFrame(datos)

  #Llamamos a la función "group_and_aggregate" para agrupar por "Categoria" y calcular la media de "Precio"
  resultado = group_and_aggregate(df, 'Categoria', 'Precio')

  #Imprimimos el dataframe resultante y listo
  print("Dataframe agrupado y agregado:")
  print(resultado)

### Ejercicio 6: Modelo de clasificación logística

#### Función: `train_logistic_regression`

- **Descripción**: La función `train_logistic_regression` entrena un modelo de regresión logística utilizando un conjunto de datos con variables independientes (`X`) y una variable dependiente binaria (`y`). Este modelo es útil para tareas de clasificación binaria.

- **Instrucciones para realizar las pruebas**:
  - Asegurarse de tener datos binarios para las variables independientes y la variable dependiente como lo están en el código.
  - Puedes modificar los valores en el `DataFrame` `X` o en la `Serie` `y` para probar diferentes configuraciones de datos, como lo pueden ser operadores lógicos booleanos por ejemplo, pero eso se define al criterio propio.
  - La entrada es un `DataFrame` `X` con las variables independientes y una `Serie` `y` con la variable dependiente binaria. La salida es un modelo entrenado de `LogisticRegression`.

- **Ejemplo de Código**:

  ```python
  #
  #Ejercicio 6
  #

  #Importamos LogisticRegression para construir el modelo que está dentro de la librería de scikit-learn
  from sklearn.linear_model import LogisticRegression

  #Definimos la función train_logistic_regression, esta función entrena un modelo de regresión logística para un conjunto de datos binarios
  def train_logistic_regression(X, y):
   
  #Se crea una instancia del modelo de regresión logística
      model = LogisticRegression()
      
  #Se entrena el modelo usando el conjunto de datos proporcionado
      model.fit(X, y)
      
  #Retornamos el modelo entrenado
      return model

  #Creamos datos de ejemplo
  import pandas as pd
  datos = {
      'columna1': [0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
      'columna2': [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
  }
  target = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]

  #Se convierten los datos en dataframe y en serie el target o variable dependiente
  X = pd.DataFrame(datos)
  y = pd.Series(target)

  #Entrenamos el modelo
  modelo_entrenado = train_logistic_regression(X, y)

  #Se imprime el modelo entrenado, sus coeficientes e intercepto para analizar el resultado
  print("\nModelo entrenado:", modelo_entrenado)
  print("Coeficientes:", modelo_entrenado.coef_)
  print("Intersección:", modelo_entrenado.intercept_)

### Ejercicio 7: Aplicar función a una columna con pandas

#### Función: `apply_function_to_column`

- **Descripción**: La función `apply_function_to_column` permite aplicar una función personalizada a cada valor de una columna específica en un `DataFrame`. En este caso, se utiliza para convertir precios de USD a MXN.

- **Instrucciones para realizar las pruebas**:
  - Se puede modificar los valores en el `DataFrame` `df` o el tipo de conversión en la función `usd_to_mxn` para probar diferentes escenarios.
  - La entrada incluye un `DataFrame`, el nombre de una columna sobre la cual aplicar la función, y la función personalizada que se aplicará a cada elemento de la columna.
  - La salida es el `DataFrame` con la columna modificada según la función aplicada, en este caso ` pesos mexicanos (mxn)`.

- **Ejemplo de Código**:

  ```python
  #
  #Ejercicio 7
  #

  #Importamos nuestra librería para nuestros dataframes
  import pandas as pd

  #Definimos la función apply_function_to_column, con esta función aplicaremos una función personalizada 
  # para modificar cada valor de una columna en un dataframe, en este caso convertiremos de USD a MXN
  def apply_function_to_column(df, column_name, func):
      
  #Usamos .apply(func) para aplicar la función en específico a cada valor en (column_name)
      df[column_name] = df[column_name].apply(func)
      
  #Retornamos el dataframe modificado
      return df

  #Se crea un dataframe para aplicar la función personalizada
  datos = {
      'Nombre': ['Producto A', 'Producto B', 'Producto C'],
      'Precio': [10, 15, 20]
  }

  df = pd.DataFrame(datos)

  #Definimos una función personalizada que convierte el precio de USD a MXN (1USD = 20MXN)
  def usd_to_mxn(precio):
      return precio * 20 

  #Aplicamos la función "usd_to_mxn" a la columna "Precio en USD"
  df_modificado = apply_function_to_column(df, 'Precio', usd_to_mxn)

  #Se imprime el dataframe resultante
  print("Dataframe con la columna modificada (Precio en MXN):")
  print(df_modificado)    

### Ejercicio 8: Comprehensions con condiciones

#### Función: `filter_and_square`

- **Descripción**: La función `filter_and_square` toma una lista de números, filtra aquellos que son mayores que 5, y devuelve una nueva lista con el cuadrado de esos números. Utiliza una list comprehension para realizar el filtrado y el cálculo en una sola línea de código.

- **Instrucciones para realizar las pruebas**:
  - Se pueden modificar los valores de la lista `numeros` para probar cómo la función responde a diferentes conjuntos de datos.
  - Si se cambian los números, igual se puede cambiar el condicionante 5, aunque la instrucción la manda así, para tener resultados distintos en números elevados al cuadrado.
  - Pero en este caso, con el código en su estado actual, la salida es una lista con los números mayores que 5, elevados al cuadrado.

- **Ejemplo de Código**:

  ```python
  #
  #Ejercicio 8
  #

  #Definimos la función filter_and_square, esta función filtrará los números mayores a 5 por la condición del list comprehension
  #y luego elevará al cuadrado esos números
  def filter_and_square(numeros):
      
  #Usamos list comprehension para filtrar y elevar al cuadrado en una sola línea
      result = [x**2 for x in numeros if x > 5]
      
  #Retornamos la lista resultante
      return result

  #Desarrollamos nuestro ejemplo, donde se ha aplicado la condición de números mayores a 5 que sean incluidos
  numeros = [2, 3, 5, 6, 7, 10]
  print("Lista de números originales:", numeros)
  print("Lista de números mayores a 5 elevados al cuadrado:", filter_and_square(numeros))  