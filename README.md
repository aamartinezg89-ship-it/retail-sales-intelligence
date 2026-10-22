# Inteligencia de ventas

**End-to-End Ciencia de datos | Econometria | Machine Learning**

## 1) Descripcion general del proyecto

El proyecto desarrolla un marco analitico de e2e para determinar si las variaciones
que existen enntre diferentes tiendas pueden ser explicados por factores asociados, como
categorias de producto, regiones geograficas y condiciones comerciales.

La solución integra un analisis exploratorio de datos, modelacion estadistica,
machine learning, modelos jerarquicos, validacion temporal y la interpretacion modelistica.

**Objetivo central**: Explicar las diferencias en ventas mediante la identificacion
de factores asociados comerciales y estructurales.

## 2) Problema de negocio

Las ventas al por menor varian segun el formato de la tienda, categoria de productos, regiones,
estrucutra socieconomica y calendario de eventos.

La dirección del proyecto sigue las siguientes pregunas:

* ¿Qué factores están estadisticamente asociados con las diferencias en las ventas?
* ¿Cuanta variación se mantiene entre las tiendas, despues de considerar las caracteristicas comerciales observdas?
* ¿Que variables contribuyen en mayor medida al rendimiento productivo?

El objtivo es el análisis explicativo con respealdo de la validación predictiva, en lugar de tener solo el pronostico.

## 3) Estructura analitica y conjunto de datos

El conjunto de datos esta organizado como un panel de ventas al día

| Componentes              | Descripcion               |
|--------------------------|---------------------------|
| Observaciones            | 203, 958                  |
| Tiendas                  | 80                        |
| Categorias deproducto    | 6                         |  
| Perido                   | Enero 2023 - Febrero 2024 |
| Nivel observacional      | Tienda - Categoria - Día  |
| Variable objetivo        | amount_total              |
| Periodo de entrenamiento | 2023                      |
| Periodo de prueba        | Enero-Febrero 2024        |

El proyecto integra informacion transaccional, caracteristicas de tiendas y eventos calendarizados

## 4) Ingenieria de datos y procesamiento previo

El pipeline incluye:

* Ingesta de datos y validación de esquema
* Análisis exploratorio de valores faltantes y calidad de datos
* Imputación de valores faltantes usando medianas mensuales por tienda y categoria
* Integración de transacciones, información de tiendas y datos de calendario
* Codificación "one-hot" de variables categoricas para el Machine Learning
* Separacion temporal del conjunto de datos para el entrenamiento y pruebas

**Las variables categoricas en su primera versión son preservadas en el panel analitico
para la interpretación economica**

## 5) Modelacion econometrica

### Minimos cuadrados ordinarios

Se estima una regresion log-lineal para cuantificar asociaciones entre las ventas y las caracteristicas comericales

La especificacion econometrica considera:

* Formato de tiendas y categorias
* Interacciones entre formato de tiendas y categoria
* Region geografica y nivel socioeconomico
* Caracteristicas de las tiendas
* Promociones y calendario de eventos
* Indicadores estacionales

**Grupos de Referencia**: Bodega para el formato de tienda y Abarrotes para la categoria. El fundamento es que son los
que mayor contribuyen a la venta en 43.60% y 25.15%, respectivamente.

Se usan errores estandar clusterizados a nivel tienda para considerar la dependecia dentro de cada tienda.

### Modelos Jerarquicos 

Se emplea un modelo de efecto fijos con interceptos aleatorios por tienda.

El objetivo es cuantificar la heterogenidad residual entre tiendas al considerar el panel de variables explicativas
seleccionadas.

Se calcula el coeficiente de correlacion entre clases (ICC) que ronda en 27.25%. Lo que implica que
la varianza condicional modelada para el logartimo de las ventas se atribuye a diferencias persistentes
entre tiendas, despues de controlar por formato de tienda, categoria y temporada.

En otras palabras, aunque las tiendas tengas caracteristicas similares, pueden tener ventas diferentes relacionadas con 
caracteristicas propias de cada tienda, que el modelo no logra explicar. Es decir, 27.25% es la variacion que queda por
explicar se debe a diferencias entre tiendas.

**Nota**: El modelo reporta covergencia exitosa.

## 6) Modelos de Machine Learning

Se evaluan 3 modelos predictivos adicionales:

* Regresor de árbol de decisión con GriedSearchCV
* Regresor temporal XGBoost con GriedSearchCV
* Red Neuronal MLP

Todos los modelos usan el mismo periodo de entrenamiento y prueba.

### Rendimiento fuera de la muestra

| Modelo            | MAE     | RMSE    | R^2     |
|-------------------|---------|---------|---------|
| MCO               | 45,400  | 63,661  | 0.6186  |
| Árbol de decision | 52,470  | 81,253  | 0.3788  |
| XGboost           | 55,520  | 97,654  | 0.1026  |
| Red Neuronal MLP  | 56,213  | 95,798  | 0.1364  |

MCO mejora el poder predictivo entre las especificaciones evaluadas sobre el periodo de prueba. 

## 7) Diagnostico del modelo y validacion

El flujo analitico de trabajo considera:

* Validación por retencion temporal
* Evaluación MAE, RMSE y R^2
* Analisis de error por formato de tienda y categoria
* Inferencia de cluster robustos en MCO
* Descomposición de varianza jerarquica.

## 8) Hallazgos clave

* La especificacion log-lineal por MCO superó a los modelos de ML durante el periodo de prueba.
* Las dimensiones comerciales, en especial, categoria y el formato, contribuyen al rendimiento predictivo.
* El modelo jerarquico estimo que el 27.25% de la varianza residual corresponde a la diferencia entre tiendas.

## 9) Limitaciones 

* Los resultados son especificos al periodo de estudio.
* El rendimiento predictivo varia según el segmento comercial
* LOs modelos de intercepto aleatorio se basan en supuestos sobre la heterogeneidad no observada.
* La importancia de las variables no corresponde a causalidad

## 10) Conclusión

| Analisis          | Resultado principal    | Interpretacion                                                  |
|-------------------|------------------------|-----------------------------------------------------------------|
| Modelo jerarquico | ICC = 27.25%           | Existen diferencias entre tiendas que las variables no explican |
| Categoria         | +18,777 a + 19,882 MAE | Factor relevante para los 3 modelos                             |
| Formato           | +7005 a 58,577 MAE     | Importancia considerable, especial para la red                  |
| NSE               | +7653 a +16275 MAE     | Aporta informacion sobre las diferencias en venta               |
| Region            | +186 a +8801 MAE       | Contribución varia según el modelo                              |
| Estaciones        | 0 MAE                  | No hay importancia                                              |
| Mejor Modelo      | MCO, R^2 = 0.6186      | Mejor desempeño                                                 |