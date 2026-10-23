[Volver al README](README.md)
# Inteligencia de ventas

**End-to-End Ciencia de datos | Econometria | Machine Learning**

## 1) Descripcion general del proyecto

El proyecto desarrolla un marco analitico End-to-End para investigar las diferencias en las ventas entre formatos
comerciales y determinar en que grado se pueden asociarse con caracteristicas, estructurales, operativas, socioeconomicas y
temporales.

Se integran tres fuentes de información: transacciones, caracteristicas de tienda y calendario de eventos. Esta combinacion
permite construir un panel de ventas a nivel tienda-categoria-dia.

El enfoque metodologico combina econometria tradicional, algoritmos de Machine Learning, modelos de efectos mixtos y validacion temporal.

**Objetivo central**: Identificar los factores asociados con la variabilidad de las ventas, evaluar posibles relaciones
no lineales y cuantificar la heterogneidad entre tiendas que se puede explciar con las variables observadas.

## 2) Problema de negocio

Las diferencia en las ventas pueden responde a diversas condiciones:

* Formato de tienda y categoria
* Tamaño de tiendas y numero de cajas registradoras
* Promociones y eventos comerciales
* Farmacias y gasolineras
* Regiones y nviel socioeconomico

La investigacion parte de dos preguntas:

* ¿Qué factores están estadisticamente asociados con las diferencias en las ventas?
* ¿Existen relaciones no lineales que los modelo lineales pueden estar dejando afuera?


## 3) Estructura analitica y conjunto de datos

El conjunto de datos esta organizado como un panel de ventas al día

| Componentes              | Descripcion                 |
|--------------------------|-----------------------------|
| Observaciones            | 203, 958                    |
| Tiendas                  | 80                          |
| Categorias deproducto    | 6                           |  
| Perido                   | Enero 2023 - Febrero 2024   |
| Nivel observacional      | Tienda - Categoria - Día    |
| Variable objetivo        | amount_total                |
| Periodo de entrenamiento | Enero-noviembre 2023        |
| Periodo de prueba        | Diciembre 2023-Febrero 2024 |

La disponibilidad de 14 meses de informacion limita el estudio de patrones temporales estacionales de largo plazo. En ese sentido,
se prioriza modelo explicativos.

## 4) Calidad de Datos y EDA

En el analisis inicial se identificaton 36, 198 valores faltantes, sin registros duplicados.

### Variables con valores faltantes

| Variables         | Porcentaje faltante |
|-------------------|---------------------|
| cash_transactions | 5.94                |
| amount_cash       | 5.94                |
| units_sold        | 3.00                |
| avg_ticket        | 2.39                |
| replenishment     | 0.47                | 

### Deteccion de agrupamientos temporales de los valores faltantes

Se aplica una prueba de rachas (Runs test) para determinar si las secuencias de observaciones faltantes y no faltantes
presentan un comportamiento compatible con sentido aleatorio. La hipotesis nula dice que la secuencia se ordena aleatoriamente.

| Indicador                                   | Resultado |
|---------------------------------------------|-----------|
| Contrastes realizados                       | 1650      |
| Contrastes significativos                   | 5         |
| Proporcion significativa                    | 0.30 %    |
| Registros duplicados                        | 0         |
| Valores Faltantes despues de la imputacion  | 0         | 

Se identifican cinco secuencias con evidencia estadistica de comportamiento no aleatorio. Despues se aplica
una imputacion por medianas mensuales por tienda y categoria.

## 5) Seleccion de modelos

| Modelo                             | Fundamento de selección                                           | 
|------------------------------------|-------------------------------------------------------------------|
| Mínimos Cuadrados Ordinarios (MCO) | Interpretabilidad económica y contraste estadístico               |
| Decision Tree                      | Identificación de reglas e interacciones no lineales              | 
| XGBoost                            | Modelación de relaciones no lineales mediante árboles potenciados |
| Red neuronal MLP                   | Aproximación flexible de relaciones multivariadas                 | 
| Modelo jerárquico                  | Descomposición de variabilidad entre tiendas y dentro de ellas    | 

## 6) Resultados econometricos y estadisticos

### MCO
| Indicador                                               | Resultado                      |
|---------------------------------------------------------|--------------------------------|
| Observaciones de entrenamiento                          | 160,296                        |
| R² ajustado en escala logarítmica                       | 0.7268                         |
| Coeficientes estadísticamente significativos reportados | 17                             |
| Covarianza                                              | Robustez por clúster de tienda |

### Modelo Jerarquico
| Indicador              | Resultado  |
|------------------------|------------|
| Varianza entre tiendas | 0.0610     |
| Varianza residual      | 0.1237     |
| ICC                    | **33.00%** |
| Convergencia           | Exitosa    |

El ICC(coeficiente de correlación intraclase) de 33% indica que hay heterogeneidad estructural perisisnte
que no se integra por las variables incluidas.

## 7) Evuacion fuera de la muestra

### Entrenamiento y prueba
| Modelo            | MAE Train     | MAE Test      | RMSE Test      | R² Test    |
|-------------------|---------------|---------------|----------------|------------|
| MCO               | 46,630.07     | 89,849.11     | 184,263.55     | 0.1046     |
| **Decision Tree** | **48,289.71** | **81,232.06** | **164,859.48** | **0.2832** |
| XGBoost           | 41,960.85     | 84,207.45     | 180,066.01     | 0.1449     |
| MLP               | 42,901.51     | 84,264.44     | 179,638.57     | 0.1489     |

### Parametros principales
| Modelo        | Configuración reportada                                 |
|---------------|---------------------------------------------------------|
| Decision Tree | Profundidad 8; mínimo 100 observaciones por hoja        |
| XGBoost       | 200 estimadores; profundidad 5; learning rate 0.10      |
| MLP           | Arquitectura de capas ocultas (64, 32); 300 iteraciones |

El entrenamiento del MLP alcanzó el máximo de iteraciones sin satisfacer 
el criterio de convergencia, lo que constituye una limitación del resultado presentado.

## 8) Diagnostico de sesgo y estabilidad temporal
- **PBIAS:** sesgo porcentual agregado de las predicciones respecto a las ventas observadas.
- **Kolmogorov–Smirnov (KS):** diferencia máxima entre las distribuciones empíricas de errores de entrenamiento y prueba.

| Modelo            | PBIAS Test  | KS de errores |
|-------------------|-------------|---------------|
| **Decision Tree** | **+25.12%** | **0.1865**    |
| XGBoost           | +35.18%     | 0.2730        |
| MLP               | +34.80%     | 0.2645        |
| MCO               | +38.99%     | 0.2467        |

## 9) Transparencia en el desarollo e integracion IA
Este proyecto incorpora herramientas de AI como asistente del desarrollo, soporte de la metodología estadistica, y
en la automatizacion de la documentacion.

| Tecnologia               | Aplicacion                                                                                                                                                           |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GitHub Copilot           | Autocompletar el codigo y sugerir la linea siguiente durante el desarrollo en Python                                                                                 |
| ChatGPT (1)              | Asistencia tecnica para investigar y aplicar pruebas cuando se tienen valores faltantes en paneles de tiempo                                                         |
| ChatGPT (2)              | Asistencia en la estructuración, redacción y presentación de resúmenes de resultados, README y PROCCESS, utilizando como referencia las salidas generadas por Python |
| Ollama + Llamma 3.2 (3B) | Integracion LLM a nivel local para automatizar la generacion de Data Card del los resultado de EDA                                                                   |

### FLujo de trabajo LLM local

El proyecto integra un modelo de lenguaje que esta instalado localmente mediante Ollama API. EL LLM
recibe los resultados de EDA en JSON y el agente genera una documentacion en Markdown bajo
formatos predefinidos e instrucciones numericas.

### Validacion humana y responsabilidad de IA

* Pruebas estadisticas y los modelos de machine learning son ejecutados en Python
* Asistencia IA en la codificacion y las sugerencias metologicas son revisadas antes de la integracion
* La IA se utilizó como herramienta de asistencia para el desarrollo y la documentación
* La documentacion generado por LLM son revisados comparando el origen de los resultado estadisticos
* La seleccion de los modelos, interpretacion estadistica, y conclusiones finales se mantienen bajo la responsabilidad del humano

## 10) Hallazgos

* Se genera un panel de datos que permite evaluar la diferencia entre ventas con base en condiciones comerciales y estructurales
* La prueba de rachas indica que los valores faltantes no son continuos
* La regresion log-lineal da un parametro para investigar asociaciones entre condiciones comerciales y ventas. 
* Los modelos de ML complementan la econometria al explorar relaciones no lineales, aunque una mayor complejidad no mejoro desempeño fuera de muestra. 
* Decision Tree obtuvo el mejor desempeño predictivo en el periodo de prueba.

## 11) Ejecucion del proyecto
Clonar el repositorio e instalar las dependencias:

```bash
git clone https://github.com/aamartinezg89-ship-it/retail-sales-intelligence.git
cd retail-sales-intelligence
pip install -r requirements.txt
```

Colocar los tres archivos CSV en la carpeta de datos configurada.

Ejecutar el proyecto:

```bash
python main.py
```

Visualizar los experimentos en MLflow:

```bash
mlflow ui
```
si da error, probar:
```bash
mlflow ui --host 127.0.0.1 --port 5001
```

Los resultados se almacenan en la carpeta `reports/`, incluyendo métricas, diagnósticos estadísticos y documentación.

> Nota: verificar que `requirements.txt` exista en el repositorio.
