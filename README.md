# Inteligencia de ventas | Resumen General

## Descripción

Investigación aplicada para explicar diferencias de ventas entre establecimientos a partir de tres fuentes: `transactions.csv`, `stores.csv` y `calendar.csv`. La integración construye un panel **tienda–categoría–día** con variables comerciales, estructurales y temporales.

**Objetivo:** analizar asociaciones con las ventas, explorar relaciones no lineales y cuantificar heterogeneidad residual entre tiendas. No se atribuye causalidad a las asociaciones estimadas.

## Datos y enfoque

| Indicador     | Valor                              |
|---------------|------------------------------------|
| Observaciones | 203,958                            |
| Tiendas       | 80                                 |
| Categorías    | 6                                  |
| Cobertura     | Enero 2023–febrero 2024 (14 meses) |
| Entrenamiento | Enero–noviembre 2023               |
| Prueba        | Diciembre 2023–febrero 2024        |

Se comparan **MCO log-lineal** (asociaciones interpretables), **Decision Tree, XGBoost y MLP** (relaciones no lineales) y un **modelo jerárquico** (variación entre tiendas).

## Resultados principales

| Modelo            | MAE prueba    | RMSE prueba    | R² prueba  |
|-------------------|---------------|----------------|------------|
| MCO               | 89,849.11     | 184,263.55     | 0.1046     |
| **Decision Tree** | **81,232.06** | **164,859.48** | **0.2832** |
| XGBoost           | 84,207.45     | 180,066.01     | 0.1449     |
| MLP               | 84,264.44     | 179,638.57     | 0.1489     |

El **árbol de decisión** obtuvo el mejor desempeño fuera de muestra en el corte evaluado. El modelo jerárquico estimó **ICC de 33.00%**. 
Los modelos presentan subestimación agregada de ventas en prueba y diferencias entre las distribuciones de errores de entrenamiento y prueba.

## Documentación técnica

**[Proceso completo: metodología, tratamiento de faltantes, modelos, sesgo, deriva y uso de IA](PROCESS.md)**

Los resultados se guardan en `reports/` e incluyen `eda_results.json`, `model_results.json` y `diagnostics.json`.

## Ejecución

```bash
git clone https://github.com/aamartinezg89-ship-it/retail-sales-intelligence.git
cd retail-sales-intelligence
pip install -r requirements.txt
python main.py
```

Los tres CSV deben estar en la ruta configurada en `config.py`. Comprueba que `requirements.txt` esté publicado.

Para visualizar experimentos registrados:

```bash
mlflow ui
```
si da error, probar:
```bash
mlflow ui --host 127.0.0.1 --port 5001
```

**[Ver repositorio en GitHub](https://github.com/aamartinezg89-ship-it/retail-sales-intelligence)**