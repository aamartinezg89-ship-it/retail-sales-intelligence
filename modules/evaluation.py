import numpy as np
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import date
import polars as pl
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from pathlib import Path

logger = logging.getLogger(__name__)

# 1) Evalucion MCO
"""
Evalua la regresion MCO sobre datos de prueba.
Transforma las predicciones logaritmicas a ventas originales.
Calcula MAE, RMSE y R^2
"""
def evaluate_ols(modelo, df_test):

    datos = df_test.to_pandas()

    y_real = datos["amount_total"]
    y_pred = np.exp(modelo.predict(datos))

    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)

    logger.info("Evaluación OLS - datos de prueba")
    logger.info(f"MAE: {mae:,.2f}")
    logger.info(f"RMSE: {rmse:,.2f}")
    logger.info(f"R² prueba: {r2:.4f}")

    return mae, rmse, r2

# 2) Evalua modelo de ML
"""
Evalua modelos de ML sobre datos de prueba.
Compara ventas observadas y predichas mediante MAE, RMSE y R^2 para medir su capacidad de generalizacion
"""

def evaluate_model(modelo, X_test, y_test, nombre):
    real = y_test.to_numpy()
    pred = modelo.predict(X_test.to_pandas())

    mae = mean_absolute_error(real, pred)
    rmse = np.sqrt(mean_squared_error(real, pred))
    r2 = r2_score(real, pred)

    logger.info(f"Evaluación {nombre} - Prueba 2024")
    logger.info(f"MAE: {mae:,.2f}")
    logger.info(f"RMSE: {rmse:,.2f}")
    logger.info(f"R²: {r2:.4f}")

    return mae, rmse, r2

# 3) Importancia de variables
"""
Identifica las variables más importantes para el modelo.
Ordena su contribución relativa según la importancia interna del algoritmo, sin implicar causalidad
"""
def feature_importance(modelo, X, nombre):

    importancia = sorted(
        zip(X.columns, modelo.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )

    logger.info(f"Importancia de variables - {nombre}")

    for variable, valor in importancia[:15]:
        logger.info(f"{variable}: {valor:.2%}")

    return importancia

# 4) Diagnosticos
"""
Analiza los errores de prediccion por formato y categoria.
Calcula el MAE de cada segmento comercial para identificar
diferencias en el desempeño predictivo.
"""
def diagnostic_model(modelo, X_test, df_test, nombre):

    resultado = df_test.select([
        "store_format", "category", "amount_total"
    ]).with_columns(
        pl.Series("prediccion", modelo.predict(X_test.to_pandas()))
    )

    resultado = resultado.with_columns(
        (pl.col("amount_total") - pl.col("prediccion"))
        .abs()
        .alias("error")
    )

    for variable in ["store_format", "category"]:

        tabla = (
            resultado.group_by(variable)
            .agg(
                pl.len().alias("observaciones"),
                pl.col("error").mean().round(2).alias("MAE")
            )
            .sort("MAE", descending=True)
        )

        logger.info(f"{nombre} - MAE por {variable}")
        logger.info(f"\n{tabla}")

    return resultado

# 5) Importacia agrupada
"""
Evalua la importancia predictiva de dimensiones comerciales.
Permuta conjuntamente variables binarias de cada grupo y mide el aumento de MAE, sin causalidad.
"""
def grouped_importance(modelo, X_test, y_test, nombre):

    X = X_test.to_pandas()
    y = y_test.to_numpy()

    grupos = {
        "Formato": "store_format_",
        "Categoría": "category_",
        "Región": "region_",
        "NSE": "socioeconomic_level_",
        "Temporada": "season_"
    }

    base = mean_absolute_error(y, modelo.predict(X))
    resultados = []
    rng = np.random.default_rng(42)

    for grupo, prefijo in grupos.items():

        columnas = [c for c in X.columns if c.startswith(prefijo)]
        errores = []

        for _ in range(5):
            X_perm = X.copy()
            orden = rng.permutation(len(X))

            X_perm[columnas] = X[columnas].iloc[orden].to_numpy()

            mae = mean_absolute_error(y, modelo.predict(X_perm))
            errores.append(mae - base)

        resultados.append((grupo, np.mean(errores)))

    resultados.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"Importancia agrupada - {nombre}")

    for grupo, aumento in resultados:
        logger.info(f"{grupo}: +{aumento:,.2f} MAE")

    return resultados

# 6) Exporta el arbol
"""
Exporta una representacion grafica del arbol en PNG.
Muestra sus primeros tres niveles, reglas de división y valores
predichos sin modificar el entrenamiento.
"""
def export_tree(modelo, X, ruta="reports/decision_tree.png"):

    Path(ruta).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(22, 12))

    plot_tree(
        modelo,
        feature_names=X.columns,
        max_depth=3,
        filled=True,
        rounded=True,
        fontsize=9
    )

    plt.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info("Arbol exportado: {ruta}")