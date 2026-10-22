import polars as pl
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
import numpy as np
import statsmodels.formula.api as smf
from datetime import date
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)

# 1) Capa de preparacion para ML
def prepare_features(df: pl.DataFrame):

    excluir = [
        "date", "store_id", "amount_total",
        "amount_cash", "amount_card",
        "total_transactions", "cash_transactions",
        "card_transactions", "units_sold",
        "avg_ticket", "replenishment_signal",
        "opening_year"
    ]

    X = df.drop(excluir)
    y = df["amount_total"]

    logger.info(f"X: {X.shape}")
    logger.info(f"Y: {y.shape}")
    logger.info(f"Variables predictoras: {X.columns}")
    logger.info(f"Nulos en X: {X.null_count().sum_horizontal()[0]}")

    return X, y

# 2) Division para el entrenamiento y prueba
def split_data(df, X, y):

    train = df["date"] < date(2024, 1, 1)

    X_train = X.filter(train)
    X_test = X.filter(~train)

    y_train = y.filter(train)
    y_test = y.filter(~train)

    return X_train, X_test, y_train, y_test

# 3) Modelo MCO
def train_ols(df):

    formula = """
        np.log(amount_total) ~
        C(store_format, Treatment(reference="Bodega"))
        * C(category, Treatment(reference="Abarrotes"))
        + C(region)
        + C(socioeconomic_level)
        + has_pharmacy
        + has_fuel_station
        + np.log(size_sqm)
        + np.log(num_checkouts)
        + has_promotion
        + C(season)
        + is_holiday
        + is_buen_fin
    """
    datos = df.to_pandas()

    modelo = smf.ols(
        formula=formula,
        data=datos
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": datos["store_id"]}
    )

    significativas = modelo.pvalues[modelo.pvalues < 0.05].drop("Intercept", errors = "ignore")

    logger.info(f"Resultados de regresion econometrica")
    logger.info(f"R² ajustado: {modelo.rsquared_adj:.4f}")
    logger.info(f"Observaciones: {int(modelo.nobs):,}")
    logger.info(f"Covarianza: {modelo.cov_type}")

    logger.info("Variables estadísticamente significativas:")

    for variable in significativas.index:
        logger.info(
            f"{variable}: "
            f"β={modelo.params[variable]:.4f} | "
            f"p={modelo.pvalues[variable]:.4f}"
        )

    return modelo

#4) Arbol de decision
"""
Objetivo: Identificar patrones no lineales y segmentaciones comerciales asociadas con las diferencias en ventas 

Algoritmo: CART (Classificacion and Regression Trees) aplicado a la regresion

    1. Selecciona la variable y el umbral que minimiza el error cuadratico ponderaro de los nodos resultantes
    2. Divide las observaciones recursivas
    3. Construye reglas de decision entre grupos comerciales
    4. Predice las ventas mediante el prodemio de cada hoja

Optimizacion: GridSearchCV con validacion temporal por fechas completas

"""
def train_decision_tree(X_train, y_train, fechas_train):

    fechas = fechas_train.to_numpy()
    dias = np.unique(fechas)

    cv = [
        (
            np.flatnonzero(np.isin(fechas, dias[tr])),
            np.flatnonzero(np.isin(fechas, dias[val]))
        )
        for tr, val in TimeSeriesSplit(n_splits=3).split(dias)
    ]

    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        {
            "max_depth": [4, 6, 8],
            "min_samples_leaf": [30, 60, 100]
        },
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    grid.fit(X_train.to_pandas(), y_train.to_numpy())

    logger.info(f"Mejores parámetros: {grid.best_params_}")
    logger.info(f"MAE validación: {-grid.best_score_:,.2f}")

    return grid.best_estimator_

# 5) XGboost
"""
Objetivo: Identificar relaciones no lineales asociadas con las ventas

Metodo: 
    1. Gradiente Boosting: construye árboles secuenciales que corrigen los errores de predicción anteriores
    
Optimizacion: GridSearchCV con validacion temporal
"""

def train_xgboost(X_train, y_train, fechas_train):

    fechas = fechas_train.to_numpy()
    dias = np.unique(fechas)

    cv = [
        (
            np.flatnonzero(np.isin(fechas, dias[tr])),
            np.flatnonzero(np.isin(fechas, dias[val]))
        )
        for tr, val in TimeSeriesSplit(n_splits=3).split(dias)
    ]

    parametros = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1]
    }

    grid = GridSearchCV(
        XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=2
        ),
        param_grid=parametros,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=1
    )

    grid.fit(X_train.to_pandas(), y_train.to_numpy())

    logger.info(f"XGBoost - Mejores parámetros: {grid.best_params_}")
    logger.info(f"XGBoost - MAE validación: {-grid.best_score_:,.2f}")

    return grid.best_estimator_

# 6) Red neuronal - Regresion MLP
"""
Objetivo: Modelar patrones complejos y no lineales de ventas

Metodo: Red Neuroal con dos capas ocultas (64 y 32 neuronas) activada por ReUL y optimizacion mediante retropropagacion

Procesamiento: Estandarizacion de variables con StandardScaler
"""

def train_neural_network(X_train, y_train):

    modelo = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            random_state=42
        )
    )

    modelo.fit(X_train.to_pandas(), y_train.to_numpy())

    logger.info("Red neuronal MLP entrenada")

    return modelo

#8) Modelo Jerarquico - Efecto mixtos
"""
Objetivo: Cuantificar las diferencias persistentes de ventas entre tiendas

Metodo: Modelo de efectos mixtos con intercepto aleatorio por tienda y efectos fijos en el formato, categoria y temporalidad
"""
def train_hierarchical(df_train):

    datos = df_train.to_pandas()

    modelo = smf.mixedlm(
        "np.log(amount_total) ~ C(store_format) + C(category) + C(season)",
        data=datos,
        groups=datos["store_id"]
    ).fit(reml=True)

    # Descomposición de varianza
    var_tienda = modelo.cov_re.iloc[0, 0]
    var_residual = modelo.scale

    icc = var_tienda / (var_tienda + var_residual)

    logger.info("Modelo jerárquico")
    logger.info(f"Varianza entre tiendas: {var_tienda:.4f}")
    logger.info(f"Varianza residual: {var_residual:.4f}")
    logger.info(f"ICC: {icc:.2%}")
    logger.info(f"Convergencia: {modelo.converged}")

    return modelo

