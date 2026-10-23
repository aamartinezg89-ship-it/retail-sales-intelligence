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

    logger.info(
        f"\nDatos\n"
        f"Observados : {X.height:,}\n"
        f"Predictores: {X.width}\n"
        f"Nulos     : {X.null_count().sum_horizontal()[0]}"
    )

    return X, y

# 2) Division para el entrenamiento y prueba
def split_data(df, X, y):

    train = df["date"] < date(2023, 12, 1)

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

    logger.info(
        f"\nOLS\n"
        f"Observaciones  : {int(modelo.nobs):,}\n"
        f"R^2 ajustado   : {modelo.rsquared_adj:.4f}\n"
        f"Covarianza     : Cluster por tienda\n"
        f"Significativas : {len(significativas)}"
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

    logger.info(
        f"\nDecesion Tree\n"
        f"MAE validacion: {-grid.best_score_:,.2f}\n"
        f"Profundidad   : {grid.best_params_['max_depth']}\n"
        f"Minimo hoja  : {grid.best_params_['min_samples_leaf']}"
    )

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

    logger.info(
        f"\nXGBOOST\n"
        f"MAE validacion: {-grid.best_score_:,.2f}\n"
        f"Estimadores  : {grid.best_params_['n_estimators']}\n"
        f"Profundidad   : {grid.best_params_['max_depth']}\n"
        f"Learning  : {grid.best_params_['learning_rate']:.2f}"
    )

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

    red=modelo.named_steps["mlpregressor"]
    logger.info(
        f"\nNeural Network MLP\n"
        f"Arquitectura : {red.hidden_layer_sizes}\n"
        f"Iteraciones : {red.n_iter_}\n"
        f"Loss final: {red.loss_:.4}\n"
        f"R^2 validacion: {red.best_validation_score_:.4f}"
    )

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

    logger.info(
        f"\nModelo Jerarquico\n"
        f"Var. tiendas : {var_tienda:.4f}\n"
        f"Var. residual: {var_residual:.4f}\n"
        f"ICC          : {icc:.2%}\n"
        f"Convergencia : {modelo.converged}\n"
    )

    return modelo

