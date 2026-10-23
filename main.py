import logging
import numpy as np
import polars as pl
from datetime import date

from modules.dependecies import check_depencies
from modules.data_loader import load_data
from modules.eda import run_eda
from config import DATA_DIR

from modules.preprocessing import (
    preprocess_data, validate_imputation, merge_data, participacion_ventas, transform_data)

from modules.modeling import (
    train_ols, prepare_features, split_data, train_decision_tree, train_xgboost, train_neural_network, train_hierarchical)

from modules.evaluation import (
    metricas, evaluate_ols, evaluate_model, feature_importance, diagnostic_model, grouped_importance, export_tree)

from modules.mlflow_tracking import registrar_modelo

from modules.evaluation_2 import diagnosticar
from modules.results import guardar_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # 1) carga
    check_depencies()
    datasets = load_data()

    df = pl.from_pandas(datasets["transactions"])
    df_tiendas = pl.from_pandas(datasets["stores"])
    df_calendario = pl.from_pandas(datasets["calendar"])

    # 2) eda
    run_eda(df)

    # 3) preproceso
    df_imputado = preprocess_data(df)
    validate_imputation(df, df_imputado)

    df_panel = merge_data(df_imputado, df_tiendas, df_calendario)

    participacion_ventas(df_panel)
    df_model = transform_data(df_panel)

    # 4) tiempo
    X, y = prepare_features(df_model)
    X_train, X_test, y_train, y_test = split_data(df_panel, X, y)

    corte = date(2023, 12, 1)

    df_train = df_panel.filter(pl.col("date") < corte)
    df_test = df_panel.filter(pl.col("date") >= corte)

    fechas_train = df_model.filter(pl.col("date") < corte)["date"]

    # 5) entrenamiento
    modelo_ols = train_ols(df_train)
    modelo_tree = train_decision_tree(X_train, y_train, fechas_train)
    modelo_xgb = train_xgboost(X_train, y_train, fechas_train)
    modelo_mlp = train_neural_network(X_train, y_train)
    modelo_hierarchical = train_hierarchical(df_train)

    # 6) evaluacion train/test
    ols_train, ols_test = evaluate_ols(modelo_ols, df_train, df_test)

    modelos_ml = {
        "Decision Tree": modelo_tree,
        "XGBoost": modelo_xgb,
        "MLP": modelo_mlp
    }

    evaluaciones = {}

    for nombre, modelo in modelos_ml.items():
        train, test = evaluate_model(modelo, X_train, y_train, X_test, y_test, nombre)

        evaluaciones[nombre] = {"train": train, "test": test}

    guardar_json("model_results", {
        "OLS": {"train": ols_train, "test": ols_test},
        "modelos_ml": evaluaciones
    })

    # 7) diagnostico comercial
    diagnosticos = {}

    for nombre, modelo in modelos_ml.items():
        diagnosticos[nombre] = diagnosticar(
            y_train.to_numpy(),
            modelo.predict(X_train.to_pandas()),
            y_test.to_numpy(),
            modelo.predict(X_test.to_pandas())
        )

    diagnosticos["OLS"] = diagnosticar(
        df_train["amount_total"].to_numpy(),
        np.exp(modelo_ols.predict(df_train.to_pandas())),
        df_test["amount_total"].to_numpy(),
        np.exp(modelo_ols.predict(df_test.to_pandas()))
    )

    guardar_json("diagnostics", diagnosticos)

    print("\n========== SESGO Y DERIVA ==========")
    print(f"{'Modelo':<16} {'PBIAS Test':>12} {'KS Error':>12} {'p-value':>12}")

    for nombre, r in diagnosticos.items():
        print(
            f"{nombre:<16} "
            f"{r['test']['pbias_pct']:>11.2f}% "
            f"{r['deriva_errores']['ks']:>12.4f} "
            f"{r['deriva_errores']['p_value']:>12.4g}"
        )

    print("Resultados: reports/diagnostics.json")

    # mlflow
    for nombre, modelo in modelos_ml.items():
        registrar_modelo(
            nombre,
            modelo.get_params(),
            evaluaciones[nombre]["test"]
        )

    registrar_modelo(
        "OLS",
        {"modelo": "OLS", "covarianza": "cluster"},
        ols_test
    )
if __name__ == "__main__":
    main()