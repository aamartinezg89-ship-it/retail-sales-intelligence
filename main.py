import logging
import polars as pl
from modules.dependecies import check_depencies
from config import DATA_DIR
from datetime import date

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # 1) verificar dependencias
    check_depencies()

    # 2) cargar datasets
    from modules.data_loader import load_data
    datasets = load_data()

    #3) confirmar que los datasets se cargaron correctamente
    for nombre, df in datasets.items():
        print(nombre, df.shape)

    # eda
    from modules.eda import run_eda
    df = pl.from_pandas(datasets["transactions"])

    run_eda(df)

    # preproceso

    from modules.preprocessing import preprocess_data, validate_imputation, merge_data, participacion_ventas, transform_data

    df_imputado = preprocess_data(df)
    resumen = validate_imputation(df, df_imputado)

    # integracion

    df_tiendas = pl.from_pandas(datasets["stores"])
    df_calendario = pl.from_pandas(datasets["calendar"])

    df_panel = merge_data(df_imputado, df_tiendas, df_calendario)
    resultados = participacion_ventas(df_panel)

    # modelo
    df_model = transform_data(df_panel)

    from modules.modeling import prepare_features, split_data, train_decision_tree, train_xgboost, train_hierarchical

    X, y = prepare_features(df_model)
    X_train, X_test, y_train, y_test = split_data(df_panel, X, y)

    print(X.select(pl.col(pl.String)).columns)


    from modules.modeling import train_ols

    df_train = df_panel.filter(
        pl.col("date") < date(2024, 1, 1)
    )

    modelo_ols = train_ols(df_train)


    from modules.evaluation import evaluate_ols, evaluate_model, feature_importance, diagnostic_model, grouped_importance, export_tree

    df_test = df_panel.filter(
        pl.col("date") >= date(2024, 1, 1)
    )

    mae, rmse, r2 = evaluate_ols(modelo_ols, df_test)

    fechas_train = df_model.filter(
        pl.col("date") < date(2024, 1, 1)
    )["date"]

    modelo_tree = train_decision_tree(
        X_train, y_train, fechas_train
    )

    export_tree(modelo_tree, X_train)

    mae_tree, rmse_tree, r2_tree = evaluate_model(
        modelo_tree, X_test, y_test, "Decision Tree"
    )

    modelo_xgb = train_xgboost(
        X_train, y_train, fechas_train
    )

    mae_xgb, rmse_xgb, r2_xgb = evaluate_model(
        modelo_xgb, X_test, y_test, "XGBoost"
    )

    feature_importance(modelo_tree, X_train, "Decision Tree")
    feature_importance(modelo_xgb, X_train, "XGBoost")

    from modules.modeling import train_neural_network

    modelo_mlp = train_neural_network(X_train, y_train)

    mae_mlp, rmse_mlp, r2_mlp = evaluate_model(
        modelo_mlp, X_test, y_test, "Neural Network MLP"
    )

    df_test = df_panel.filter(
        pl.col("date") >= date(2024, 1, 1)
    )

    diagnostic_model(modelo_tree, X_test, df_test, "Decision Tree")

    diagnostic_model(modelo_xgb, X_test, df_test, "XGBoost")

    diagnostic_model(modelo_mlp, X_test, df_test, "MLP")

    df_train = df_panel.filter(
        pl.col("date") < date(2024, 1, 1)
    )

    modelo_hierarchical = train_hierarchical(df_train)

    grouped_importance(modelo_tree, X_test, y_test, "Decision Tree")

    grouped_importance(modelo_xgb, X_test, y_test, "XGBoost")

    grouped_importance(modelo_mlp, X_test, y_test, "MLP")

    from modules.mlflow_tracking import registrar_modelo

    registrar_modelo(
        "Decision Tree",
        modelo_tree.get_params(),
        {
            "MAE": mae_tree,
            "RMSE": rmse_tree,
            "R2": r2_tree
        }
    )
if __name__ == "__main__":
    main()