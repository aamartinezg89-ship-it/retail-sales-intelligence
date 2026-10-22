import mlflow

mlflow.set_experiment("Retail sales intelligence")

def registrar_modelo(nombre, parametros, metricas):

    with mlflow.start_run(run_name=nombre):
        mlflow.log_params(parametros)
        mlflow.log_metrics(metricas)