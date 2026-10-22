import polars as pl
import logging

logger = logging.getLogger(__name__)

def preprocess_data(df):

    columnas = [
        c for c, tipo in df.schema.items()
        if tipo.is_numeric()
    ]

    logger.info(f"Nulos antes: {df.null_count().sum_horizontal()[0]}")

    df_imputado = (
        df.with_columns(
            pl.col("date").str.to_datetime("%Y-%m-%d")
        )
        .with_columns(pl.col("date").dt.truncate("1mo").alias("mes"))
        .with_columns(
            pl.col(columnas).fill_null(
                pl.col(columnas).median().over(["store_id", "category", "mes"])
            )
        )
        .drop("mes")
    )

    logger.info(f"Nulos despues: {df_imputado.null_count().sum_horizontal()[0]}")
    logger.info("Imputacion mensual finalizada")

    return df_imputado

def validate_imputation(df, df_imputado):

    columns = [
        c for c, tipo in df.schema.items()
        if tipo.is_numeric()
    ]

    antes = df.select(columns)
    despues = df_imputado.select(columns)

    resumen = pl.DataFrame({
        "variable": columns,
        "media_antes": antes.mean().row(0),
        "media_despues": despues.mean().row(0),
        "mediana_antes": antes.median().row(0),
        "mediana_despues": despues.median().row(0),
        "std_antes": antes.std().row(0),
        "std_despues": despues.std().row(0)
    })

    logger.info(f"Comparacion estadistica:\n{resumen}")

def merge_data(df, tiendas, calendario):

    columnas_calendario = ["date","season", "is_holiday", "is_buen_fin"]

    df = df.with_columns(
        pl.col("date").cast(pl.Date)
    )

    calendario = calendario.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d")
    )

    df_panel = (
        df.join(tiendas, on="store_id", how="left", validate="m:1")
          .join(
            calendario.select(columnas_calendario),
            on="date",
            how="left",
            validate="m:1"
        )
    )

    logger.info(f"Observaciones: {df_panel.height:,}")
    logger.info(f"Variables: {df_panel.width}")

    return df_panel

def participacion_ventas(df):

    resultados = {}

    for variable in ["store_format", "category"]:

        tabla = (
            df.group_by(variable)
            .agg(pl.col("amount_total").sum().alias("ventas"))
            .with_columns(
                (pl.col("ventas") / pl.col("ventas").sum() * 100)
                .round(2)
                .alias("participacion_pct")
            )
            .sort("ventas", descending=True)
        )

        resultados[variable] = tabla
        logger.info(f"Participación por {variable}:\n{tabla}")

    return resultados

def transform_data(df):

    categoricas = ["store_format", "category", "region", "socioeconomic_level", "season"]

    df_model = (
        df.to_dummies(columns=categoricas)
        .with_columns(pl.col(pl.Boolean).cast(pl.Int8))
    )

    logger.info(f"Observaciones: {df_model.height:,}")
    logger.info(f"Variables transformadas: {df_model.width}")

    return df_model

