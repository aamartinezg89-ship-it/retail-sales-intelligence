import polars as pl
import logging
from skimpy import skim
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

def run_eda(df: pl.DataFrame):

    logger.info("Iniciando análisis exploratorio de datos (EDA)")

    logger.info(f"Dimensiones: {df.shape}")
    logger.info(f"Tipo de datos: {df.schema}")
    logger.info(f"Valores nulos:\n{df.null_count()}")
    logger.info(f"Duplicados:\n{df.is_duplicated().sum()}")
    logger.info(f"Estadísticas descriptivas:\n{df.describe()}")

    # Porcentaje de valores nulos por columna
    df_null =(
        df.select((pl.all().is_null().sum() / df.height * 100))
        .transpose(
            include_header=True,
            header_name="variable",
            column_names=["null_percentage"]
        )
        .filter(pl.col("null_percentage") > 0)
        .sort("null_percentage", descending=True)
    )
    logger.info(f"Porcentaje de valores nulos por columna:\n{df_null}")

    # Agrupacion temporal de nulos por mes y año

    df_null_time =(
        df.select("date","store_id","cash_transactions","amount_cash","units_sold","avg_ticket","replenishment_signal")
        .with_columns(pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d").alias("date"))
        .with_columns(pl.col("date").dt.year().alias("year"),
                      pl.col("date").dt.month().alias("month"),
                      pl.col("date").dt.week().alias("week"))
        .group_by(["year", "month", "week"])
        .agg(pl.col("cash_transactions","amount_cash","units_sold","avg_ticket","replenishment_signal").is_null().sum(),
             pl.col("store_id").n_unique().alias("count"),
             pl.col("store_id")
             .filter(pl.any_horizontal(pl.col("cash_transactions","amount_cash","units_sold","avg_ticket","replenishment_signal").is_null()))
             .n_unique().alias("stores_with_nulls")
             )
        .sort(["year", "month", "week"])
    )
    logger.info(f"Agrupación temporal de valores nulos por mes y año:\n{df_null_time}")

    # binario

    df_binario =(
        df.select("date", "store_id", "category",
                  *[pl.col(c).is_not_null().cast(pl.Int8).alias(c)
                   for c in ["cash_transactions","amount_cash","units_sold","avg_ticket","replenishment_signal"]]
    ).with_columns(pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")
    ).with_columns(pl.col("date").dt.year().alias("year"),
                      pl.col("date").dt.month().alias("month"),
                      pl.col("date").dt.week().alias("week"),
                      pl.col("date").dt.weekday().alias("weekday"))
     .with_columns(pl.sum_horizontal(
            "cash_transactions","amount_cash","units_sold","avg_ticket","replenishment_signal").alias("total")
    ).filter(pl.col("total") < 5)
     .filter(pl.col("store_id") == "STR_055")
     .group_by("year","month","week","weekday","category","total")
     .agg(pl.len().alias("observaciones"))
     .pivot(
        on="category",
        index=["year", "month", "week", "weekday", "total"],
        values="observaciones",
        aggregate_function="sum"
      )
     .sort("year","month","week", "weekday","weekday","total", descending=False)
    )

    logger.info(f"DataFrame binario de valores nulos:\n{df_binario}")

    # Pruba run
    """
    RUNS TEST - Analisis de valores faltantes
    
    Objetivo: Evaluar si la secuencia temporal de los valores faltantes presenta un comportamiento aleatorio o se agrupa
    
    H0: Los valores faltantes se distribuyen de manera aleatoria
    H1: Los valores faltantes presentan un patrón no aleatorio
    
    """
    df_run = (
        df.unpivot(
            index=["date", "store_id", "category"],
            on=[c for c in df.columns
                if c not in ["date", "store_id", "category", "has_promotion"]],
            variable_name="variable",
            value_name="value"
        )
        .with_columns(
            pl.col("value").is_not_null().cast(pl.Int8).alias("missing")
        )
        .select("date", "store_id", "category", "variable", "missing")
        .sort(["store_id", "category", "variable", "date"])
    )

    resultados = []

    for (tienda, categoria, variable), grupo in df_run.group_by(
        ["store_id", "category", "variable"]
    ):
        serie = grupo.sort("date")["missing"].to_numpy()

        if (serie == 0).sum() <10 or (serie == 1).sum() < 10:
            continue
        z, p =runstest_1samp(serie, cutoff=0.05)

        resultados.append({
            "store_id": tienda,
            "category": categoria,
            "variable": variable,
            "z_stat": z,
            "p_value": p
        })

    df_test = pl.DataFrame(resultados)

    logger.info(f"Resultados de la prueba de aleatoriedad:\n{df_test}")

    # Corrección de p-valores por múltiples comparaciones
    """
    Correcion de Benjamini - Hochberg para controlar los falsos descubrimientos
    
    """
    df_test = df_test.with_columns(
        pl.Series(
            name="p_adjusted",
            values=multipletests(df_test["p_value"].to_numpy(), method="fdr_bh")[1]
        )
    )

    print(
        df_test.select(
            pl.len().alias("pruebas"),
            (pl.col("p_adjusted") < 0.05).sum().alias("significativas_ajustadas"),
            (pl.col("p_adjusted") >= 0.05).sum().alias("no_significativas_ajustadas")
        )
    )

    logger.info(f"Resultados de la prueba de aleatoriedad con corrección de p-valores:\n{df_test}")

    df_significativos = df_test.filter(pl.col("p_adjusted") < 0.05).sort("p_adjusted")

    logger.info(f"Resultados de la prueba de aleatoriedad significativos:\n{df_significativos}")

    # resumen final
    logger.info("Resumen final del análisis exploratorio de datos (EDA)")
    logger.info(f"Observaciones: {df.height:,}")
    logger.info(f"Variables: {df.width}")
    logger.info(f"Tiendas únicas: {df.select('store_id').n_unique()}")
    logger.info(f"Categorías únicas: {df.select('category').n_unique()}")
    logger.info(f"Periodo: {df['date'].min()} - {df['date'].max()}")
    logger.info(f"Duplicados: {df.is_duplicated().sum()}")

    logger.info("Pruebdas de rachas:")
    logger.info(f"Contraste realizados: {df_test.height}")
    logger.info(
        f"Significativas ajustadas:"
        f"{df_significativos.height}"
    )

    logger.info("Análisis exploratorio de datos (EDA) finalizado")