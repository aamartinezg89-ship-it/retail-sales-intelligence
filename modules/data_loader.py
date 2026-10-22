import pandas as pd
import logging
import traceback
from config import DATA_DIR

logger = logging.getLogger(__name__)

def load_data():
    try:
        datasets = {}

        for archivo in DATA_DIR.glob("*.csv"):
            datasets[archivo.stem] = pd.read_csv(archivo)
            logger.info(f"Cargado: {archivo.name}")

        if not datasets:
            raise FileNotFoundError("No se encontraron archivos CSV en el directorio de datos.")

        logger.info(f"Carga de datos finalizada")
        return datasets

    except Exception:
        logger.error(f"Error en la carpeta:\n{traceback.format_exc()}")
        raise

    