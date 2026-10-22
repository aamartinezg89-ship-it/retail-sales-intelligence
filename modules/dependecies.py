import subprocess
import sys
import logging
from pathlib import Path

def check_depencies():
    archivo = Path(__file__).resolve().parents[1] / "requirements.txt"
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "q" ,"-r", str(archivo)],
            check=True,
            capture_output=True
        )
        logging.info("Dependecias verificadas")
    except Exception:
        logging.error("Error al verificar dependencias")
        raise
