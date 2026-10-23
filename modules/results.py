import json
from pathlib import Path

def guardar_json(nombre, datos):
    ruta = Path("reports") / f"{nombre}.json"
    ruta.parent.mkdir(exist_ok=True)

    ruta.write_text(
        json.dumps(datos, indent=2, ensure_ascii=False),
        encoding="utf8"
    )