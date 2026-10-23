import json
import requests
from pathlib import Path

datos = json.loads(Path("reports/eda_results.json").read_text())

total = datos["runs_test"]
significativos = len(datos["significativos"])


respuesta = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:3b",
        "prompt": f"""
        Genera únicamente una Data Card en Markdown
        a partir de estos datos:

        {json.dumps(datos, ensure_ascii=False)}

        REGLAS:
        - Usa títulos con ##.
        - Usa tablas Markdown válidas.
        - Cada fila de tabla debe ocupar una línea.
        - Cada tabla debe incluir encabezados y separadores.
        - No inventes cifras.
        - No agregues explicaciones ni conclusiones.
        - No escribas introducciones.
        - Devuelve solamente Markdown.
        """,
        "stream": False,
        "options": {"temperature": 0.01}
    },
    timeout=180
)

respuesta.raise_for_status()

reporte = respuesta.json()["response"].strip()
reporte = reporte.removeprefix("```markdown").removeprefix("```")
reporte = reporte.removesuffix("```").strip()

reporte += "\n## Resumen estadisticas\n\n"
reporte += f"- Pruebas realizadas: {total:,}\n"
reporte += f"- Significativas: {significativos} ({significativos/total:.2%})\n"
reporte += f"- Ajuste Benjamini-Hochberg (p ajustado < 0.05)\n"

Path("reports/data_card.md").write_text(
    reporte, encoding="utf-8"
)

print("Data Card generada OK")