import sys
from pathlib import Path

# Garantiza la resolución correcta de módulos
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.bot import analyze_market


def main():
    print("🤖 Agente de Inteligencia de Mercado de Café Listo.\n")

    # Países a analizar de forma predeterminada o mediante consulta
    paises = ["Colombia", "Brazil"]
    print(f"📊 Consultando predicciones para: {', '.join(paises)} hasta 2025...\n")

    respuesta = analyze_market(countries=paises, end_year=2025)

    print("--- ANÁLISIS DEL AGENTE ---")
    print(respuesta)


if __name__ == "__main__":
    main()