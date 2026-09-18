import os
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def fetch_coffee_forecast(countries: list[str], end_year: int = 2025) -> dict:
    """Consulta la API FastAPI para obtener los datos de consumo histórico y pronosticado."""
    endpoint = f"{API_URL.strip('/')}/predict"
    payload = {"countries": countries, "end_year": end_year}

    try:
        response = requests.post(endpoint, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Error HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"No se pudo conectar con la API de FastAPI: {str(e)}"}