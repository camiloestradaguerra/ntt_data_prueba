import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.agent.tools import fetch_coffee_forecast

load_dotenv()

# Lee el modelo del .env
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-0528")

hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("❌ No se encontró el token de Hugging Face en el archivo .env")

# Cliente oficial de Hugging Face
client = InferenceClient(api_key=hf_token)


def analyze_market(countries: list[str], end_year: int = 2025) -> str:
    """Obtiene los datos de la API y genera el análisis con el cliente oficial de Hugging Face."""
    forecast_data = fetch_coffee_forecast(countries, end_year)

    if "error" in forecast_data:
        return f"❌ No se pudo realizar el análisis: {forecast_data['error']}"

    system_prompt = (
        "Eres el Agente Senior de Innovación e Inteligencia de Mercado de Café.\n"
        "Analiza la información de demanda (histórica y proyectada) entregada por la API de ML.\n"
        "Instrucciones:\n"
        "- Explica la tendencia de consumo de cada país.\n"
        f"- Calcula el porcentaje de crecimiento entre el último año histórico (2019) y el proyectado ({end_year}).\n"
        "- Propón 2 oportunidades de innovación en mercado y estrategia de precios.\n"
        "- Responde de forma clara, profesional y estructurada con viñetas."
    )

    user_prompt = f"Aquí están los datos de la API:\n\n{forecast_data}"

    try:
        # Intento mediante Chat Completion (ideal para DeepSeek/Qwen/Llama)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1500,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception:
        # Fallback a text_generation si el modelo es legacy
        prompt = f"{system_prompt}\n\nDatos:\n{user_prompt}\n\nAnálisis:"
        return client.text_generation(prompt, model=MODEL_NAME, max_new_tokens=1500)