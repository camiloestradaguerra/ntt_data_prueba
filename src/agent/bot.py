from __future__ import annotations

import os
import re

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

HISTORICAL_END_YEAR = 2019

# Patrones de notación matemática (LaTeX / Markdown-math) que a veces generan
# modelos de razonamiento (p.ej. DeepSeek-R1) al "mostrar su trabajo" -- se
# usan como red de seguridad para sanear la respuesta incluso si el modelo
# ignora las instrucciones del prompt.
_LATEX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\\text\{([^}]*)\}"), r"\1"),  # \text{Crecimiento} -> Crecimiento
    (re.compile(r"\\frac\{([^}]*)\}\{([^}]*)\}"), r"(\1 / \2)"),  # \frac{a}{b} -> (a / b)
    (re.compile(r"\\times"), "x"),
    (re.compile(r"\\approx"), "≈"),
    (re.compile(r"\\cdot"), "x"),
    (re.compile(r"\\rightarrow|\\to"), "→"),
    (re.compile(r"\\%"), "%"),
    (re.compile(r"\$\$([^$]*)\$\$"), r"\1"),  # bloques $$...$$
    (re.compile(r"\$([^$]*)\$"), r"\1"),  # inline $...$
    (re.compile(r"\\left|\\right"), ""),
]


def _strip_latex_artifacts(text: str) -> str:
    """Convierte cualquier resto de notación LaTeX/Markdown-math a texto plano.

    Red de seguridad: aunque el prompt ya prohíbe LaTeX y los cálculos se
    entregan pre-resueltos, algunos modelos de razonamiento igual insertan
    fragmentos como `\\text{Crecimiento} = ...`. Esta función normaliza esos
    residuos antes de mostrar el reporte al usuario.
    """
    cleaned = text
    for pattern, replacement in _LATEX_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def _compute_growth_summary(forecast_data: dict, end_year: int) -> list[dict[str, object]]:
    """Calcula en Python (no delegado al LLM) el % de crecimiento por país.

    Compara el último valor histórico real (2019) contra el valor
    proyectado en `end_year`, usando exactamente los mismos puntos que
    devuelve la API (`src/api/services.py`). Resolver esta aritmética aquí
    evita que el LLM la "muestre" con notación matemática (LaTeX) y garantiza
    que el porcentaje que aparece en el reporte sea siempre correcto.
    """
    summary: list[dict[str, object]] = []

    for country_result in forecast_data.get("results", []):
        country = country_result.get("country", "desconocido")

        # País inválido (nombre no encontrado en el dataset, ver
        # ForecastingService.predict_country / predict_countries): la API ya
        # no hace fallar todo el lote por esto (sin 404), sino que marca
        # SOLO esta entrada con `error` y `data` vacío. Se propaga el
        # mensaje tal cual para que el LLM lo mencione en vez de inventar
        # una tendencia inexistente para este país.
        api_error = country_result.get("error")
        if api_error:
            summary.append({
                "country": country,
                "base_year": HISTORICAL_END_YEAR,
                "base_value": None,
                "end_year": end_year,
                "end_value": None,
                "growth_pct": None,
                "error": api_error,
            })
            continue

        points = {p["year"]: p["consumption"] for p in country_result.get("data", [])}

        base_value = points.get(HISTORICAL_END_YEAR)
        end_value = points.get(end_year)

        if base_value is None or end_value is None:
            growth_pct = None
        elif base_value == 0:
            growth_pct = None
        else:
            growth_pct = (end_value - base_value) / base_value * 100.0

        summary.append({
            "country": country,
            "base_year": HISTORICAL_END_YEAR,
            "base_value": base_value,
            "end_year": end_year,
            "end_value": end_value,
            "growth_pct": growth_pct,
            "error": None,
        })

    return summary


def _format_growth_summary(summary: list[dict[str, object]]) -> str:
    """Texto plano con los cálculos ya resueltos, listo para inyectar en el prompt.

    Ejemplo de línea generada: "Porcentaje de Crecimiento (2019-2025): 22.11%".
    """
    lines = []
    for item in summary:
        country = item["country"]
        base_year = item["base_year"]
        end_year = item["end_year"]
        growth_pct = item["growth_pct"]

        if item.get("error"):
            lines.append(f"- {country}: no soportado por la API ({item['error']}).")
            continue

        if growth_pct is None:
            lines.append(f"- {country}: crecimiento no disponible (datos insuficientes o base en cero).")
            continue

        base_value = item["base_value"]
        end_value = item["end_value"]
        lines.append(
            f"- {country}: consumo {base_year} = {base_value:,.0f} | "
            f"consumo proyectado {end_year} = {end_value:,.0f} | "
            f"Porcentaje de Crecimiento ({base_year}-{end_year}): {growth_pct:.2f}%"
        )
    return "\n".join(lines)


def analyze_market(countries: list[str], end_year: int = 2025) -> str:
    """Obtiene los datos de la API y genera el análisis con el cliente oficial de Hugging Face."""
    forecast_data = fetch_coffee_forecast(countries, end_year)

    if "error" in forecast_data:
        return f"❌ No se pudo realizar el análisis: {forecast_data['error']}"

    # 1. Calcular el % de crecimiento en Python -- ya resuelto, el LLM solo lo redacta.
    growth_summary = _compute_growth_summary(forecast_data, end_year)
    growth_summary_text = _format_growth_summary(growth_summary)

    system_prompt = (
        "Eres el Agente Senior de Innovación e Inteligencia de Mercado de Café.\n"
        "Analiza la información de demanda (histórica y proyectada) entregada por la API de ML.\n"
        "Instrucciones:\n"
        "- Explica la tendencia de consumo de cada país.\n"
        "- Ya se te entrega, PRE-CALCULADO, el porcentaje de crecimiento entre el último año "
        f"histórico ({HISTORICAL_END_YEAR}) y el proyectado ({end_year}) para cada país (ver bloque "
        "'Porcentajes de crecimiento (pre-calculados)' más abajo). NO recalcules estos valores ni "
        "muestres el procedimiento aritmético: cópialos tal cual en tu respuesta.\n"
        "- Propón 2 oportunidades de innovación en mercado y estrategia de precios.\n"
        "- Si algún país aparece marcado como 'no soportado por la API' en el bloque de "
        "porcentajes de crecimiento, NO inventes tendencias ni cifras para ese país: menciónalo "
        "brevemente en una línea aparte (p.ej. 'Nota: no se encontró información para X en el "
        "dataset') y continúa el análisis normalmente con el resto de países.\n"
        "- Responde de forma clara, profesional y estructurada con viñetas.\n"
        "- FORMATO DE SALIDA: texto plano únicamente. Prohibido usar notación LaTeX o de fórmulas "
        "matemáticas (nada de \\text{}, \\frac{}, símbolos $...$, \\times, etc.). Escribe cada "
        "cálculo ya resuelto, con el resultado final directamente en el texto, por ejemplo: "
        "\"Porcentaje de Crecimiento (2019-2025): 22.11%\"."
    )

    user_prompt = (
        f"Aquí están los datos de la API:\n\n{forecast_data}\n\n"
        f"Porcentajes de crecimiento (pre-calculados, {HISTORICAL_END_YEAR}-{end_year}):\n"
        f"{growth_summary_text}"
    )

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
        raw_text = response.choices[0].message.content
    except Exception:
        # Fallback a text_generation si el modelo es legacy
        prompt = f"{system_prompt}\n\nDatos:\n{user_prompt}\n\nAnálisis:"
        raw_text = client.text_generation(prompt, model=MODEL_NAME, max_new_tokens=1500)

    # 2. Red de seguridad: sanear cualquier resto de LaTeX que el modelo haya insertado.
    return _strip_latex_artifacts(raw_text)
