from __future__ import annotations

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Payload de entrada para solicitar la proyección de demanda de múltiples países."""

    countries: list[str] = Field(
        ...,
        min_length=1,
        description="Lista de nombres exactos de países, tal como aparecen en el dataset "
        "(ej. ['Colombia', 'Brazil']). Usa GET /countries para obtener la lista completa y "
        "evitar errores de nombre (p.ej. 'Viet Nam', no 'Vietnam').",
        example=["Colombia", "Brazil"],
    )
    end_year: int = Field(
        default=2025,
        ge=2020,
        le=2035,
        description="Año límite de la proyección (entre 2020 y 2035)",
        example=2025,
    )


class DataPoint(BaseModel):
    """Representa un punto en la serie temporal (histórico o proyectado)."""

    year: int = Field(..., description="Año del dato")
    consumption: float = Field(..., description="Consumo de café (60kg bags)")
    is_forecast: bool = Field(
        ..., description="True si es valor predicho, False si es histórico real"
    )


class CountryForecastResponse(BaseModel):
    """Estructura de respuesta individual por país.

    Si el país solicitado no existe en el registro histórico (nombre mal
    escrito, o país fuera del dataset), esta entrada NO se omite ni hace
    fallar la respuesta completa: se devuelve con `error` describiendo el
    problema y el resto de campos en su valor por defecto (`data` vacío).
    Así, un país inválido en un lote de varios países no le impide a la UI
    ni al agente LLM procesar los países que sí son válidos.
    """

    country: str
    coffee_type: str | None = None
    historical_end_year: int | None = None
    forecast_end_year: int | None = None
    data: list[DataPoint] = Field(default_factory=list)
    error: str | None = Field(
        default=None,
        description="Mensaje explicando por qué no se pudo generar la proyección para este "
        "país (p.ej. nombre no encontrado en el dataset). None si la proyección fue exitosa.",
    )


class MultiForecastResponse(BaseModel):
    """Estructura de respuesta global conteniendo todos los países solicitados."""

    results: list[CountryForecastResponse]


class AvailableCountriesResponse(BaseModel):
    """Lista de países que la API puede proyectar, tal como aparecen en el dataset."""

    countries: list[str]