from __future__ import annotations

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Payload de entrada para solicitar la proyección de demanda de múltiples países."""

    countries: list[str] = Field(
        ...,
        description="Lista de nombres exactos de países (ej. ['Colombia', 'Brazil'])",
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
    """Estructura de respuesta individual por país."""

    country: str
    coffee_type: str
    historical_end_year: int = 2019
    forecast_end_year: int
    data: list[DataPoint]


class MultiForecastResponse(BaseModel):
    """Estructura de respuesta global conteniendo todos los países solicitados."""

    results: list[CountryForecastResponse]