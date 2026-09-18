
# Proyecto: Inteligencia de Mercado de Café

## Stack

- Frontend: Streamlit (`app.py`)
- Backend/API: FastAPI (`api/main.py`)
- Modelo ML: LightGBM (`data/04_models/lgbm_model.joblib`)
- Datos: Parquet (`data_long.parquet`, `data_features.parquet`)
- Agente LLM: LangChain/OpenAI (`src/agent/bot.py`)

## Reglas de Arquitectura

- Frontend desacoplado: Streamlit solo consume endpoints de FastAPI.
- Backend: microservicio con endpoints `/predict` y `/analyze`.
- Inferencia ML: homologar categóricas con `.astype('category')`.
- Agente LLM: Structured Outputs con Pydantic (`schemas.py`).
- Gráficas: Subplots con colores definidos (azul entrenamiento, verde reales, negro predicciones históricas, rojo proyecciones futuras).
- Salidas: JSON limpio, floats redondeados a 2 decimales, sin LaTeX roto.

## Estilo de Código

- Python 3.10+
- Tipado estricto con `typing`
- Logs con `logging`
- Manejo de errores con `HTTPException` en FastAPI
