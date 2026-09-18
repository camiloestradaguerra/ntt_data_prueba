PYTHON := python
RAW_DATA := data/01_raw/coffee_db.parquet
PROCESSED_DATA := data/02_processed/data_long.parquet
FEATURES_DATA := data/03_features/data_features.parquet
FEATURE_ARTIFACTS := data/03_features/artifacts
STREAMLIT_PORT := 8501
API_PORT := 8000

# --- Exportar puertos como variable de entorno: sintaxis distinta según el SO ---
# `make` en Windows (fuera de Git Bash/WSL) usa cmd.exe como SHELL por
# defecto, no sh/bash, así que la sintaxis POSIX `VAR=valor comando` (que
# cmd.exe no entiende -- interpreta "STREAMLIT_PORT=8501" como el NOMBRE de
# un comando a ejecutar, de ahí el error "no se reconoce como un comando
# interno o externo") no sirve ahí. `$(OS)` es una variable de entorno que
# el propio Windows define como "Windows_NT" (en Linux/Mac no existe, así
# que el `ifeq` cae al `else`), y la usamos para elegir en tiempo de
# lectura del Makefile la sintaxis correcta de cada plataforma:
#   - Windows (cmd.exe): `set VAR=valor && comando` -- `set` sin `&&` solo
#     definiría la variable para ese cmd.exe, y como make abre un cmd.exe
#     NUEVO por cada línea de receta, el proceso del comando nunca la vería;
#     encadenar con `&&` garantiza que la variable ya esté definida en el
#     mismo cmd.exe que ejecuta a continuación el comando.
#   - Linux/Mac (sh/bash): `VAR=valor comando` -- exporta la variable solo
#     para ese comando puntual, sin ensuciar el resto de la sesión de shell.
# Esta variable de entorno es en realidad un mecanismo de RESPALDO: tanto
# `run-ui` como `run-api` pasan el puerto TAMBIÉN como argumento de línea de
# comandos (ver más abajo), que es la vía que de verdad usan los scripts
# (`scripts/run_ui_tunnel.py` / `scripts/run_api_tunnel.py`) por ser idéntica
# en cmd.exe, PowerShell, bash o zsh. Se mantiene la variable de entorno por
# si alguien invoca esos scripts directamente sin pasar por `make`.
ifeq ($(OS),Windows_NT)
SET_STREAMLIT_PORT := set STREAMLIT_PORT=$(STREAMLIT_PORT) &&
SET_API_PORT := set API_PORT=$(API_PORT) &&
else
SET_STREAMLIT_PORT := STREAMLIT_PORT=$(STREAMLIT_PORT)
SET_API_PORT := API_PORT=$(API_PORT)
endif

# --- Artefactos de modelo: DOS roles que NUNCA se intercambian ---
# MODEL_EVAL_PATH  -> entrenado solo con 1990-2015. Lo consume UNICAMENTE
#                      `evaluate` (Paso 4), para métricas de validación honestas.
# MODEL_PROD_PATH  -> re-entrenado con todo el historial 1990-2019. Lo consume
#                      UNICAMENTE la API (src/api/services.py) para inferencia.
# Ver data/04_models/artifacts/model_registry.json (generado por `train`)
# para el detalle de por qué existen dos artefactos separados.
MODEL_EVAL_PATH := data/04_models/lgbm_eval.joblib
MODEL_PROD_PATH := data/04_models/lgbm_prod.joblib
MODEL_ARTIFACTS := data/04_models/artifacts
EVAL_DIR := data/05_evaluation

.PHONY: all setup etl features train evaluate run-api run-api-dev run-agent run-ui test-agent tunnel tunnel-ui clean

# Corre el pipeline MLOps completo de punta a punta, en orden: crea las
# carpetas de datos, transforma el raw a long panel, genera features,
# entrena AMBOS modelos (eval + prod) y evalúa el modelo eval contra el
# holdout honesto. Es exactamente lo que ejecuta el workflow de CI/CD
# (`.github/workflows/mlops_pipeline.yml`, paso "Ejecutar Pipeline MLOps
# Completo") tras descargar `coffee_db.parquet`. No incluye `run-api` ni
# `run-ui`: esos son procesos de servicio de larga duración, no pasos de un
# pipeline batch.
all: setup etl features train evaluate

# Crea el árbol de carpetas de `data/` si no existe todavía (idempotente:
# `exist_ok=True`). Se corre antes de `etl` porque `1_data_etl/main.py`
# escribe en `data/02_processed/` y falla si la carpeta no existe.
setup:
	@$(PYTHON) -c "import os; [os.makedirs(d, exist_ok=True) for d in ['data/01_raw', 'data/02_processed', 'data/03_features/artifacts', 'data/04_models/artifacts', 'data/05_evaluation']]"

# Paso 1/4 del pipeline: lee el panel "wide" (`coffee_db.parquet`, columnas
# tipo "1990", "1991", ...) y lo transforma a formato "long" estandarizado
# (una fila por país/año) en `data/02_processed/data_long.parquet`.
etl:
	$(PYTHON) src/pipelines/1_data_etl/main.py --input_path $(RAW_DATA) --output_path $(PROCESSED_DATA)

# Paso 2/4: genera las features de series de tiempo (lags, medias móviles,
# codificación de la categórica "Country") a partir del long panel, y
# serializa los artefactos de codificación (`categorical_encoder.joblib`,
# `encoder_mappings.json`) que luego reutilizan `train`, `evaluate` y la
# API -- para que el mismo país se codifique siempre con el mismo entero,
# sin reajustar el encoder en cada corrida.
features:
	$(PYTHON) src/pipelines/2_feature_engineering/main.py --input_path $(PROCESSED_DATA) --output_path $(FEATURES_DATA) --artifacts_dir $(FEATURE_ARTIFACTS)

# Paso 3/4: entrena y serializa AMBOS modelos en una sola corrida (comparten
# los mismos hiperparámetros, elegidos una vez vía Optuna, pero se ajustan
# sobre ventanas de datos distintas -- ver el bloque de comentarios sobre
# MODEL_EVAL_PATH / MODEL_PROD_PATH arriba). `--n_trials 15` controla cuántas
# combinaciones de hiperparámetros prueba Optuna antes de reentrenar ambos
# modelos con la mejor combinación encontrada.
train:
	$(PYTHON) src/pipelines/3_training/main.py --input_path $(FEATURES_DATA) --input_artifacts_dir $(FEATURE_ARTIFACTS) --output_eval_model_path $(MODEL_EVAL_PATH) --output_prod_model_path $(MODEL_PROD_PATH) --artifacts_dir $(MODEL_ARTIFACTS) --n_trials 15

# Paso 4/4: evalúa EXCLUSIVAMENTE el modelo eval (holdout honesto 2016-2019,
# nunca visto durante su entrenamiento) y escribe métricas + predicciones en
# `data/05_evaluation/`. NUNCA apuntar esto a MODEL_PROD_PATH: el modelo prod
# fue entrenado con esos mismos años, así que "evaluarlo" ahí daría métricas
# artificialmente optimistas por data leakage -- `4_evaluation/main.py`
# rechaza explícitamente un modelo con `model_role != "eval"` (ver
# `DataLoader.load_model`).
evaluate:
	$(PYTHON) src/pipelines/4_evaluation/main.py --input_path $(FEATURES_DATA) --eval_model_path $(MODEL_EVAL_PATH) --input_artifacts_dir $(FEATURE_ARTIFACTS) --output_dir $(EVAL_DIR)

# --- run-api: FastAPI + túnel ngrok automático ---
# Lanza la API (`uvicorn src.api.main:app`) en el puerto $(API_PORT) y, en el
# mismo proceso, abre un túnel ngrok hacia ese puerto (vía
# `scripts/run_api_tunnel.py` + `pyngrok`), imprimiendo la URL pública en
# consola apenas conecta -- útil para que la UI, el agente LLM o un tercero
# llamen a /predict, /countries o /health desde fuera de esta máquina. El
# túnel vive exactamente mientras la API esté corriendo: al detenerla
# (Ctrl+C) se cierra solo, nunca queda un túnel huérfano. Sin `ngrok`
# autenticado (`ngrok config add-authtoken <token>`, o exportar
# NGROK_AUTHTOKEN antes de correr esto) la API arranca igual, solo en
# localhost, con un aviso explicándolo -- un problema del túnel nunca impide
# que la API arranque.
# El puerto se pasa DOBLE, igual que en `run-ui`: por variable de entorno
# (ver $(SET_API_PORT) arriba) Y como argumento de línea de comandos -- ver
# `scripts/run_api_tunnel.py`, que prioriza el argumento por ser a prueba de
# shell (cmd.exe, PowerShell, bash o zsh, sin sintaxis especial).
run-api:
	$(SET_API_PORT) $(PYTHON) scripts/run_api_tunnel.py $(API_PORT)

# Alternativa de desarrollo activo: uvicorn directo con `--reload`
# (autorecarga en cada cambio de código), SIN túnel ngrok. `run-api` (arriba)
# no usa `--reload` a propósito: el wrapper de túnel necesita un único
# proceso hijo que pueda terminar de forma predecible, y `--reload` lanza un
# proceso "reloader" adicional que complica ese cierre ordenado. Usa este
# target cuando estés iterando sobre `src/api/` localmente y no necesites la
# URL pública.
run-api-dev:
	uvicorn src.api.main:app --reload --port $(API_PORT)

# CLI conversacional del agente LLM (Hugging Face InferenceClient + la API
# de pronóstico). Requiere `make run-api` corriendo en paralelo (el agente le
# pide los datos de proyección a la API) y las variables de `.env`
# (HUGGINGFACE_API_KEY, API_URL, MODEL_NAME -- ver README.md).
run-agent:
	$(PYTHON) -m src.agent.cli_chat

# --- run-ui: Streamlit + túnel ngrok automático ---
# Lanza Streamlit en el puerto $(STREAMLIT_PORT) y, en el mismo proceso,
# abre un túnel ngrok hacia ese puerto (vía `scripts/run_ui_tunnel.py` +
# `pyngrok`), imprimiendo la URL pública en consola apenas conecta. El túnel
# vive exactamente mientras Streamlit esté corriendo: al parar la UI
# (Ctrl+C) se cierra solo, no queda un túnel huérfano. Requiere `ngrok`
# autenticado una sola vez en esta máquina (`ngrok config add-authtoken
# <token>`, o exportar NGROK_AUTHTOKEN antes de correr `make run-ui`) -- sin
# eso, la UI arranca igual pero solo en localhost, con un aviso explicándolo
# (nunca falla el arranque de Streamlit por un problema del túnel).
# NO requiere tocar services.py / app.py / 4_evaluation: es un wrapper de
# proceso alrededor de `streamlit run app.py`, la app en sí no cambia.
# El puerto se pasa DOBLE, por variable de entorno (ver $(SET_STREAMLIT_PORT)
# arriba, con su propia sintaxis por SO) Y como argumento de línea de
# comandos -- ver `scripts/run_ui_tunnel.py`, que acepta ambos y usa el
# argumento si está presente. El argumento es la vía a prueba de shell
# (funciona igual en cmd.exe, PowerShell, bash o zsh, sin sintaxis especial);
# la variable de entorno se mantiene por si se llama al script directamente
# fuera de `make` (p.ej. `STREAMLIT_PORT=8502 python scripts/run_ui_tunnel.py`).
run-ui:
	@echo "ℹ️  La UI consume la API en http://localhost:$(API_PORT) para la línea roja de proyección futura."
	@echo "   Si no la iniciaste ya, abre otra terminal y corre: make run-api"
	$(SET_STREAMLIT_PORT) $(PYTHON) scripts/run_ui_tunnel.py $(STREAMLIT_PORT)

# Smoke test rápido (sin red ni servidor): solo confirma que el módulo del
# agente y sus dependencias importan sin errores -- lo usa el CI/CD
# (`.github/workflows/mlops_pipeline.yml`) para validar que `src/agent/bot.py`
# no tiene errores de sintaxis/import antes de dar el pipeline por bueno.
test-agent:
	$(PYTHON) -c "from src.agent.bot import analyze_market; print('✅ Módulo de agente e InferenceClient importados correctamente')"

# Túneles ngrok manuales (binario `ngrok` instalado y autenticado en esta
# máquina) hacia un servicio que YA esté corriendo en otra terminal, sin el
# ciclo de vida automático de `run-api`/`run-ui` (aquí el túnel NO se cierra
# solo si detienes el servicio; hay que cerrarlo aparte con Ctrl+C). Sirven
# para exponer la API o la UI usando la CLI de ngrok directamente en vez de
# `pyngrok`, por ejemplo si `pyngrok` no está instalado. La vía recomendada
# para el día a día sigue siendo `make run-api` / `make run-ui`, que ya
# incluyen su propio túnel administrado y bien cerrado.
tunnel:
	ngrok http $(API_PORT)

tunnel-ui:
	ngrok http $(STREAMLIT_PORT)

# Borra todo lo generado por el pipeline (`data/02_processed/` en adelante),
# preservando `data/01_raw/` (el dataset fuente no se regenera solo). Útil
# para forzar una corrida limpia de `make all` desde cero.
clean:
	@$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) if p.is_dir() else p.unlink() for folder in pathlib.Path('data').glob('*') if folder.name != '01_raw' for p in folder.glob('*')]"
