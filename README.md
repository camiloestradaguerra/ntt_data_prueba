# ☕ Inteligencia de Mercado de Café

Pipeline MLOps completo para el pronóstico de consumo/demanda de café por
país: ETL → Feature Engineering → Entrenamiento → Evaluación → API → UI,
con un agente conversacional (LLM) que interpreta las proyecciones, y
túneles ngrok automáticos para compartir la API y la UI con terceros sin
desplegar nada.

```
Stack
-----
Frontend .......... Streamlit           (app.py)
Backend/API ........ FastAPI             (src/api/)
Modelo ML .......... LightGBM            (src/pipelines/3_training/)
Agente LLM ......... Hugging Face Hub    (src/agent/)  — InferenceClient directo
Acceso público ..... ngrok / pyngrok     (scripts/run_*_tunnel.py)
CI/CD .............. GitHub Actions      (.github/workflows/mlops_pipeline.yml)
```

> **Nota sobre el agente LLM:** el diseño original contemplaba
> LangChain/LangGraph + OpenAI; la implementación actual usa el cliente
> oficial `huggingface_hub.InferenceClient` directamente (ver
> `src/agent/bot.py`), sin esas dependencias. Este README documenta el
> proyecto **tal como está implementado**, no el diseño original.

---

## Tabla de contenidos

1. [Flujo MLOps](#1-flujo-mlops-etl--features--training--evaluation--api--ui)
2. [Arquitectura de servicios (API + UI + Agente)](#2-arquitectura-de-servicios-api--ui--agente)
3. [Instalación](#3-instalación)
4. [Cómo ejecutar cada paso](#4-cómo-ejecutar-cada-paso)
5. [ngrok: compartir la app con terceros](#5-ngrok-compartir-la-app-con-terceros)
6. [CI/CD con GitHub Actions](#6-cicd-con-github-actions)
7. [Estructura del repositorio](#7-estructura-del-repositorio)
8. [Utilidades de QA](#8-utilidades-de-qa)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Flujo MLOps: ETL → Features → Training → Evaluation → API → UI

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
│  1. ETL     │──▶│ 2. Features │──▶│ 3. Training │──▶│ 4. Evaluation │
│ (wide→long) │   │ (lags, cat.)│   │ (eval+prod) │   │ (holdout)     │
└─────────────┘   └─────────────┘   └──────┬──────┘   └──────────────┘
                                            │ modelo PROD
                                            ▼
                                   ┌─────────────────┐        ┌───────────────┐
                                   │  5. API FastAPI  │◀──────▶│ 6. UI Streamlit│
                                   │ (src/api/)       │        │ (app.py)       │
                                   └────────┬─────────┘        └───────────────┘
                                            ▲
                                            │ /predict
                                   ┌────────┴─────────┐
                                   │ Agente LLM (CLI)  │
                                   │ (src/agent/)       │
                                   └────────────────────┘
```

| Paso | Script | Entrada | Salida |
|---|---|---|---|
| **1. ETL** | `src/pipelines/1_data_etl/main.py` | `data/01_raw/coffee_db.parquet` (panel *wide*, columnas por año) | `data/02_processed/data_long.parquet` (panel *long*: una fila por país/año) |
| **2. Features** | `src/pipelines/2_feature_engineering/main.py` | panel *long* | `data/03_features/data_features.parquet` + artefactos de codificación (`categorical_encoder.joblib`, `encoder_mappings.json`) en `data/03_features/artifacts/` |
| **3. Training** | `src/pipelines/3_training/main.py` | features | **dos** modelos LightGBM (ver abajo) + `data/04_models/artifacts/{best_params,metrics,model_registry}.json` |
| **4. Evaluation** | `src/pipelines/4_evaluation/main.py` | modelo *eval* + features | `data/05_evaluation/{evaluation_metrics.json, evaluation_predictions.parquet}` |
| **5. API** | `src/api/main.py` | modelo *prod* | servicio HTTP (`/health`, `/countries`, `/predict`) |
| **6. UI** | `app.py` | API | dashboard interactivo (Streamlit) |

### Por qué existen dos modelos (`lgbm_eval.joblib` / `lgbm_prod.joblib`)

El paso 3 entrena y serializa **dos artefactos distintos** a partir de la
misma búsqueda de hiperparámetros (Optuna), pero sobre ventanas de datos
diferentes, para no mezclar "qué tan bueno es el modelo" con "el mejor
modelo que puedo dar a producción":

- **`lgbm_eval.joblib`** (`model_role="eval"`) — entrenado **solo** con
  1990–2015. Lo consume *exclusivamente* el paso 4 (`evaluate`), contra el
  holdout honesto 2016–2019 que el modelo nunca vio. Es la métrica de
  validación real del proyecto.
- **`lgbm_prod.joblib`** (`model_role="prod"`) — reentrenado con **todo**
  el historial disponible (1990–2019). Lo consume *exclusivamente* la API
  para inferencia en producción — usar el mayor historial posible da
  mejores proyecciones futuras, pero por eso mismo **nunca** debe evaluarse
  contra 2016–2019 (esos años ya están dentro de su propio entrenamiento:
  sería *data leakage* y las métricas saldrían artificialmente optimistas).

`src/pipelines/4_evaluation/main.py` rechaza explícitamente cualquier
artefacto cuyo `model_role` no sea `"eval"`, precisamente para que este
error no pueda cometerse por accidente. El detalle completo vive en
`data/04_models/artifacts/model_registry.json` (generado por `train`).

---

## 2. Arquitectura de servicios (API + UI + Agente)

- **API (FastAPI, `src/api/`)** — único punto de acceso al modelo *prod*.
  - `GET /health` — chequeo de vida.
  - `GET /countries` — lista todos los países disponibles en
    `data_long.parquet` **tal como aparecen en el dataset** (p. ej.
    `"Viet Nam"`, no `"Vietnam"`) — fuente única de verdad que consumen
    tanto la UI (para poblar el selector) como el agente.
  - `POST /predict` — proyecta demanda para uno o más países. Un país
    inválido **nunca** hace fallar el lote completo con `404`: se devuelve
    dentro de la respuesta, por país, con su propio campo `error` (y una
    sugerencia por similitud de nombre cuando aplica) — así un solo nombre
    mal escrito no bloquea el resto de países válidos del lote.
- **UI (Streamlit, `app.py`)** — consume únicamente la API (nunca el
  modelo directamente para la proyección futura), y grafica 4 series por
  país: azul (entrenamiento 1990–2015), negro (predicciones del modelo en
  validación 2016–2019), verde (valores reales de validación) y rojo
  (proyección futura ≥2020, desde el modelo *prod* vía la API).
- **Agente LLM (`src/agent/`)** — CLI conversacional (`cli_chat.py`) que le
  pide datos a la API (`tools.py::fetch_coffee_forecast`) y usa
  `huggingface_hub.InferenceClient` (`bot.py`) para resumir tendencias en
  lenguaje natural. Si un país del lote no es soportado por la API, el
  agente lo señala aparte en su respuesta en vez de inventar cifras para
  él. Configuración vía `.env`: `HUGGINGFACE_API_KEY`, `API_URL`
  (por defecto `http://127.0.0.1:8000`), `MODEL_NAME`.

---

## 3. Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Crea un archivo `.env` en la raíz (usado por el agente y, opcionalmente,
por el túnel ngrok):

```env
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxx
API_URL=http://127.0.0.1:8000
MODEL_NAME=deepseek-ai/DeepSeek-R1-0528
NGROK_AUTHTOKEN=                 # opcional -- ver sección 5
```

**ngrok** (opcional pero recomendado para acceso público): crea una cuenta
gratuita en <https://dashboard.ngrok.com/signup> y autentica una sola vez
en tu máquina —

```bash
ngrok config add-authtoken <tu-token>
```

— o exporta `NGROK_AUTHTOKEN` (en el `.env` o en el shell) antes de correr
`make run-api` / `make run-ui`. Sin esto, ambos comandos igual funcionan,
solo que quedan disponibles únicamente en `localhost` (con un aviso
explicándolo, nunca un error que detenga el arranque).

---

## 4. Cómo ejecutar cada paso

Todos los targets del `Makefile` funcionan igual en **Windows** (cmd.exe /
PowerShell) y **Linux/Mac** (bash/zsh) — ver el bloque de comentarios al
inicio del `Makefile` para el detalle de cómo se resuelve la diferencia de
sintaxis de variables de entorno entre plataformas.

### Pipeline batch (una sola vez, o tras cambiar los datos raw)

```bash
make all        # setup + etl + features + train + evaluate, en orden
```

O paso a paso: `make setup`, `make etl`, `make features`, `make train`,
`make evaluate`.

### Servicios (dos terminales en paralelo)

```bash
# Terminal 1
make run-api    # API en http://localhost:8000 + URL pública de ngrok

# Terminal 2
make run-ui     # UI en http://localhost:8501 + URL pública de ngrok
```

Ambos comandos imprimen su URL pública de ngrok en consola apenas
conectan, y el túnel se cierra solo al detener el servicio (`Ctrl+C`) —
nunca queda un túnel huérfano corriendo. Ver `run-api-dev` en el Makefile
si necesitas autorecarga de código (`--reload`) sin túnel, mientras
iteras sobre `src/api/`.

### Agente conversacional

```bash
make run-agent  # requiere `make run-api` corriendo en paralelo
```

### Otros

```bash
make test-agent   # smoke test: el módulo del agente importa sin errores
make clean        # borra data/02_processed en adelante (conserva 01_raw)
make tunnel       # túnel ngrok manual (CLI de ngrok) hacia el puerto de la API
make tunnel-ui    # túnel ngrok manual hacia el puerto de la UI
```

---

## 5. ngrok: compartir la app con terceros

`make run-api` y `make run-ui` abren su propio túnel ngrok automáticamente
(vía `pyngrok`, gestionado por `scripts/run_api_tunnel.py` /
`scripts/run_ui_tunnel.py`). Para compartir la aplicación con alguien
fuera de tu red:

1. Corre `make run-api` en una terminal — copia la URL pública que imprime
   (algo como `https://xxxx.ngrok-free.app`).
2. Corre `make run-ui` en otra terminal — copia su propia URL pública.
3. Comparte **la URL de la UI** con la otra persona: la UI, corriendo en
   tu máquina, sigue llamando a la API en `http://localhost:8000` (no
   necesita saber la URL pública de la API para eso). La URL pública de la
   API solo es necesaria si un tercero va a llamarla directamente (por
   ejemplo, para integrar el agente LLM contra tu API desde otra máquina).
4. Ambos túneles viven mientras sus respectivos procesos sigan corriendo
   en tu máquina — si cierras la terminal o presionas `Ctrl+C`, la URL
   pública deja de funcionar de inmediato.

El plan gratuito de ngrok genera una URL nueva cada vez que se abre el
túnel (a menos que tengas un dominio reservado en tu cuenta) — si
necesitas una URL estable entre reinicios, configúrala en tu cuenta de
ngrok y pásala vía las opciones de `pyngrok`/`ngrok.yml` (fuera del alcance
de este README).

---

## 6. CI/CD con GitHub Actions

Workflow: `.github/workflows/mlops_pipeline.yml`. Se dispara en `push` y
`pull_request` a `main`/`master`, y manualmente (`workflow_dispatch`).

**Qué hace, en orden:**

1. Instala dependencias (`requirements.txt`) y `rclone`.
2. Descarga `coffee_db.parquet` desde Google Drive (credenciales vía
   secrets `GDRIVE_RCLONE_CONF` / `GDRIVE_ROOT_FOLDER_ID`).
3. Corre el pipeline batch completo: `make all`.
4. Valida, con *smoke tests* de import (sin levantar ningún servidor):
   que la API cargue sus artefactos (`from src.api.main import app`), que
   el módulo del agente importe (`make test-agent`, con un `.env`
   temporal generado solo para el job), y que `app.py` (Streamlit)
   importe sin errores.
5. Solo si el push fue a `main`/`master`: sincroniza los artefactos
   generados (`data/02_processed` a `data/05_evaluation`) de vuelta a
   Google Drive.
6. Escribe un resumen de alcance en la pestaña *Summary* de la corrida.

**Qué *no* hace — y por qué:** el workflow **no** ejecuta `make run-api`
ni `make run-ui`, y **no** abre un túnel ngrok. Un runner de GitHub
Actions es efímero (se destruye al terminar el job) y no acepta tráfico
entrante desde internet, así que un servidor o un túnel dejado corriendo
ahí quedaría inalcanzable de inmediato — no existe una "URL pública del
CI" que este pipeline pueda producir. En otras palabras: **este CI/CD
valida y prepara los artefactos; no despliega ni expone nada.**

Si necesitas una demo pública persistente:
- Para uso puntual/demos: corre `make run-api` / `make run-ui` en tu
  propia máquina (túnel ngrok automático, ver secciones 4 y 5).
- Para un despliegue persistente real: agrega un job de *deploy* hacia un
  servicio que sí acepte tráfico entrante (una VM, un PaaS como
  Render/Railway/Fly.io, un contenedor en la nube, etc.) — no configurado
  en este repositorio.

---

## 7. Estructura del repositorio

```
.
├── app.py                          # UI Streamlit
├── Makefile                        # todos los comandos del proyecto
├── requirements.txt
├── .env                             # credenciales locales (no versionado)
├── .github/workflows/
│   └── mlops_pipeline.yml          # CI/CD (ver sección 6)
├── scripts/
│   ├── _tunnel_common.py           # lógica compartida de túnel ngrok + señales
│   ├── run_api_tunnel.py           # `make run-api`
│   ├── run_ui_tunnel.py            # `make run-ui`
│   ├── check_predictions.py        # QA: grilla top-3/bottom-3 países (ver sección 8)
│   └── verify_predictions.py       # QA: verificación de un país puntual
├── src/
│   ├── api/                         # FastAPI: main.py, schemas.py, services.py
│   ├── agent/                       # agente LLM: bot.py, tools.py, cli_chat.py
│   └── pipelines/
│       ├── 1_data_etl/
│       ├── 2_feature_engineering/
│       ├── 3_training/
│       └── 4_evaluation/
├── data/                            # generado por el pipeline (gitignored)
│   ├── 01_raw/                      # dataset fuente (el único que no se regenera)
│   ├── 02_processed/
│   ├── 03_features/
│   ├── 04_models/                   # lgbm_eval.joblib + lgbm_prod.joblib
│   └── 05_evaluation/
└── reports/                          # salidas de scripts/*.py de QA (gitignored)
```

---

## 8. Utilidades de QA

Estos scripts **no** forman parte del pipeline oficial (pasos 1–4); son
herramientas independientes de control de calidad sobre el modelo ya
entrenado:

- **`scripts/verify_predictions.py --country <País>`** — grafica, para un
  país puntual, entrenamiento (azul), validación real (verde) y predicha
  (negro), y proyección futura (rojo) usando el mismo bucle de inferencia
  recursiva que la API.
- **`scripts/check_predictions.py [--forecast_end_year 2025] [--metric WAPE]`**
  — calcula métricas de validación (2016–2019) para **todos** los países,
  selecciona los 3 mejores y 3 peores por WAPE (métrica de error relativo,
  comparable entre países con escalas de consumo muy distintas), y grafica
  esos 6 en una grilla 2×3 con las mismas 4 series de colores.

---

## 9. Troubleshooting

| Síntoma | Causa probable | Solución |
|---|---|---|
| `make run-ui` falla en Windows con `'STREAMLIT_PORT' no se reconoce...` | Versión antigua del Makefile con sintaxis POSIX | Usa el Makefile actual — ya resuelve la sintaxis de variable de entorno según `$(OS)` (ver comentarios al inicio del archivo) |
| La UI muestra "Sin proyección futura disponible" para un país | Nombre de país fuera del dataset o mal escrito | Usa el selector de la UI (poblado desde `GET /countries`, la fuente única de verdad de nombres) en vez de escribir el país a mano |
| `make run-api` / `make run-ui` arrancan pero sin URL pública | `pyngrok` no instalado, o ngrok sin autenticar / sin red | Revisa el aviso impreso en consola; instala `pyngrok` (`pip install -r requirements.txt`) y/o corre `ngrok config add-authtoken <token>` |
| El agente lanza `ValueError` sobre el token de Hugging Face | Falta `HUGGINGFACE_API_KEY` en `.env` | Agrega la variable al `.env` (ver sección 3) |
| CI/CD verde pero no hay URL pública que compartir | Comportamiento esperado (ver sección 6) | Corre `make run-api` / `make run-ui` en tu propia máquina para obtener una URL pública |
