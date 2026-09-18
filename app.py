import streamlit as st
import pandas as pd
import requests
import joblib
import pickle
import sys
import os
import importlib.util
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --- INTERCEPTOR DE DESERIALIZACIÓN PARA `lgbm_eval.joblib` (respaldo local) ---
# `lgbm_eval.joblib` se serializó cuando `3_training/main.py` corría como
# `__main__`, así que pickle busca la clase `DemandForecastModel` en el
# módulo "__main__" al deserializar. Streamlit ejecuta app.py de forma que
# `sys.modules["__main__"]` sigue siendo el bootstrap de `streamlit run`, no
# app.py, así que resolvemos la clase interceptando el `find_class` del
# Unpickler que usa `joblib.load` directamente (independiente de qué módulo
# sea "__main__" en cada contexto de ejecución).
#
# NOTA: el modelo de PRODUCCIÓN (`lgbm_prod.joblib`) y sus artefactos
# asociados (`CategoricalEncoder`, `TimeSeriesFeatureEngineer`) YA NO se
# cargan aquí -- la UI obtiene la proyección futura (≥2020) exclusivamente
# de la API (ver `obtener_prediccion_futura`), así que este interceptor
# solo necesita cubrir `DemandForecastModel`, requerido por
# `lgbm_eval.joblib` (línea verde).
try:
    _training_path = ROOT_DIR / "src" / "pipelines" / "3_training" / "main.py"
    _training_spec = importlib.util.spec_from_file_location("training_main_module", _training_path)
    _training_module = importlib.util.module_from_spec(_training_spec)
    # IMPORTANTE: registrar el módulo en sys.modules ANTES de ejecutarlo.
    # `3_training/main.py` define una `@dataclass`, y el decorador resuelve
    # anotaciones de tipo buscando `sys.modules[cls.__module__]` durante la
    # propia ejecución de la clase; si el módulo todavía no está registrado
    # ahí (como ocurre si se omite este paso), esa búsqueda devuelve `None`
    # y la creación de la dataclass falla con
    # "'NoneType' object has no attribute '__dict__'" -- exactamente el
    # error "'NoneType' object has no attribute 'dict'" reportado en la UI.
    sys.modules[_training_spec.name] = _training_module
    _training_spec.loader.exec_module(_training_module)

    DemandForecastModel = _training_module.DemandForecastModel

    _PICKLE_CLASS_OVERRIDES = {
        "DemandForecastModel": DemandForecastModel,
    }

    # `joblib.load` usa su propio Unpickler (`joblib.numpy_pickle.NumpyUnpickler`),
    # una subclase de Python normal (por lo tanto parcheable) que NO
    # sobrescribe `find_class`. NO se parchea `pickle.Unpickler` directamente:
    # en CPython esa es la implementación acelerada en C (`_pickle.Unpickler`),
    # un tipo inmutable, y asignarle un atributo lanza
    # "TypeError: cannot set 'find_class' attribute of immutable type
    # '_pickle.Unpickler'".
    import joblib.numpy_pickle as _joblib_numpy_pickle
    _orig_joblib_find_class = _joblib_numpy_pickle.NumpyUnpickler.find_class
    def _custom_joblib_find_class(self, module, name):
        if name in _PICKLE_CLASS_OVERRIDES:
            return _PICKLE_CLASS_OVERRIDES[name]
        return _orig_joblib_find_class(self, module, name)
    _joblib_numpy_pickle.NumpyUnpickler.find_class = _custom_joblib_find_class

    # Red de seguridad para `pickle.load` puro (implementación pura de
    # Python, `pickle._Unpickler`, que SÍ es mutable).
    _orig_pickle_find_class = pickle._Unpickler.find_class
    def _custom_pickle_find_class(self, module, name):
        if name in _PICKLE_CLASS_OVERRIDES:
            return _PICKLE_CLASS_OVERRIDES[name]
        return _orig_pickle_find_class(self, module, name)
    pickle._Unpickler.find_class = _custom_pickle_find_class
except Exception as _interceptor_error:
    class DemandForecastModel:
        pass

    st.warning(
        "No se pudo precargar la clase de deserialización `DemandForecastModel`: "
        f"{_interceptor_error}. La carga local de `lgbm_eval.joblib` puede fallar."
    )

from src.agent.bot import analyze_market

st.set_page_config(page_title="Coffee Market Intelligence", page_icon="☕", layout="wide")

st.title("☕ Agente de Inteligencia de Mercado de Café")
st.caption("Sistema de análisis prospectivo alimentado por MLOps y LLM")
st.caption(
    "🔵 Entrenamiento (1990-2015) · ⚫ Real de validación (2016-2019) · "
    "🟢 Predicción de evaluación (2016-2019, `lgbm_eval.joblib` / `4_evaluation`) · "
    "🔴 Proyección futura (≥2020, `lgbm_prod.joblib`, servida por la API)."
)


@st.cache_data(ttl=15, show_spinner=False)
def api_disponible() -> bool:
    """Chequeo rápido y cacheado de si la API (`make run-api`) responde.

    Usa el endpoint `/health` que expone `src/api/main.py`. Se cachea por
    15s (no en cada rerun de Streamlit) para no bloquear la UI con un
    timeout de red en cada interacción si la API está caída. La UI ya NO
    intenta cargar `lgbm_prod.joblib` localmente bajo ningún escenario: la
    proyección futura (línea roja) depende exclusivamente de esta API.
    """
    try:
        res = requests.get(f"{API_URL}/health", timeout=2)
        return res.status_code == 200
    except Exception:
        return False


if not api_disponible():
    st.error(
        "⚠️ La API de pronóstico no está disponible en "
        f"`{API_URL}`. Las proyecciones futuras (≥2020, línea roja) no se "
        "mostrarán hasta que la inicies. Abre otra terminal y ejecuta "
        "`make run-api`, luego recarga esta página."
    )


@st.cache_data(ttl=60, show_spinner=False)
def obtener_paises_disponibles() -> list[str]:
    """Lista de países soportados, tal como los espera la API (`GET /countries`).

    Se usa esta fuente -- y NO una lista fija en el código -- para que el
    selector de la UI nunca ofrezca un nombre que la API no reconozca. El
    dataset real usa nombres como "Viet Nam" (no "Vietnam"); una lista
    hardcodeada con el nombre "equivocado" es exactamente lo que producía
    el error "El país 'Vietnam' no existe en el registro histórico." Si la
    API no está disponible, cae de respaldo a los nombres distintos de
    `data_long.parquet` (el mismo archivo que carga `ForecastingService` en
    el backend), así que los nombres siguen siendo consistentes.
    """
    try:
        res = requests.get(f"{API_URL}/countries", timeout=3)
        if res.status_code == 200:
            countries = res.json().get("countries", [])
            if countries:
                return sorted(countries)
    except Exception:
        pass

    parquet_path = ROOT_DIR / "data" / "02_processed" / "data_long.parquet"
    if parquet_path.exists():
        df_countries = pd.read_parquet(parquet_path, columns=["Country"])
        return sorted(df_countries["Country"].dropna().unique().tolist())
    return []


# --- BARRA LATERAL ---
st.sidebar.header("Parámetros del Análisis")
paises_disponibles = obtener_paises_disponibles()
if not paises_disponibles:
    st.sidebar.warning("No se pudo determinar la lista de países disponibles.")
_default_paises = [p for p in ("Colombia", "Brazil") if p in paises_disponibles] or paises_disponibles[:2]
paises = st.sidebar.multiselect(
    "Países a analizar",
    paises_disponibles,
    default=_default_paises,
)
anio = st.sidebar.slider("Año proyectado", min_value=2020, max_value=2030, value=2025)

TRAIN_END_YEAR = 2015
VAL_START_YEAR = 2016
VAL_END_YEAR = 2019

# --- CARGAR RECURSOS CON CACHÉ ---
@st.cache_resource
def obtener_recursos():
    """Carga los recursos para las series 1-3 del gráfico (train / real / eval).

    `lgbm_eval.joblib` (entrenado solo con 1990-2015) se usa únicamente como
    respaldo de la línea verde (ver `obtener_predicciones_evaluacion`) si el
    parquet oficial de `4_evaluation` todavía no existe. La línea roja
    (predicción futura, ≥2020) NO se calcula aquí ni en ningún otro punto de
    este archivo: se obtiene exclusivamente de la API (`/predict`, ver
    `generar_grafico_subplots`); `lgbm_prod.joblib` nunca se carga en la UI.
    """
    eval_model_path = ROOT_DIR / "data" / "04_models" / "lgbm_eval.joblib"
    feat_path = ROOT_DIR / "data" / "03_features" / "data_features.parquet"
    eval_preds_path = ROOT_DIR / "data" / "05_evaluation" / "evaluation_predictions.parquet"

    model_eval = None
    if eval_model_path.exists():
        try:
            model_eval = joblib.load(eval_model_path)
        except Exception as e:
            st.error(f"Error al cargar el modelo joblib ({eval_model_path.name}): {e}")

    if model_eval is None:
        st.warning("No se encontró `lgbm_eval.joblib`. Ejecuta `make train` para regenerarlo.")

    df_feat = pd.read_parquet(feat_path) if feat_path.exists() else None
    df_eval_preds = pd.read_parquet(eval_preds_path) if eval_preds_path.exists() else None

    return model_eval, df_feat, df_eval_preds


# --- GRAFICADO SUBPLOTS ---
def generar_grafico_subplots(paises_seleccionados, anio_limite):
    parquet_path = ROOT_DIR / "data" / "02_processed" / "data_long.parquet"
    if not parquet_path.exists():
        st.error("No se encontró el archivo de datos en `data/02_processed/data_long.parquet`.")
        return None

    df_hist = pd.read_parquet(parquet_path)

    num_cols = df_hist.select_dtypes(include=['number']).columns.tolist()
    year_col = [c for c in num_cols if any(k in c.lower() for k in ['year', 'año', 'an'])][0]
    val_col = [c for c in num_cols if c != year_col][0]

    str_cols = df_hist.select_dtypes(include=['object', 'category']).columns.tolist()
    country_col = [c for c in str_cols if any(k in c.lower() for k in ['country', 'pais', 'país'])][0]

    model_eval, df_feat, df_eval_preds = obtener_recursos()

    # Predicción futura (≥2020, modelo prod) -- EXCLUSIVAMENTE vía API.
    # `api_preds_by_country` mapea país -> lista de puntos futuros ya
    # parseados desde `MultiForecastResponse` (ver src/api/schemas.py):
    # {"results": [{"country": ..., "data": [...], "error": str | None}]}.
    # Un país inválido para la API ya NO hace fallar la petición completa
    # (la API responde 200 con esa entrada marcada en `error`) -- se guarda
    # aparte en `api_errors_by_country` para mostrar un aviso específico en
    # vez del genérico "sin proyección disponible". Si la API no responde,
    # el dict queda vacío y NO se intenta ningún respaldo local con
    # `lgbm_prod.joblib` (requisito explícito: la UI no debe intentar
    # cargar el modelo de producción localmente).
    api_preds_by_country: dict[str, list[tuple[int, float]]] = {}
    api_errors_by_country: dict[str, str] = {}
    if api_disponible():
        try:
            res = requests.post(
                f"{API_URL}/predict",
                json={"countries": paises_seleccionados, "end_year": anio_limite},
                timeout=15,
            )
            if res.status_code == 200:
                payload = res.json()
                for country_result in payload.get("results", []):
                    pais_res = country_result.get("country")
                    if country_result.get("error"):
                        api_errors_by_country[pais_res] = country_result["error"]
                        api_preds_by_country[pais_res] = []
                        continue
                    puntos = [
                        (int(p["year"]), float(p["consumption"]))
                        for p in country_result.get("data", [])
                        if p.get("is_forecast") and int(p["year"]) <= anio_limite
                    ]
                    api_preds_by_country[pais_res] = puntos
            else:
                st.warning(f"La API respondió con un error ({res.status_code}) al pedir la proyección futura.")
        except Exception as e:
            st.warning(f"No se pudo obtener la proyección futura desde la API: {e}")

    n_paises = len(paises_seleccionados)
    fig = make_subplots(
        rows=n_paises, cols=1,
        subplot_titles=[f"Serie Temporal Completa: {p}" for p in paises_seleccionados],
        vertical_spacing=0.12
    )

    # Helper para ejecutar inferencias sobre un DataFrame formateando categóricas.
    # Recibe explícitamente qué modelo usar (eval o prod) -- nunca un default
    # implícito, para no repetir el bug de fuga de datos entre roles.
    def predecir_con_modelo(df_input, modelo):
        if modelo is None or df_input.empty:
            return []
        drop_cols = [c for c in [country_col, year_col, val_col, 'target'] if c in df_input.columns]
        X = df_input.drop(columns=drop_cols, errors='ignore').copy()

        # Garantizar formato categórico para LightGBM
        for col in X.columns:
            if X[col].dtype == 'object' or col == country_col:
                X[col] = X[col].astype('category')

        if hasattr(modelo, 'predict'):
            return modelo.predict(X)
        elif hasattr(modelo, 'model') and hasattr(modelo.model, 'predict'):
            return modelo.model.predict(X)
        return []

    def obtener_predicciones_evaluacion(pais):
        """Predicción del modelo para 2016-2019, calculada por `4_evaluation`.

        Fuente principal: `data/05_evaluation/evaluation_predictions.parquet`
        (generado por `src/pipelines/4_evaluation`), para que el gráfico use
        EXACTAMENTE los mismos números que las métricas oficiales de
        validación, en vez de recalcularlos con una segunda copia del
        modelo. Si ese artefacto todavía no existe (p.ej. no se ha corrido
        `make evaluate`), se recalcula aquí mismo con `lgbm_eval.joblib`
        como respaldo, para no dejar el gráfico incompleto.
        """
        if df_eval_preds is not None and country_col in df_eval_preds.columns:
            df_p_eval = df_eval_preds[df_eval_preds[country_col] == pais].sort_values(by=year_col)
            if not df_p_eval.empty and "predicted_demand" in df_p_eval.columns:
                return list(df_p_eval[year_col]), list(df_p_eval["predicted_demand"])

        # Respaldo: recalcular con el modelo eval si el parquet oficial no está disponible.
        if df_feat is not None:
            df_fp = df_feat[
                (df_feat[country_col] == pais)
                & (df_feat[year_col] >= VAL_START_YEAR)
                & (df_feat[year_col] <= VAL_END_YEAR)
            ].sort_values(by=year_col)
            if not df_fp.empty:
                y_eval_pred = predecir_con_modelo(df_fp, model_eval)
                if len(y_eval_pred) > 0:
                    return list(df_fp[year_col]), list(y_eval_pred)

        return [], []

    def obtener_prediccion_futura(pais, anio_limite):
        """Predicción futura (>=2020, modelo prod) -- EXCLUSIVAMENTE vía API.

        No hay respaldo local: si la API no está corriendo o no devolvió
        datos para este país, se retorna una lista vacía y el subplot
        simplemente no dibuja la línea roja para ese país (con un aviso
        explícito, ver más abajo).
        """
        return api_preds_by_country.get(pais, [])

    for i, pais in enumerate(paises_seleccionados, start=1):
        df_p = df_hist[df_hist[country_col] == pais].sort_values(by=year_col)

        # 1. ENTRENAMIENTO (1990 - 2015) -> AZUL
        df_train = df_p[df_p[year_col] <= TRAIN_END_YEAR]
        fig.add_trace(
            go.Scatter(
                x=df_train[year_col], y=df_train[val_col],
                mode='lines+markers', name=f'1. Entrenamiento ({df_train[year_col].min() if not df_train.empty else 1990}-{TRAIN_END_YEAR})',
                line=dict(color='#1f77b4', width=2.5),
                marker=dict(size=6),
                legendgroup='train', showlegend=(i == 1)
            ), row=i, col=1
        )

        # 2. REAL DE VALIDACIÓN (2016 - 2019) -> NEGRO
        #    Se antepone el último punto de entrenamiento (2015) para que la
        #    línea se vea continua con el segmento azul.
        df_val_real = df_p[(df_p[year_col] >= VAL_START_YEAR) & (df_p[year_col] <= VAL_END_YEAR)]
        last_train_pt = df_p[df_p[year_col] == TRAIN_END_YEAR][val_col].values
        x_val_real = ([TRAIN_END_YEAR] if len(last_train_pt) > 0 else []) + list(df_val_real[year_col])
        y_val_real = (list(last_train_pt) if len(last_train_pt) > 0 else []) + list(df_val_real[val_col])
        fig.add_trace(
            go.Scatter(
                x=x_val_real, y=y_val_real,
                mode='lines+markers', name=f'2. Real de Validación ({VAL_START_YEAR}-{VAL_END_YEAR})',
                line=dict(color='#000000', width=2.5),
                marker=dict(size=6),
                legendgroup='val_real', showlegend=(i == 1)
            ), row=i, col=1
        )

        # 3. PREDICCIÓN DE EVALUACIÓN (2016 - 2019, modelo eval / 4_evaluation) -> VERDE
        x_eval_pred, y_eval_pred = obtener_predicciones_evaluacion(pais)
        if x_eval_pred:
            x_eval_plot = ([TRAIN_END_YEAR] if len(last_train_pt) > 0 else []) + list(x_eval_pred)
            y_eval_plot = (list(last_train_pt) if len(last_train_pt) > 0 else []) + list(y_eval_pred)
            fig.add_trace(
                go.Scatter(
                    x=x_eval_plot, y=y_eval_plot,
                    mode='lines+markers', name=f'3. Predicción de Evaluación ({VAL_START_YEAR}-{VAL_END_YEAR}, modelo eval)',
                    line=dict(color='#2ca02c', width=2.5, dash='dot'),
                    marker=dict(size=8, symbol='x'),
                    legendgroup='eval_pred', showlegend=(i == 1)
                ), row=i, col=1
            )
        else:
            st.caption(f"⚠️ Sin predicción de evaluación disponible para {pais} (corre `make evaluate`).")

        # 4. PREDICCIÓN FUTURA (2020 - Año Límite, modelo prod) -> ROJO
        future_points = obtener_prediccion_futura(pais, anio_limite)

        if future_points:
            last_real_2019 = df_p[df_p[year_col] == VAL_END_YEAR][val_col].values
            x_fut = ([VAL_END_YEAR] if len(last_real_2019) > 0 else []) + [p[0] for p in future_points]
            y_fut = (list(last_real_2019) if len(last_real_2019) > 0 else []) + [p[1] for p in future_points]

            fig.add_trace(
                go.Scatter(
                    x=x_fut, y=y_fut,
                    mode='lines+markers', name='4. Predicción Futura (≥2020, modelo prod)',
                    line=dict(color='#d62728', width=2.5, dash='dash'),
                    marker=dict(size=7, symbol='diamond'),
                    legendgroup='pred_futura', showlegend=(i == 1)
                ), row=i, col=1
            )
        else:
            api_error_msg = api_errors_by_country.get(pais)
            if api_error_msg:
                # País rechazado puntualmente por la API (nombre no encontrado
                # en el dataset) -- se muestra el motivo exacto que devolvió
                # el backend (incluye sugerencias de nombres similares, ver
                # ForecastingService.predict_country) en vez del aviso
                # genérico de "API no disponible".
                st.caption(f"⚠️ {pais}: {api_error_msg}")
            else:
                st.caption(
                    f"⚠️ Sin proyección futura disponible para {pais}. "
                    f"Verifica que la API esté corriendo (`make run-api`, {API_URL})."
                )

        fig.update_xaxes(title_text="Año", row=i, col=1)
        fig.update_yaxes(title_text="Consumo / Demanda", row=i, col=1)

    fig.update_layout(
        height=400 * n_paises,
        title_text="📈 Serie Temporal Completa por País",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

# --- INTERFAZ PRINCIPAL ---
if st.sidebar.button("🚀 Generar Informe y Gráficos", type="primary"):
    if not paises:
        st.warning("Selecciona al menos un país.")
    else:
        st.subheader("📊 Series de Tiempo por País (Subplots)")
        with st.spinner("Calculando inferencia y generando subplots..."):
            fig = generar_grafico_subplots(paises, anio)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("🤖 Informe del Agente de Inteligencia")
        with st.spinner("El agente está redactando el informe..."):
            try:
                respuesta = analyze_market(paises)
                st.markdown(respuesta)
            except Exception as e:
                st.error(f"Error en la consulta del agente: {e}")
                st.info("Verifica que la API esté corriendo en otra terminal (`make run-api`).")
else:
    st.info("Haz clic en **Generar Informe y Gráficos** en la barra lateral.")
