import streamlit as st
import sys
from pathlib import Path

# Garantiza la importación de módulos internos sin errores de ruta
sys.path.append(str(Path(__file__).parent))

from src.agent.bot import analyze_market

st.set_page_config(page_title="Coffee Market Intelligence", page_icon="☕", layout="wide")

st.title("☕ Agente de Inteligencia de Mercado de Café")
st.caption("Sistema de análisis prospectivo alimentado por MLOps y LLM")

st.sidebar.header("Parámetros del Análisis")
paises = st.sidebar.multiselect("Países a analizar", ["Colombia", "Brazil", "Vietnam", "Ethiopia"], default=["Colombia", "Brazil"])
anio = st.sidebar.slider("Año proyectado", min_value=2020, max_value=2030, value=2025)

if st.sidebar.button("🚀 Generar Informe", type="primary"):
    if not paises:
        st.warning("Selecciona al menos un país.")
    else:
        paises_str = ", ".join(paises)
        prompt = f"Analiza la demanda de café para {paises_str} hasta {anio} e incluye recomendaciones de precios e innovación."
        
        with st.spinner("El agente está consultando FastAPI y generando el informe..."):
            try:
                respuesta = analyze_market(prompt)
                st.markdown("### 📊 Informe Generado")
                st.markdown(respuesta)
            except Exception as e:
                st.error(f"Error en la consulta: {e}")
                st.info("Verifica que la API esté corriendo en otra terminal (`make run-api`).")
else:
    st.info("Haz clic en **Generar Informe** en la barra lateral para iniciar el análisis del agente.")