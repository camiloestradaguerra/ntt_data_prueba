"""Lanza la API FastAPI (`uvicorn src.api.main:app`) y, en paralelo, un túnel
ngrok hacia ese mismo puerto -- usado por `make run-api` (ver Makefile).

Mismo mecanismo que `run_ui_tunnel.py` (ver `_tunnel_common.py`): túnel ngrok
gestionado con `pyngrok`, cierre ordenado del túnel y del proceso local ante
Ctrl+C, SIGTERM o CTRL_BREAK_EVENT (Windows), y degradación silenciosa (solo
un aviso, nunca un error fatal) si `pyngrok` no está instalado o ngrok no
está disponible -- la API siempre queda accesible al menos en localhost.

Nota sobre `--reload`: a diferencia de `uvicorn --reload` lanzado
directamente, aquí Uvicorn corre como el único proceso hijo de este wrapper
(sin el proceso "reloader" adicional que crea `--reload`), para que
`terminate()`/señales lo cierren de forma predecible. Si necesitas
autorecarga en desarrollo activo, usa `make run-api-dev` (uvicorn directo,
sin túnel -- ver Makefile) en vez de este script.

Uso: python scripts/run_api_tunnel.py [puerto]
    El puerto puede venir como argumento posicional (tiene prioridad) o por
    la variable API_PORT. Por defecto: 8000.

Variables de entorno opcionales:
    API_PORT         Puerto local de la API (por defecto 8000; ignorado si
                      se pasa el argumento posicional).
    NGROK_AUTHTOKEN  Token de ngrok (`ngrok config add-authtoken <token>`
                      también funciona si ya está configurado globalmente).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tunnel_common import TunnelRunner, resolver_puerto  # noqa: E402


def main() -> None:
    port = resolver_puerto("API_PORT", 8000)
    cmd = [
        sys.executable, "-m", "uvicorn", "src.api.main:app",
        "--host", "0.0.0.0", "--port", str(port),
    ]
    runner = TunnelRunner(port=port, label="API de Pronóstico de Café")
    sys.exit(runner.ejecutar(cmd))


if __name__ == "__main__":
    main()
