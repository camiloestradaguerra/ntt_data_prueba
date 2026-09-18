"""Lanza Streamlit (`app.py`) y, en paralelo, un túnel ngrok hacia ese mismo
puerto -- usado por `make run-ui` (ver Makefile).

Por qué un script en vez de solo comandos en el Makefile: `make` no puede
lanzar dos procesos en paralelo (Streamlit + ngrok) y garantizar que el
túnel se cierre limpio al detener la UI (Ctrl+C) de forma portable entre
shells (cmd/PowerShell/Git Bash en Windows, bash en Linux/Mac). `pyngrok` da
control programático del túnel -- arranque, URL pública, cierre ordenado --
sin depender de parsear la salida de un binario `ngrok` lanzado a mano ni de
abrir una segunda terminal. La lógica de túnel/señales vive en
`_tunnel_common.py`, compartida con `run_api_tunnel.py` (mismo mecanismo para
`make run-api`).

Uso: python scripts/run_ui_tunnel.py [puerto]
    El puerto puede venir como argumento posicional (tiene prioridad --
    funciona igual en cmd.exe, PowerShell, bash o zsh, sin sintaxis
    especial de variable de entorno) o por la variable STREAMLIT_PORT
    (usada por `make run-ui` en Linux/Mac, y disponible para invocar el
    script directamente). Por defecto: 8501.

Variables de entorno opcionales:
    STREAMLIT_PORT   Puerto local de Streamlit (por defecto 8501; ignorado
                      si se pasa el argumento posicional).
    NGROK_AUTHTOKEN  Token de ngrok (`ngrok config add-authtoken <token>`
                      también funciona si ya está configurado globalmente).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tunnel_common import TunnelRunner, resolver_puerto  # noqa: E402


def main() -> None:
    port = resolver_puerto("STREAMLIT_PORT", 8501)
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(port),
    ]
    runner = TunnelRunner(port=port, label="UI de Streamlit")
    sys.exit(runner.ejecutar(cmd))


if __name__ == "__main__":
    main()
