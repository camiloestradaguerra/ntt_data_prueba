"""Utilidades compartidas para lanzar un proceso local (Streamlit, Uvicorn, ...)
junto con un túnel ngrok hacia su puerto, con arranque y cierre ordenados.

Usado por `run_ui_tunnel.py` (Streamlit/UI) y `run_api_tunnel.py` (Uvicorn/API)
-- ver Makefile, targets `run-ui` y `run-api` -- para no duplicar tres piezas
de lógica que ambos necesitan igual:

1. Resolución de puerto multiplataforma: argumento CLI > variable de entorno >
   valor por defecto. El argumento posicional es la vía a prueba de shell (
   funciona igual en cmd.exe, PowerShell, bash o zsh, sin sintaxis especial de
   variable de entorno); la variable de entorno queda como mecanismo alterno
   para quien invoque el script directamente en vez de vía `make`.
2. Apertura/cierre del túnel ngrok (vía `pyngrok`), degradando con un aviso
   claro -- nunca una excepción que tumbe el proceso -- si `pyngrok` no está
   instalado o no hay red/autenticación de ngrok disponible.
3. Manejo de señales (Ctrl+C / SIGINT, SIGTERM, y CTRL_BREAK_EVENT en Windows
   vía `signal.SIGBREAK`) para que el túnel jamás quede huérfano corriendo sin
   el proceso que exponía, sin importar cómo se detenga (Ctrl+C, `taskkill`,
   cierre de terminal, `make` interrumpido).
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys


def resolver_puerto(env_var: str, default: int) -> int:
    """Puerto: argumento CLI (sys.argv[1]) > variable de entorno `env_var` > `default`.

    `make run-ui` / `make run-api` pasan SIEMPRE el puerto como argumento
    posicional (ver Makefile), sin importar el sistema operativo, así que
    esta vía funciona incluso si la sintaxis de variable de entorno de la
    receta de Make (`set VAR=val &&` en Windows, `VAR=val` en Linux/Mac)
    fallara por alguna razón del shell del usuario.
    """
    if len(sys.argv) > 1 and sys.argv[1].strip():
        try:
            return int(sys.argv[1])
        except ValueError:
            print(
                f"⚠️  Puerto inválido en el argumento ('{sys.argv[1]}'), "
                f"usando {env_var} / valor por defecto."
            )
    return int(os.getenv(env_var, str(default)))


class TunnelRunner:
    """Lanza un comando local y, en paralelo, un túnel ngrok hacia su puerto.

    Uso:
        runner = TunnelRunner(port=8501, label="UI de Streamlit")
        runner.ejecutar(["streamlit", "run", "app.py", "--server.port", "8501"])

    `ejecutar()` bloquea hasta que el proceso hijo termina (o hasta Ctrl+C /
    una señal de terminación), y garantiza que el túnel ngrok se cierre en
    todos los casos.
    """

    def __init__(self, port: int, label: str) -> None:
        self.port = port
        self.label = label
        self.public_url: str | None = None
        self._proc: subprocess.Popen | None = None

    # -- ngrok -----------------------------------------------------------

    def abrir_tunel(self) -> str | None:
        """Abre el túnel ngrok hacia `self.port`. Devuelve la URL pública o None.

        Nunca lanza una excepción: si `pyngrok` no está instalado, o ngrok no
        está autenticado/alcanzable, imprime un aviso claro y el proceso
        local sigue funcionando igual, solo sin URL pública.
        """
        try:
            from pyngrok import conf, ngrok
        except ImportError:
            print(
                "⚠️  `pyngrok` no está instalado -- "
                f"{self.label} se abrirá solo en http://localhost:{self.port}, "
                "sin URL pública. Instala la dependencia con "
                "`pip install -r requirements.txt` para habilitar el túnel automático."
            )
            return None

        authtoken = os.getenv("NGROK_AUTHTOKEN")
        if authtoken:
            conf.get_default().auth_token = authtoken

        print(f"🌐 Abriendo túnel ngrok hacia http://localhost:{self.port} ({self.label}) ...")
        try:
            tunnel = ngrok.connect(self.port, "http")
        except Exception as e:
            print(
                f"⚠️  No se pudo crear el túnel ngrok ({e}). Verifica que ngrok "
                "esté instalado/autenticado (`ngrok config add-authtoken <token>` "
                "o la variable de entorno NGROK_AUTHTOKEN). "
                f"{self.label} seguirá disponible en http://localhost:{self.port}, "
                "solo sin URL pública."
            )
            return None

        self.public_url = tunnel.public_url
        atexit.register(self._cerrar_tunel)
        return self.public_url

    def _cerrar_tunel(self) -> None:
        """Cierra el túnel ngrok. Idempotente -- puede llamarse más de una vez
        sin error (registrado con atexit Y llamado explícitamente en el
        `finally` de `ejecutar()`, doble seguro)."""
        if not self.public_url:
            return
        from pyngrok import ngrok

        try:
            ngrok.disconnect(self.public_url)
        except Exception:
            pass
        try:
            ngrok.kill()
        except Exception:
            pass

    # -- proceso local -----------------------------------------------------

    def _terminar_proceso(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    def _manejar_senal(self, signum, frame) -> None:  # noqa: ANN001
        """Convierte SIGTERM (Windows: taskkill / Ctrl+Break; Unix: kill) en una
        salida ordenada -- sin esto, solo Ctrl+C (SIGINT/KeyboardInterrupt)
        pasaría por el `finally` de `ejecutar()`, y el túnel podría quedar
        corriendo si el proceso se detiene de otra forma (p.ej. desde el
        Administrador de tareas o un `kill` en Git Bash)."""
        self._terminar_proceso()
        sys.exit(0)

    def ejecutar(self, cmd: list[str]) -> int:
        """Abre el túnel, lanza `cmd` y bloquea hasta que termine. Devuelve su
        código de salida. Cierra el túnel y el proceso ordenadamente en
        cualquier ruta de salida (fin normal, Ctrl+C, SIGTERM/SIGBREAK)."""
        public_url = self.abrir_tunel()

        if public_url:
            print("=" * 72)
            print(f"✅ {self.label} disponible públicamente en: {public_url}")
            print(f"   (redirige a http://localhost:{self.port})")
            print("=" * 72)
        else:
            print(f"ℹ️  {self.label} disponible en: http://localhost:{self.port}")

        signal.signal(signal.SIGTERM, self._manejar_senal)
        if hasattr(signal, "SIGBREAK"):  # Windows
            signal.signal(signal.SIGBREAK, self._manejar_senal)

        exit_code = 0
        try:
            self._proc = subprocess.Popen(cmd)
            exit_code = self._proc.wait()
        except KeyboardInterrupt:
            pass
        finally:
            # Cierre explícito y ordenado, sin depender solo de atexit:
            # primero el proceso local, luego el túnel -- así el túnel jamás
            # sobrevive a lo que exponía.
            self._terminar_proceso()
            self._cerrar_tunel()
        return exit_code
