"""Small Windows launcher for Deep-Live-Cam.

It deliberately does not embed the application itself.  The face-processing
stack uses Qt, CUDA and large ONNX models, and launching the working project
environment is both more reliable and lets the desktop EXE always use the
latest local fixes.
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent


def show_error(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(None, message, "Deep-Live-Cam", 0x10)


def main() -> None:
    pythonw = APP_ROOT / "venv" / "Scripts" / "pythonw.exe"
    entrypoint = APP_ROOT / "run.py"
    if not pythonw.is_file() or not entrypoint.is_file():
        show_error(
            "Файлы Deep-Live-Cam не найдены. Ожидаемая папка:\n"
            f"{APP_ROOT}"
        )
        return
    try:
        subprocess.Popen(
            [str(pythonw), str(entrypoint), "--execution-provider", "cuda"],
            cwd=str(APP_ROOT),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except OSError as exc:
        show_error(f"Не удалось запустить Deep-Live-Cam:\n{exc}")


if __name__ == "__main__":
    main()
