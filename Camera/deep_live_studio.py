"""Deep Live Studio — one control centre for the local camera and voice apps.

The camera and RVC voice engine deliberately stay in independent processes.
That isolates GPU/audio failures: stopping one module can never terminate the
other, while the user still has one compact desktop control panel.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

# Studio is a lightweight control panel. Its widgets do not need the GPU that
# the camera's TensorRT/CUDA work requires, so do not compete for it.
os.environ.setdefault("QT_OPENGL", "software")

import psutil
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parent
VOICE_ROOT = ROOT.parent / "Voice"
SCENES_PATH = ROOT / "studio_scenes.json"
CAMERA_STATE_PATH = ROOT / "switch_states.json"


QSS = """
QMainWindow, QWidget { background: #0b1020; color: #e7ebff; font-family: Segoe UI; font-size: 13px; }
QTabWidget::pane { border: 1px solid #293553; border-radius: 10px; top: -1px; }
QTabBar::tab { background: #121a2e; color: #aeb9dc; padding: 10px 22px; margin-right: 3px; border: 1px solid #293553; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; }
QTabBar::tab:selected { background: #7047ef; color: white; }
QFrame#card { background: #121a2e; border: 1px solid #293553; border-radius: 12px; }
QLabel#title { font-size: 25px; font-weight: 700; color: #f1eeff; }
QLabel#subtitle { color: #98a6cf; }
QLabel#cardTitle { font-size: 17px; font-weight: 700; }
QLabel#statusReady { color: #63e6a5; font-weight: 600; }
QLabel#statusStopped { color: #aeb9dc; font-weight: 600; }
QLabel#metric { background: #0d1426; border-radius: 8px; padding: 9px 13px; color: #bac6eb; }
QPushButton { background: #273552; border: 1px solid #3a4b73; border-radius: 8px; padding: 9px 14px; font-weight: 600; color: #eaf0ff; }
QPushButton:hover { background: #34466d; }
QPushButton#primary { background: #7047ef; border-color: #8d70ff; color: white; }
QPushButton#primary:hover { background: #815bfa; }
QPushButton#danger { background: #52293c; border-color: #84445f; }
QPushButton#scene { text-align: left; min-height: 64px; }
"""


DEFAULT_SCENES = {
    "Стрим": {
        "description": "Камера в Unity Video Capture, качество 720p, умный FPS включён.",
        "camera": {"virtual_camera": True, "virtual_background": True, "smart_fps": True,
                   "performance_mode": "Balanced", "preview_quality": "720p", "target_fps": 25},
    },
    "Discord": {
        "description": "Умеренная нагрузка для звонка: без виртуального фона.",
        "camera": {"virtual_camera": True, "virtual_background": False, "smart_fps": True,
                   "performance_mode": "Balanced", "preview_quality": "720p", "target_fps": 20},
    },
    "Запись": {
        "description": "Максимум качества, без автоматического снижения частоты фона.",
        "camera": {"virtual_camera": True, "virtual_background": True, "smart_fps": False,
                   "performance_mode": "Quality", "preview_quality": "1080p", "target_fps": 25},
    },
}


def _matching_processes(kind: str) -> Iterable[psutil.Process]:
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or ()).lower()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if kind == "camera" and str(ROOT).lower() in command and "run.py" in command:
            yield process
        if kind == "voice" and str(VOICE_ROOT).lower() in command and "realtime_gui.py" in command:
            yield process


def _running(kind: str) -> bool:
    return any(True for _ in _matching_processes(kind))


def _stop(kind: str) -> None:
    processes = list(_matching_processes(kind))
    for process in processes:
        try:
            process.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    _, alive = psutil.wait_procs(processes, timeout=3)
    for process in alive:
        try:
            process.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass


def _start_camera() -> Optional[str]:
    if _running("camera"):
        return "Камера уже запущена."
    executable = ROOT / "venv" / "Scripts" / "pythonw.exe"
    if not executable.is_file():
        return "Не найдено окружение Deep Live Camera."
    try:
        subprocess.Popen(
            [str(executable), "run.py", "--execution-provider", "cuda"],
            cwd=str(ROOT), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return None
    except OSError as exc:
        return f"Не удалось запустить камеру: {exc}"


def _start_voice() -> Optional[str]:
    if _running("voice"):
        return "Голосовой модуль уже запущен."
    executable = VOICE_ROOT / ".venv" / "Scripts" / "pythonw.exe"
    script = VOICE_ROOT / "realtime_gui.py"
    if not executable.is_file() or not script.is_file():
        return f"Не найден Deep Live Voice в {VOICE_ROOT}."
    try:
        child_environment = os.environ.copy()
        # RVC lives in its own virtual environment, so it remains an isolated
        # audio engine.  This flag makes its Qt window start hidden; Studio
        # then hosts that real window in the Voice page.
        child_environment["DEEP_LIVE_STUDIO_EMBED"] = "1"
        subprocess.Popen(
            [str(executable), str(script)], cwd=str(VOICE_ROOT),
            env=child_environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return None
    except OSError as exc:
        return f"Не удалось запустить голос: {exc}"


def _read_json(path: Path, fallback: dict) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else fallback
    except (OSError, ValueError):
        return fallback


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


class StudioWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Deep Live Studio")
        self.setMinimumSize(900, 600)
        self.resize(1040, 680)
        self.setStyleSheet(QSS)
        # nvidia-smi may briefly block while TensorRT is busy.  It must never
        # run in Qt's GUI thread, otherwise the Studio window visibly freezes
        # whenever the live camera consumes the GPU.
        self._gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="studio-gpu")
        self._gpu_future: Optional[Future[str]] = None
        self._last_gpu_text = "GPU: проверка…"
        self._last_gpu_request = 0.0
        self._moving_until = 0.0

        tabs = QTabWidget()
        tabs.addTab(self._build_studio_tab(), "Студия")
        tabs.addTab(self._build_scenes_tab(), "Сцены")
        tabs.addTab(self._build_help_tab(), "Справка")
        self.setCentralWidget(tabs)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(2000)
        self._refresh()

    def _build_studio_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)

        title = QLabel("Deep Live Studio")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel("Единый центр управления камерой и голосом. Модули запускаются независимо и используют локальные модели.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        actions = QHBoxLayout()
        self.start_all = QPushButton("▶ Запустить всё")
        self.start_all.setObjectName("primary")
        self.start_all.clicked.connect(self._start_all)
        self.stop_all = QPushButton("■ Остановить всё")
        self.stop_all.setObjectName("danger")
        self.stop_all.clicked.connect(self._stop_all)
        actions.addWidget(self.start_all)
        actions.addWidget(self.stop_all)
        actions.addStretch(1)
        layout.addLayout(actions)

        cards = QGridLayout()
        cards.setSpacing(14)
        self.camera_card, self.camera_status = self._module_card(
            "Камера", "Замена лица, фон и Unity Video Capture", self._start_camera, self._stop_camera
        )
        self.voice_card, self.voice_status = self._module_card(
            "Голос", "RVC, микрофон и преобразованный выход", self._start_voice, self._stop_voice
        )
        cards.addWidget(self.camera_card, 0, 0)
        cards.addWidget(self.voice_card, 0, 1)
        layout.addLayout(cards)

        metrics = QHBoxLayout()
        self.cpu_metric = QLabel("CPU: —")
        self.ram_metric = QLabel("RAM: —")
        self.gpu_metric = QLabel("GPU: —")
        for label in (self.cpu_metric, self.ram_metric, self.gpu_metric):
            label.setObjectName("metric")
            metrics.addWidget(label)
        metrics.addStretch(1)
        layout.addLayout(metrics)

        self.notice = QLabel("Система готова.")
        self.notice.setObjectName("subtitle")
        layout.addWidget(self.notice)
        layout.addStretch(1)
        return page

    def _module_card(self, title_text: str, description: str, start_cb, stop_cb):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        title = QLabel(title_text)
        title.setObjectName("cardTitle")
        description_label = QLabel(description)
        description_label.setObjectName("subtitle")
        description_label.setWordWrap(True)
        status = QLabel("Проверка…")
        buttons = QHBoxLayout()
        start = QPushButton("Запустить")
        start.setObjectName("primary")
        start.clicked.connect(start_cb)
        stop = QPushButton("Остановить")
        stop.setObjectName("danger")
        stop.clicked.connect(stop_cb)
        buttons.addWidget(start)
        buttons.addWidget(stop)
        layout.addWidget(title)
        layout.addWidget(description_label)
        layout.addWidget(status)
        layout.addLayout(buttons)
        return card, status

    def _build_scenes_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(22, 22, 22, 22)
        title = QLabel("Сцены")
        title.setObjectName("title")
        layout.addWidget(title)
        info = QLabel("Сцена меняет реальные настройки камеры. Если камера уже запущена, перезапусти только её, чтобы применить сцену.")
        info.setObjectName("subtitle")
        info.setWordWrap(True)
        layout.addWidget(info)
        for scene_name, scene in DEFAULT_SCENES.items():
            button = QPushButton(f"{scene_name}\n{scene['description']}")
            button.setObjectName("scene")
            button.clicked.connect(lambda _checked=False, name=scene_name: self._apply_scene(name))
            layout.addWidget(button)
        layout.addStretch(1)
        return page

    def _build_help_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(22, 22, 22, 22)
        title = QLabel("Как пользоваться")
        title.setObjectName("title")
        text = QLabel(
            "1. Выбери сцену, если нужна.\n"
            "2. Запусти «Камера» и выбери Unity Video Capture в OBS/TikTok.\n"
            "3. Запусти «Голос» и выбери устройства в его окне.\n"
            "4. Статусы и системная загрузка обновляются каждую секунду.\n\n"
            "Используй лицо и голос только с согласия человека, которого они затрагивают."
        )
        text.setWordWrap(True)
        text.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(text)
        layout.addStretch(1)
        return page

    def _start_camera(self) -> None:
        error = _start_camera()
        self.notice.setText(error or "Камера запускается…")
        self._refresh()

    def _stop_camera(self) -> None:
        _stop("camera")
        self.notice.setText("Камера остановлена.")
        self._refresh()

    def _start_voice(self) -> None:
        error = _start_voice()
        self.notice.setText(error or "Голосовой модуль запускается…")
        self._refresh()

    def _stop_voice(self) -> None:
        _stop("voice")
        self.notice.setText("Голосовой модуль остановлен.")
        self._refresh()

    def _start_all(self) -> None:
        errors = [error for error in (_start_camera(), _start_voice()) if error and "уже запущен" not in error]
        self.notice.setText("; ".join(errors) if errors else "Камера и голос запускаются…")
        self._refresh()

    def _stop_all(self) -> None:
        _stop("camera")
        _stop("voice")
        self.notice.setText("Камера и голос остановлены.")
        self._refresh()

    def _apply_scene(self, name: str) -> None:
        scene = DEFAULT_SCENES[name]
        state = _read_json(CAMERA_STATE_PATH, {})
        state.update(scene["camera"])
        _write_json(CAMERA_STATE_PATH, state)
        self.notice.setText(f"Сцена «{name}» сохранена. Перезапусти камеру, если она уже работает.")

    def _refresh(self) -> None:
        # Defer non-essential telemetry during a window drag. Windows sends
        # many move events here and re-styling while TensorRT is busy makes
        # the title bar feel sticky.
        if time.monotonic() < self._moving_until:
            return
        camera_running = _running("camera")
        voice_running = _running("voice")
        self._set_module_status(self.camera_status, camera_running, "Камера запущена", "Камера остановлена")
        self._set_module_status(self.voice_status, voice_running, "Голосовой модуль запущен", "Голосовой модуль остановлен")
        self.cpu_metric.setText(f"CPU: {psutil.cpu_percent():.0f}%")
        memory = psutil.virtual_memory()
        self.ram_metric.setText(f"RAM: {memory.percent:.0f}%  •  {memory.used // 1024**2} MB")
        self.gpu_metric.setText(self._gpu_text())

    def moveEvent(self, event) -> None:
        self._moving_until = time.monotonic() + 0.5
        super().moveEvent(event)

    def closeEvent(self, event) -> None:
        self._gpu_executor.shutdown(wait=False, cancel_futures=True)
        event.accept()

    @staticmethod
    def _set_module_status(label: QLabel, running: bool, ready: str, stopped: str) -> None:
        text = "● " + (ready if running else stopped)
        object_name = "statusReady" if running else "statusStopped"
        if label.text() == text and label.objectName() == object_name:
            return
        label.setText(text)
        if label.objectName() != object_name:
            label.setObjectName(object_name)
            label.style().unpolish(label)
            label.style().polish(label)

    def _gpu_text(self) -> str:
        """Return the last GPU reading and schedule the next one off-thread."""
        if self._gpu_future is not None and self._gpu_future.done():
            try:
                self._last_gpu_text = self._gpu_future.result()
            except Exception:
                self._last_gpu_text = "GPU: недоступно"
            self._gpu_future = None
        if self._gpu_future is None and time.monotonic() - self._last_gpu_request >= 4.0:
            self._gpu_future = self._gpu_executor.submit(self._read_gpu_text)
            self._last_gpu_request = time.monotonic()
        return self._last_gpu_text

    @staticmethod
    def _read_gpu_text() -> str:
        try:
            response = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                text=True, capture_output=True, timeout=2, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if response.returncode == 0 and response.stdout.strip():
                used = response.stdout.strip().split(",")
                return f"GPU: {used[0].strip()}%  •  VRAM {used[1].strip()}/{used[2].strip()} MB"
        except (OSError, subprocess.TimeoutExpired):
            pass
        return "GPU: недоступно"


def main() -> int:
    from studio_shell import run_studio
    return run_studio()


if __name__ == "__main__":
    raise SystemExit(main())
