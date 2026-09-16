"""PySide6 UI for Deep-Live-Cam.

Public API kept stable for the rest of the codebase:
    init(start, destroy, lang) -> _Window
        Returned object has .mainloop() that core.py calls.
    update_status(text)
        Thread-safe; routed through Qt signal when called off-UI.
    check_and_ignore_nsfw(target, destroy=None) -> bool
"""

from __future__ import annotations

import os
import platform
import queue
import sys
import tempfile
import threading
import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QObject,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import modules.globals

# Lazy wrapper: importing face_swapper at module import time creates
# core -> ui -> face_swapper -> core circular-imports. The worker calls
# this only after the application has finished importing.
def prepare_target_face(face, frame_shape=None, track_key=0):
    # Lazy import avoids core -> ui -> face_swapper -> core circular imports.
    from modules.processors.frame.face_swapper import prepare_target_face as _prepare_target_face
    return _prepare_target_face(face, frame_shape, track_key=track_key)


def restore_target_mouth(frame, target_frame, face, track_key=0):
    # Lazy import for the same reason as prepare_target_face().
    from modules.processors.frame.face_swapper import restore_target_mouth as _restore_target_mouth
    return _restore_target_mouth(frame, target_frame, face, track_key=track_key)
import modules.metadata
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    add_blank_map,
    detect_many_faces_fast,
    detect_one_face_fast,
    ensure_landmarks,
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    has_valid_map,
    simplify_maps,
)
from modules.gettext import LanguageManager
from modules.gpu_processing import gpu_cvt_color, gpu_flip, gpu_resize
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
)
from modules import imread_unicode
from modules.video_capture import VideoCapturer
from modules.virtual_camera import VirtualCameraPublisher
from modules.virtual_background import VirtualBackground

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

import json


# ─── constants ────────────────────────────────────────────────────────────

ROOT_HEIGHT = 820
ROOT_WIDTH = 640

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 640
PREVIEW_DEFAULT_HEIGHT = 360

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = 720
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = 870
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_SIZE = 100
SOURCE_TARGET_PREVIEW_SIZE = 200


# Cached 8-bit lookup tables make brightness/contrast/gamma effectively a
# single OpenCV pass.  Saturation is only converted to HSV if it is changed.
_COLOR_LUT_KEY = None
_COLOR_LUT = None
_SATURATION_LUT_KEY = None
_SATURATION_LUT = None
_VIBRANCE_LUT_KEY = None
_VIBRANCE_LUT = None


def _apply_output_color_adjustments(frame: np.ndarray) -> np.ndarray:
    """Apply user colour controls without doing work at neutral settings."""
    global _COLOR_LUT_KEY, _COLOR_LUT, _SATURATION_LUT_KEY, _SATURATION_LUT
    global _VIBRANCE_LUT_KEY, _VIBRANCE_LUT
    brightness = float(getattr(modules.globals, "brightness", 0.0))
    contrast = float(getattr(modules.globals, "contrast", 1.0))
    saturation = float(getattr(modules.globals, "saturation", 1.0))
    gamma = float(getattr(modules.globals, "gamma", 1.0))
    vibrance = float(getattr(modules.globals, "digital_vibrance", 0.0))
    if (
        abs(brightness) < 0.01
        and abs(contrast - 1.0) < 0.001
        and abs(saturation - 1.0) < 0.001
        and abs(gamma - 1.0) < 0.001
        and abs(vibrance) < 0.01
    ):
        return frame

    lut_key = (round(brightness, 2), round(contrast, 3), round(gamma, 3))
    if _COLOR_LUT_KEY != lut_key:
        x = np.arange(256, dtype=np.float32) / 255.0
        corrected = (np.power(x, max(0.10, gamma)) * 255.0 * contrast) + brightness
        _COLOR_LUT = np.clip(corrected, 0, 255).astype(np.uint8)
        _COLOR_LUT_KEY = lut_key
    if _COLOR_LUT is not None and lut_key != (0.0, 1.0, 1.0):
        frame = cv2.LUT(frame, _COLOR_LUT)

    if abs(saturation - 1.0) >= 0.001 or abs(vibrance) >= 0.01:
        # One HSV conversion serves both colour controls.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        channel = hsv[:, :, 1]
        if abs(saturation - 1.0) >= 0.001:
            sat_key = round(saturation, 3)
            if _SATURATION_LUT_KEY != sat_key:
                _SATURATION_LUT = np.clip(
                    np.arange(256, dtype=np.float32) * saturation, 0, 255
                ).astype(np.uint8)
                _SATURATION_LUT_KEY = sat_key
            channel = cv2.LUT(channel, _SATURATION_LUT)
        if abs(vibrance) >= 0.01:
            vib_key = round(vibrance, 2)
            if _VIBRANCE_LUT_KEY != vib_key:
                values = np.arange(256, dtype=np.float32)
                # Muted colours are affected most; fully saturated colours
                # remain almost unchanged. This mirrors Digital Vibrance more
                # closely than a second global saturation multiplier.
                factor = 1.0 + (vibrance / 100.0) * (1.0 - values / 255.0)
                _VIBRANCE_LUT = np.clip(values * factor, 0, 255).astype(np.uint8)
                _VIBRANCE_LUT_KEY = vib_key
            channel = cv2.LUT(channel, _VIBRANCE_LUT)
        hsv[:, :, 1] = channel
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame


# ─── modern dark stylesheet ───────────────────────────────────────────────

QSS = """
/* Camera may be embedded in Deep Live Studio.  Use the Studio navy surfaces
   here too, so the real Camera controls do not turn into grey slabs inside
   the unified launcher. */
QMainWindow, QDialog { background-color: #070b14; color: #edf2ff; }
QWidget { color: #e6e6e6; font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif; font-size: 10pt; }

QGroupBox {
    background-color: #0e1727;
    border: 1px solid #263752;
    border-radius: 10px;
    margin-top: 0;
    padding-top: 26px;
    font-weight: 600;
}
QGroupBox::title {
    /* Keep every section name inside its card: never cut the top border. */
    subcontrol-origin: padding;
    subcontrol-position: top left;
    top: 5px;
    left: 8px;
    padding: 0;
    color: #9ec5ff;
}

QPushButton {
    background-color: #2d6cdf;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 6px 12px;
    font-weight: 600;
}
QPushButton:hover  { background-color: #3a7af0; }
QPushButton:pressed{ background-color: #1d57c2; }
QPushButton:disabled { background-color: #444; color: #888; }
QPushButton#secondary {
    background-color: #17243a;
}
QPushButton#secondary:hover { background-color: #22324e; }
QPushButton#danger { background-color: #c2412d; }
QPushButton#danger:hover  { background-color: #d8523c; }

QComboBox {
    background-color: #0b1423;
    border: 1px solid #30435f;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 20px;
}
QComboBox:hover { border-color: #2d6cdf; }
QComboBox QAbstractItemView {
    background-color: #101b2e;
    selection-background-color: #2d6cdf;
    border: 1px solid #404040;
}

QCheckBox {
    spacing: 6px;
    padding: 2px 0;
}
QCheckBox::indicator {
    width: 32px; height: 16px;
    border-radius: 8px;
    background-color: #34455f;
}
QCheckBox::indicator:checked {
    background-color: #2d6cdf;
}

QSlider::groove:horizontal {
    height: 5px;
    background: #283a58;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    width: 14px; height: 14px;
    margin: -4px 0;
    border-radius: 8px;
    border: 1px solid #cccccc;
}
QSlider::sub-page:horizontal {
    background: #2d6cdf;
    border-radius: 3px;
}

QLabel#imageDrop {
    background-color: #0b1423;
    border: 2px dashed #34496a;
    border-radius: 8px;
}
QLabel#statusLabel {
    color: #b9b9b9;
    font-size: 10pt;
    font-style: italic;
}
QLabel#linkLabel {
    color: #6ea8ff;
    text-decoration: underline;
}

QScrollArea { border: none; background-color: #070b14; }
QScrollArea#mainScroll { background-color: #070b14; }
QWidget#mainContent { background-color: #070b14; }

QFrame#card {
    background-color: #0e1727;
    border-radius: 10px;
}

QWidget#controlContent, QScrollArea#controlScroll, QScrollArea#controlScroll > QWidget > QWidget {
    background-color: #070b14;
}
QLabel#dashboardTitle {
    color: #e6e6e6;
    font-size: 16pt;
    font-weight: 700;
    padding: 2px 4px;
}
QLabel#dashboardPreview {
    background-color: #111318;
    border: 1px solid #353b4a;
    border-radius: 12px;
    color: #80879a;
    font-size: 13pt;
}
QLabel#metricLabel {
    color: #9ec5ff;
    font-weight: 600;
    padding: 4px 8px;
}
QPushButton#liveButton {
    background-color: #2f9e55;
    min-width: 110px;
    padding: 10px 18px;
}
QPushButton#liveButton:hover { background-color: #38b866; }
QSplitter::handle { background-color: #18263d; }
QSplitter::handle:horizontal { background-color: #18263d; }
QSplitter::handle:horizontal:hover { background-color: #2d6cdf; }
QScrollBar:vertical {
    background: #070b14;
    width: 12px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #34435c;
    min-height: 35px;
    border-radius: 6px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }

/* Camera / Face Swap — premium Studio workspace layer.  Object-name scoped
   rules keep the established widgets and signals intact while giving the
   embedded page the same visual language as the Studio shell. */
QFrame#cameraHero {
    background: transparent;
    border: none;
}
QLabel#cameraHeroIcon {
    background: transparent;
    border: none;
}
QLabel#cameraHeroTitle { color: #f4f6ff; font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 21px; font-weight: 600; }
QLabel#cameraHeroSubtitle { color: #c0c9dd; font-family: "Segoe UI Variable Text", "Segoe UI"; font-size: 11px; font-weight: 400; }
QLabel#cameraBreadcrumb { color: #bdc7df; font-family: "Segoe UI Variable Text", "Segoe UI"; font-size: 11px; padding: 4px 4px; }

QWidget#controlContent, QScrollArea#controlScroll, QScrollArea#controlScroll > QWidget > QWidget {
    background: #070c18;
}
QFrame#controlsFooter {
    background: #091326;
    border: 1px solid #294a7d;
    border-radius: 10px;
}
QGroupBox#controlCard, QGroupBox#cameraControlCard {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #101d35, stop:1 #0b1529);
    border: 1px solid #28518b;
    border-radius: 11px;
    margin-top: 0;
    padding: 27px 11px 10px 11px;
}
QGroupBox#controlCard::title, QGroupBox#cameraControlCard::title {
    color: #d9e6ff;
    font-size: 12px;
    font-weight: 750;
    subcontrol-origin: padding;
    subcontrol-position: top left;
    top: 5px;
    left: 8px;
    padding: 0;
}
QGroupBox#liveSwapCard, QGroupBox#previewCard {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0d1b35, stop:.6 #0a142a, stop:1 #101a3b);
    border: 1px solid #3a66b3;
    border-radius: 12px;
    margin-top: 0;
    padding-top: 26px;
}
/* Live Face Swap has its own title label, so it needs no empty title band. */
QGroupBox#liveSwapCard { border-color: #5756d6; padding-top: 0; }
QGroupBox#previewCard { border-color: #6746df; }
QGroupBox#mediaSourceCard, QGroupBox#mediaTargetCard {
    background: #0a162c;
    border: 1px solid #315a9c;
    border-radius: 9px;
    margin-top: 0;
    padding-top: 25px;
}
QGroupBox#mediaSourceCard::title, QGroupBox#mediaTargetCard::title {
    color: #a8c7ff;
    font-size: 11px;
    font-weight: 700;
    subcontrol-origin: padding;
    subcontrol-position: top left;
    top: 5px;
    left: 8px;
    padding: 0;
}
QGroupBox#previewCard::title {
    subcontrol-origin: padding;
    subcontrol-position: top left;
    top: 5px;
    left: 8px;
    padding: 0;
}
QLabel#dashboardTitle {
    color: #f2f5ff;
    font-size: 15px;
    font-weight: 800;
    padding: 0 1px;
}
QLabel#dashboardPreview {
    background: qradialgradient(cx:.5,cy:.5,r:1,fx:.5,fy:.5, stop:0 #142a58, stop:.55 #0a162c, stop:1 #070d1b);
    border: 1px solid #405fa9;
    border-radius: 10px;
    color: #b3c3e8;
    font-size: 12px;
    font-weight: 600;
}
QLabel#imageDrop {
    background: #091427;
    border: 1px dashed #5875b7;
    border-radius: 8px;
    color: #b8c9ec;
}
QLabel#metricLabel {
    color: #bbccff;
    background: #0b1730;
    border: 1px solid #31528a;
    border-radius: 7px;
    font-weight: 700;
    padding: 5px 9px;
}

QPushButton {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2b71e7, stop:1 #1760d8);
    border: 1px solid #5194ff;
    border-radius: 7px;
    color: #f7f9ff;
    font-weight: 700;
    padding: 6px 12px;
}
QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7842f3, stop:1 #2a7cf1); border-color: #b096ff; }
QPushButton:pressed { background: #224ea9; }
QPushButton#secondary { background: #142440; border: 1px solid #2d4b7b; }
QPushButton#secondary:hover { background: #1b3359; border-color: #7968e9; }
QPushButton#danger { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #8d263d, stop:1 #bf3f3a); border-color: #ec6463; }
QPushButton#liveButton {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0b9d60, stop:1 #18c877);
    border: 1px solid #55f0a2;
    min-width: 118px;
    border-radius: 8px;
    padding: 9px 19px;
}
QPushButton#liveButton:hover { background: #15b971; }

QComboBox {
    background: #0b1830;
    border: 1px solid #315384;
    border-radius: 7px;
    padding: 5px 9px;
    min-height: 21px;
    color: #e8eeff;
}
QComboBox:hover, QComboBox:focus { border-color: #8167fa; }
QComboBox QAbstractItemView { background: #0d1a33; border: 1px solid #436397; selection-background-color: #5932d5; }
QCheckBox { spacing: 7px; padding: 2px 0; color: #d8e2fa; }
QCheckBox::indicator { width: 29px; height: 16px; border-radius: 8px; background: #334764; border: 1px solid #526987; }
QCheckBox::indicator:checked { background: #7a45f2; border-color: #b394ff; }
QSlider::groove:horizontal { height: 5px; background: #263953; border-radius: 3px; }
QSlider::sub-page:horizontal { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #446dff, stop:1 #8b4dff); border-radius: 3px; }
QSlider::handle:horizontal { background: #f2f5ff; width: 12px; height: 12px; margin: -5px 0; border-radius: 7px; border: 2px solid #9b7bff; }
QSplitter::handle:horizontal { background: #112039; width: 5px; }
QSplitter::handle:horizontal:hover { background: #6748d8; }
"""


# ─── module-level state ───────────────────────────────────────────────────

_APP: Optional[QApplication] = None
_MAIN: Optional["MainWindow"] = None
_PREVIEW: Optional["PreviewWindow"] = None
_WEBCAM_PREVIEW: Optional["WebcamPreviewWindow"] = None
_OUTPUT_WINDOW: Optional["OutputWindow"] = None
_OUTPUT_WINDOW_ACTIVE = False
_DATASET_RECORDER: Optional["DatasetRecorderDialog"] = None
_MAPPER: Optional["MapperDialog"] = None
_LIVE_MAPPER: Optional["LiveMapperDialog"] = None
_LANG: Optional[LanguageManager] = None
_UI_LANGUAGE = "en"
QUALITY_OPTIONS = {
    "360p": (640, 360),
    "480p": (854, 480),
    "480p (4:3)": (640, 480),
    "520p": (960, 520),
    "720p": (1280, 720),
    "960p (4:3)": (1280, 960),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
}

# Capture/output format and processing detail are intentionally separate.
# The first selector asks the physical/virtual camera for a frame format; the
# second limits the processing frame used by Face Swap for performance.
PROCESSING_QUALITY_OPTIONS = {
    "Высокое · 1080p": "1080p",
    "Сбалансированное · 720p": "720p",
    "Быстрое · 480p": "480p",
}

def _quality_process_scale(frame_w: int, frame_h: int, quality: str) -> float:
    """Return a processing scale that honors the selected quality on fixed-resolution webcams.

    Iriun commonly exposes 1280x960 (4:3). The old quality selector only requested
    a camera mode, which could be silently ignored by the device. Here 360p/520p/720p
    also control the actual live processing resolution, while 1080p/1440p keep the
    camera's native detail rather than inventing detail by upscaling before processing.
    """
    target_h = QUALITY_OPTIONS.get(str(quality), QUALITY_OPTIONS["720p"])[1]
    if frame_h <= 0 or frame_w <= 0 or frame_h <= target_h:
        return 1.0
    return max(0.25, min(1.0, target_h / float(frame_h)))

_RU = {
    "Options": "Настройки", "Keep fps": "Сохранять FPS", "Keep audio": "Сохранять аудио",
    "Keep frames": "Сохранять кадры", "Many faces": "Несколько лиц", "Map faces": "Сопоставление лиц",
    "Show FPS": "Показывать FPS", "Poisson Blend": "Смешивание Poisson", "Fix Blueish Cam": "Исправить синеву камеры",
    "Face Enhancer:": "Улучшение лица:", "None": "Нет", "GFPGAN": "GFPGAN", "GPEN-512": "GPEN-512", "GPEN-256": "GPEN-256",
    "Refinement": "Улучшение", "Transparency": "Прозрачность", "Sharpness": "Резкость",
    "Mouth Mask": "Маска рта", "Edge Softness": "Мягкость края", "Advanced": "Дополнительно",
    "Color Match": "Сопоставление цвета", "Performance Mode": "Режим производительности",
    "Quality": "Качество", "Balanced": "Сбалансированный", "Performance": "Производительность",
    "Face Stabilization": "Стабилизация лица", "Eye Protection": "Защита глаз", "Face X": "Лицо X",
    "Face Y": "Лицо Y", "Face Scale": "Масштаб лица", "Face Rotation": "Поворот лица", "Target FPS": "Целевой FPS",
    "Start": "Запуск", "Stop video": "Стоп видео", "Stop live video and release camera": "Остановить видео и освободить камеру", "Preview": "Предпросмотр", "Camera": "Камера",
    "Output": "Окно", "Close": "Закр.",
    "Select Camera:": "Камера:", "Live": "LIVE", "Virtual Camera": "Виртуальная камера", "Virtual Background": "Виртуальный фон", "Smart FPS": "Умный FPS", "Choose background": "Выбрать фон", "Select a face": "Выбрать лицо", "Select a target": "Выбрать цель",
    "Source Face": "Исходное лицо", "Source face": "Исходное лицо", "Target": "Цель", "Live Face Swap": "Live Face Swap",
    "Source x Target Mapper": "Сопоставление источника и цели", "Select source image": "Выбрать исходное изображение",
    "Select target image": "Выбрать целевое изображение", "Submit": "Применить", "Add": "Добавить", "Clear": "Очистить",
    "Quality:": "Качество:", "Camera resolution:": "Разрешение камеры:", "Processing quality:": "Качество обработки:", "Language:": "Язык:", "English": "English", "Russian": "Русский",
    "FPS: —": "FPS: —", "Select a target or press LIVE\nto open the real-time preview": "Выберите цель или нажмите LIVE\nдля запуска предпросмотра в реальном времени",
    }
_BRIDGE: Optional["_UIBridge"] = None


def _(text: str) -> str:
    """Translate UI text. Russian strings are kept local so the UI works without external locale files."""
    if _UI_LANGUAGE == "ru":
        return _RU.get(text, text)
    if _LANG is None:
        return text
    return _LANG._(text)


# Preserve original cwd state for file dialogs.
_RECENT_SOURCE_DIR: Optional[str] = None
_RECENT_TARGET_DIR: Optional[str] = None
_RECENT_OUTPUT_DIR: Optional[str] = None

# QFileDialog filter strings, built from the canonical extension sets in
# globals so every dialog stays in sync (no hand-copied lists to drift).
_IMAGE_FILE_FILTER = "Images (" + " ".join(
    f"*{ext}" for ext in modules.globals.IMAGE_EXTENSIONS
) + ")"
_MEDIA_FILE_FILTER = "Media (" + " ".join(
    f"*{ext}" for ext in (*modules.globals.IMAGE_EXTENSIONS, *modules.globals.VIDEO_EXTENSIONS)
) + ")"
_VIDEO_FILE_FILTER = "Videos (" + " ".join(
    f"*{ext}" for ext in modules.globals.VIDEO_EXTENSIONS
) + ")"


# ─── image utilities ─────────────────────────────────────────────────────


def fit_image_to_size(image, width: int, height: int):
    """BGR ndarray → BGR ndarray scaled to fit within (width, height)."""
    if width is None and height is None or width <= 0 or height <= 0:
        return image
    h, w = image.shape[:2]
    ratio_w = width / w
    ratio_h = height / h
    ratio = min(ratio_w, ratio_h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return gpu_resize(image, dsize=new_size)


def fit_image_to_widget(image, width: int, height: int):
    """Scale only for Qt painting, without using the inference GPU."""
    if (width is None and height is None) or width <= 0 or height <= 0:
        return image
    h, w = image.shape[:2]
    ratio = min(width / w, height / h)
    size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    interpolation = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)


def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Zero-copy BGR ndarray → QPixmap."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """PIL.Image → QPixmap."""
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimg = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def render_image_preview(
    image_path: str, size: Tuple[int, int], crop: bool = True
) -> QPixmap:
    """Render an image preview.

    Source portraits use ``crop=False`` so the user can inspect the entire
    selected photo rather than a centre-cropped strip.
    """
    image = Image.open(image_path).convert("RGB")
    if size and crop:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    elif size:
        contained = ImageOps.contain(image, size, Image.LANCZOS)
        canvas = Image.new("RGB", size, (30, 30, 30))
        x = (size[0] - contained.width) // 2
        y = (size[1] - contained.height) // 2
        canvas.paste(contained, (x, y))
        image = canvas
    return _pil_to_qpixmap(image)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> Optional[QPixmap]:
    capture = cv2.VideoCapture(video_path)
    try:
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if not has_frame:
            return None
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return _pil_to_qpixmap(image)
    finally:
        capture.release()


# ─── persistence ─────────────────────────────────────────────────────────


def save_switch_states():
    state = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "virtual_camera": getattr(modules.globals, "virtual_camera", False),
        "virtual_background": getattr(modules.globals, "virtual_background", False),
        "virtual_background_path": getattr(modules.globals, "virtual_background_path", None),
        "smart_fps": getattr(modules.globals, "smart_fps", True),
        "smart_fps_minimum": getattr(modules.globals, "smart_fps_minimum", 18),
        "virtual_background_interval": getattr(modules.globals, "virtual_background_interval", 2),
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "mouth_mask_size": modules.globals.mouth_mask_size,
        "full_head_coverage": getattr(modules.globals, "full_head_coverage", True),
        "mask_profile": getattr(modules.globals, "mask_profile", "Chin"),
        "trained_identity_mode": getattr(modules.globals, "trained_identity_mode", False),
        "show_diagnostics": getattr(modules.globals, "show_diagnostics", False),
        "sharpness": modules.globals.sharpness,
        "brightness": getattr(modules.globals, "brightness", 0.0),
        "contrast": getattr(modules.globals, "contrast", 1.0),
        "saturation": getattr(modules.globals, "saturation", 1.0),
        "gamma": getattr(modules.globals, "gamma", 1.0),
        "digital_vibrance": getattr(modules.globals, "digital_vibrance", 0.0),
        "texture_preservation": getattr(modules.globals, "texture_preservation", 0.0),
        "opacity": getattr(modules.globals, "opacity", 1.0),
        "mask_feather": getattr(modules.globals, "mask_feather", 50.0),
        "color_match": getattr(modules.globals, "color_match", False),
        "face_stabilization": getattr(modules.globals, "face_stabilization", 0.0),
        "eye_protection": getattr(modules.globals, "eye_protection", 85.0),
        "face_offset_x": getattr(modules.globals, "face_offset_x", 0.0),
        "face_offset_y": getattr(modules.globals, "face_offset_y", 0.0),
        "face_scale": getattr(modules.globals, "face_scale", 1.0),
        "face_rotation": getattr(modules.globals, "face_rotation", 0.0),
        "performance_mode": getattr(modules.globals, "performance_mode", "Balanced"),
        "target_fps": getattr(modules.globals, "target_fps", 20),
        "preview_quality": getattr(modules.globals, "preview_quality", "720p"),
        "camera_resolution": getattr(modules.globals, "camera_resolution", getattr(modules.globals, "preview_quality", "720p")),
        "quality_profile": getattr(modules.globals, "quality_profile", "Quality"),
        "full_head_mode": getattr(modules.globals, "full_head_mode", False),
        "last_source_media_path": getattr(modules.globals, "last_source_media_path", None),
        "last_source_cache_path": getattr(modules.globals, "last_source_cache_path", None),
        "source_profile_from_video": getattr(modules.globals, "source_profile_from_video", False),
        "last_camera_name": getattr(modules.globals, "last_camera_name", None),
        "ui_language": globals().get("_UI_LANGUAGE", "en"),
    }
    try:
        with open("switch_states.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    except OSError:
        pass


def load_switch_states():
    global _UI_LANGUAGE
    # Defaults for newly added refinement controls.
    modules.globals.opacity = getattr(modules.globals, "opacity", 1.0)
    modules.globals.brightness = getattr(modules.globals, "brightness", 0.0)
    modules.globals.contrast = getattr(modules.globals, "contrast", 1.0)
    modules.globals.saturation = getattr(modules.globals, "saturation", 1.0)
    modules.globals.gamma = getattr(modules.globals, "gamma", 1.0)
    modules.globals.digital_vibrance = getattr(modules.globals, "digital_vibrance", 0.0)
    modules.globals.texture_preservation = getattr(modules.globals, "texture_preservation", 0.0)
    modules.globals.mask_feather = getattr(modules.globals, "mask_feather", 50.0)
    modules.globals.color_match = getattr(modules.globals, "color_match", False)
    modules.globals.face_stabilization = getattr(modules.globals, "face_stabilization", 0.0)
    modules.globals.eye_protection = getattr(modules.globals, "eye_protection", 85.0)
    modules.globals.face_offset_x = getattr(modules.globals, "face_offset_x", 0.0)
    modules.globals.face_offset_y = getattr(modules.globals, "face_offset_y", 0.0)
    modules.globals.face_scale = getattr(modules.globals, "face_scale", 1.0)
    modules.globals.face_rotation = getattr(modules.globals, "face_rotation", 0.0)
    modules.globals.performance_mode = getattr(modules.globals, "performance_mode", "Balanced")
    modules.globals.target_fps = getattr(modules.globals, "target_fps", 20)
    modules.globals.preview_quality = getattr(modules.globals, "preview_quality", "720p")
    modules.globals.camera_resolution = getattr(modules.globals, "camera_resolution", modules.globals.preview_quality)
    modules.globals.virtual_camera = getattr(modules.globals, "virtual_camera", False)
    modules.globals.virtual_background = getattr(modules.globals, "virtual_background", True)
    modules.globals.virtual_background_path = getattr(modules.globals, "virtual_background_path", None)
    modules.globals.smart_fps = getattr(modules.globals, "smart_fps", True)
    modules.globals.smart_fps_minimum = getattr(modules.globals, "smart_fps_minimum", 18)
    modules.globals.virtual_background_interval = getattr(modules.globals, "virtual_background_interval", 2)
    modules.globals.full_head_coverage = getattr(modules.globals, "full_head_coverage", True)
    modules.globals.mask_profile = getattr(modules.globals, "mask_profile", "Chin")
    modules.globals.trained_identity_mode = getattr(modules.globals, "trained_identity_mode", False)
    modules.globals.show_diagnostics = getattr(modules.globals, "show_diagnostics", False)
    modules.globals.quality_profile = getattr(modules.globals, "quality_profile", "Quality")
    modules.globals.last_source_media_path = getattr(modules.globals, "last_source_media_path", None)
    modules.globals.last_source_cache_path = getattr(modules.globals, "last_source_cache_path", None)
    modules.globals.last_camera_name = getattr(modules.globals, "last_camera_name", None)
    try:
        with open("switch_states.json", "r", encoding="utf-8") as f:
            state = json.load(f)
        modules.globals.keep_fps = state.get("keep_fps", True)
        modules.globals.keep_audio = state.get("keep_audio", True)
        modules.globals.keep_frames = state.get("keep_frames", False)
        modules.globals.many_faces = state.get("many_faces", False)
        modules.globals.map_faces = state.get("map_faces", False)
        # The fast affine blend already owns the mask. A second wide Poisson
        # blend can reveal a translucent rectangular face area on skin.
        modules.globals.poisson_blend = False
        modules.globals.color_correction = state.get("color_correction", False)
        modules.globals.nsfw_filter = state.get("nsfw_filter", False)
        modules.globals.live_mirror = state.get("live_mirror", False)
        modules.globals.live_resizable = state.get("live_resizable", False)
        saved_enhancer = state.get("fp_ui", {})
        if not isinstance(saved_enhancer, dict):
            saved_enhancer = {}
        # 512 is a deliberate quality choice and should survive a restart.
        # 1024 never starts automatically: its live-frame cost is too high.
        use_512 = bool(saved_enhancer.get("face_enhancer_gpen512", False))
        modules.globals.fp_ui = {
            "face_enhancer": False,
            "face_enhancer_gpen256": not use_512,
            "face_enhancer_gpen512": use_512,
            "face_enhancer_gpen1024": False,
        }
        modules.globals.show_fps = state.get("show_fps", False)
        modules.globals.virtual_camera = bool(state.get("virtual_camera", False))
        modules.globals.virtual_background = bool(state.get("virtual_background", True))
        modules.globals.smart_fps = bool(state.get("smart_fps", True))
        try:
            modules.globals.smart_fps_minimum = max(10, min(30, int(state.get("smart_fps_minimum", 18))))
            modules.globals.virtual_background_interval = max(1, min(6, int(state.get("virtual_background_interval", 2))))
        except (TypeError, ValueError):
            modules.globals.smart_fps_minimum = 18
            modules.globals.virtual_background_interval = 2
        saved_background = state.get("virtual_background_path")
        if isinstance(saved_background, str) and os.path.isfile(saved_background):
            modules.globals.virtual_background_path = saved_background
        modules.globals.trained_identity_mode = bool(state.get("trained_identity_mode", False))
        saved_mask_profile = str(state.get("mask_profile", "Chin"))
        modules.globals.mask_profile = saved_mask_profile if saved_mask_profile in ("Tight", "Chin", "Full") else "Chin"
        # Full-head is experimental and must never become the silent startup
        # default. The normal face swap is the reliable, high-FPS mode; users
        # can explicitly enable the beta button during a session when needed.
        modules.globals.full_head_mode = False
        try:
            modules.globals.opacity = max(0.0, min(1.0, float(state.get("opacity", 1.0))))
        except (TypeError, ValueError):
            modules.globals.opacity = 1.0
        try:
            modules.globals.brightness = max(-100.0, min(100.0, float(state.get("brightness", 0.0))))
            modules.globals.contrast = max(0.50, min(1.50, float(state.get("contrast", 1.0))))
            modules.globals.saturation = max(0.0, min(2.0, float(state.get("saturation", 1.0))))
            modules.globals.gamma = max(0.50, min(1.80, float(state.get("gamma", 1.0))))
            modules.globals.digital_vibrance = max(-100.0, min(100.0, float(state.get("digital_vibrance", 0.0))))
            # Return to the validated baseline. The optional texture pass had
            # been set to 100 and changes identity detail too strongly.
            modules.globals.texture_preservation = 0.0
        except (TypeError, ValueError):
            modules.globals.brightness = 0.0
            modules.globals.contrast = 1.0
            modules.globals.saturation = 1.0
            modules.globals.gamma = 1.0
            modules.globals.digital_vibrance = 0.0
            modules.globals.texture_preservation = 0.0
        modules.globals.face_swapper_enabled = modules.globals.opacity > 0.0
        try:
            saved_feather = max(0.0, min(100.0, float(state.get("mask_feather", 50.0))))
            # A near-zero feather exposes the rectangular lower edge of a
            # quick face swap, especially against bare skin. Start from a
            # natural soft edge; the slider remains available for tuning.
            modules.globals.mask_feather = 18.0 if saved_feather < 8.0 else saved_feather
        except (TypeError, ValueError):
            modules.globals.mask_feather = 50.0
        modules.globals.color_match = bool(state.get("color_match", False))
        try:
            modules.globals.face_stabilization = max(0.0, min(90.0, float(state.get("face_stabilization", 0.0))))
            modules.globals.eye_protection = max(0.0, min(100.0, float(state.get("eye_protection", 85.0))))
            modules.globals.face_offset_x = max(-100.0, min(100.0, float(state.get("face_offset_x", 0.0))))
            modules.globals.face_offset_y = max(-100.0, min(100.0, float(state.get("face_offset_y", 0.0))))
            modules.globals.face_scale = max(0.60, min(1.40, float(state.get("face_scale", 1.0))))
            modules.globals.face_rotation = max(-30.0, min(30.0, float(state.get("face_rotation", 0.0))))
        except (TypeError, ValueError):
            modules.globals.face_stabilization = 0.0
            modules.globals.eye_protection = 85.0
            modules.globals.face_offset_x = 0.0
            modules.globals.face_offset_y = 0.0
            modules.globals.face_scale = 1.0
            modules.globals.face_rotation = 0.0
        modules.globals.performance_mode = state.get("performance_mode", "Balanced")
        if modules.globals.performance_mode not in ("Quality", "Balanced", "Performance"):
            modules.globals.performance_mode = "Balanced"
        try:
            modules.globals.target_fps = int(state.get("target_fps", 20))
        except (TypeError, ValueError):
            modules.globals.target_fps = 20
        if modules.globals.target_fps not in (15, 20, 24, 25, 30, 60):
            modules.globals.target_fps = 20
        q = str(state.get("preview_quality", "720p"))
        modules.globals.preview_quality = q if q in QUALITY_OPTIONS else "720p"
        resolution = str(state.get("camera_resolution", q))
        modules.globals.camera_resolution = resolution if resolution in QUALITY_OPTIONS else "720p"
        profile = str(state.get("quality_profile", "Quality"))
        modules.globals.quality_profile = profile if profile in ("Quality", "Balanced", "Max FPS") else "Quality"
        modules.globals.last_source_media_path = state.get("last_source_media_path")
        modules.globals.last_source_cache_path = state.get("last_source_cache_path")
        modules.globals.source_profile_from_video = bool(state.get("source_profile_from_video", False))
        modules.globals.last_camera_name = state.get("last_camera_name")
        cached_source = modules.globals.last_source_cache_path
        if isinstance(cached_source, str) and os.path.isfile(cached_source) and is_image(cached_source):
            modules.globals.source_path = cached_source
            profile_path = os.path.join(os.path.dirname(cached_source), "video_source_embedding.npy")
            if modules.globals.source_profile_from_video and is_video(str(modules.globals.last_source_media_path or "")):
                try:
                    profile = np.load(profile_path).astype(np.float32).reshape(-1)
                    modules.globals.source_profile_embedding = profile if profile.size and np.all(np.isfinite(profile)) else None
                except (OSError, ValueError):
                    modules.globals.source_profile_embedding = None
            else:
                modules.globals.source_profile_embedding = None
        elif isinstance(modules.globals.last_source_media_path, str) and is_image(modules.globals.last_source_media_path):
            modules.globals.source_path = modules.globals.last_source_media_path
        if not (
            isinstance(modules.globals.last_source_media_path, str)
            and os.path.isfile(modules.globals.last_source_media_path)
            and is_video(modules.globals.last_source_media_path)
        ):
            modules.globals.full_head_mode = False
        lang = str(state.get("ui_language", "en")).lower()
        _UI_LANGUAGE = "ru" if lang in ("ru", "russian") else "en"
        try:
            modules.globals.sharpness = max(0.0, min(5.0, float(state.get("sharpness", 0.0))))
        except (TypeError, ValueError):
            modules.globals.sharpness = 0.0
        try:
            modules.globals.mouth_mask_size = max(
                0.0, min(100.0, float(state.get("mouth_mask_size", 0.0)))
            )
        except (TypeError, ValueError):
            modules.globals.mouth_mask_size = 0.0
        modules.globals.mouth_mask = modules.globals.mouth_mask_size > 0.0
        modules.globals.show_mouth_mask_box = False
        # Use the full affine face crop so the lower beard edge is included.
        # It does not move or enlarge the crop beyond the face/upper-jaw area.
        modules.globals.full_head_coverage = True
        # The landmark outline and static source-beard experiments caused
        # visible artefacts with live expressions. Keep the validated mask.
        modules.globals.adaptive_mask = False
        modules.globals.show_diagnostics = bool(state.get("show_diagnostics", False))
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError):
        pass


# ─── thread-safe status bridge ───────────────────────────────────────────


class _UIBridge(QObject):
    """Single QObject that owns cross-thread signals."""

    statusChanged = Signal(str)


def _emit_status(text: str) -> None:
    if _BRIDGE is None:
        print(text)
        return
    _BRIDGE.statusChanged.emit(text)


# ─── public API ──────────────────────────────────────────────────────────


def update_status(text: str) -> None:
    """Thread-safe status update — uses signal if called off-UI thread."""
    _emit_status(_(text))
    if _APP is not None and QThread.currentThread() is _APP.thread():
        # On UI thread — flush events so the user sees the update during
        # long synchronous start() runs.
        _APP.processEvents()


def _choose_source_video_frame(path: str) -> Optional[Tuple[np.ndarray, float, Optional[np.ndarray]]]:
    """Pick a sharp, frontal and correctly exposed source face from a video.

    This runs once at source selection; webcam FPS is unaffected.
    """
    cap = cv2.VideoCapture(path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return None
        # Avoid title cards and fade-outs while covering enough of a talking
        # head clip to find a natural, frontal expression.
        positions = np.unique(np.linspace(total * .03, total * .90, 15).astype(int))
        best_frame: Optional[np.ndarray] = None
        best_score = -1.0
        profile_samples: List[Tuple[float, np.ndarray]] = []
        for position in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            face = get_one_face(frame)
            if face is None:
                continue
            x0, y0, x1, y1 = [float(v) for v in face.bbox]
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(frame.shape[1], int(x1)), min(frame.shape[0], int(y1))
            if x1 - x0 < 48 or y1 - y0 < 48:
                continue
            gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            exposure, contrast = float(gray.mean()), float(gray.std())
            sharp_score = min(1.0, np.log1p(sharpness) / np.log1p(350.0))
            exposure_score = max(0.0, 1.0 - abs(exposure - 122.0) / 122.0)
            contrast_score = min(1.0, contrast / 52.0)
            area_score = min(1.0, np.sqrt(((x1 - x0) * (y1 - y0)) / (frame.shape[0] * frame.shape[1] * .11)))
            frontal_score = 1.0
            kps = getattr(face, "kps", None)
            if kps is not None and len(kps) >= 3:
                kps = np.asarray(kps, dtype=np.float32)
                eye_distance = float(np.linalg.norm(kps[1] - kps[0]))
                if eye_distance > 1.0:
                    eyes_mid = (kps[0] + kps[1]) * .5
                    yaw = abs(float(kps[2, 0] - eyes_mid[0])) / eye_distance
                    roll = abs(float(kps[1, 1] - kps[0, 1])) / eye_distance
                    frontal_score = max(0.0, 1.0 - min(1.0, yaw / .34 + roll / .26))
            score = (.34 * sharp_score + .25 * frontal_score + .18 * exposure_score
                     + .11 * contrast_score + .12 * area_score)
            embedding = getattr(face, "normed_embedding", None)
            if embedding is not None and frontal_score >= .55:
                embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
                if embedding.size and np.all(np.isfinite(embedding)):
                    profile_samples.append((score, embedding))
            if score > best_score:
                best_frame, best_score = frame.copy(), score
        profile = None
        if profile_samples:
            # A small quality-weighted average avoids a single speaking frame
            # pulling identity toward a blink, blur or extreme head turn.
            chosen = sorted(profile_samples, key=lambda item: item[0], reverse=True)[:8]
            weights = np.asarray([max(.05, item[0]) for item in chosen], dtype=np.float32)
            stacked = np.stack([item[1] for item in chosen], axis=0)
            profile = (stacked * weights[:, None]).sum(axis=0) / weights.sum()
            norm = float(np.linalg.norm(profile))
            profile = profile / norm if norm > 1e-6 else None
        return (best_frame, best_score, profile) if best_frame is not None else None
    finally:
        cap.release()


def check_and_ignore_nsfw(target, destroy: Optional[Callable] = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_frame, predict_image, predict_video

    check_nsfw = None
    if isinstance(target, str):
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("Processing ignored!")
        return True
    return False


# ─── camera enumeration (unchanged from tk version) ──────────────────────


def get_available_cameras() -> Tuple[List[int], List[str]]:
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if devices:
                return list(range(len(devices))), devices
            return [], ["No cameras found"]
        except Exception as exc:
            print(f"Error detecting cameras: {exc}")
            return [], ["No cameras found"]

    if platform.system() == "Darwin":
        return [0, 1], ["Camera 0", "Camera 1"]

    # Linux probe
    indices: List[int] = []
    names: List[str] = []
    for i in range(10):
        cap = cv2.VideoCapture(f"/dev/video{i}")
        if cap.isOpened():
            indices.append(i)
            names.append(f"Camera {i}")
            cap.release()
    return (indices, names) if names else ([], ["No cameras found"])


# ─── main window ─────────────────────────────────────────────────────────


def _make_image_drop(text: str, size: Tuple[int, int]) -> QLabel:
    label = QLabel(text)
    label.setObjectName("imageDrop")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFixedSize(size[0], size[1])
    label.setText(text)
    return label


class _Switch(QWidget):
    """Compact toggle switch with label + optional tooltip."""

    toggled = Signal(bool)

    def __init__(self, text: str, initial: bool, tooltip: str = ""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._checkbox = QCheckBox(text)
        self._checkbox.setChecked(initial)
        self._checkbox.toggled.connect(self.toggled.emit)
        if tooltip:
            self._checkbox.setToolTip(tooltip)
        layout.addWidget(self._checkbox)
        layout.addStretch(1)

    def isChecked(self) -> bool:
        return self._checkbox.isChecked()

    def setChecked(self, value: bool) -> None:
        self._checkbox.setChecked(value)

    def setText(self, text: str) -> None:
        self._checkbox.setText(text)


class CameraHeroFrame(QFrame):
    """Camera page header with a dedicated decorative background asset."""

    def __init__(self) -> None:
        super().__init__()
        asset_root = os.path.dirname(os.path.dirname(__file__))
        self._camera_background = QPixmap(
            os.path.join(asset_root, "assets", "camera-header-art-4x3.png")
        )

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = self.rect().adjusted(0, 0, -1, -1)
        clip = QPainterPath()
        clip.addRoundedRect(bounds, 10, 10)
        painter.save()
        painter.setClipPath(clip)

        base = QLinearGradient(0, 0, self.width(), 0)
        base.setColorAt(0.0, QColor("#050b1b"))
        base.setColorAt(0.72, QColor("#07122e"))
        base.setColorAt(1.0, QColor("#160c3f"))
        painter.fillRect(self.rect(), base)

        # Crop the original artwork to 4:3 before putting it into the header.
        # It therefore never gets squeezed horizontally into a thin strip.
        if not self._camera_background.isNull():
            # The camera occupies the far-right portion of the original
            # artwork.  Crop around the whole body and lens, rather than
            # taking the right edge where only part of the lens remains.
            source_width = min(650, self._camera_background.width())
            source_height = min(488, self._camera_background.height())
            source_x = max(0, self._camera_background.width() - source_width)
            source_y = max(0, min(320, self._camera_background.height() - source_height))
            target_height = self.height()
            target_width = round(target_height * 4 / 3)
            # The artwork itself is dark; 50% painter opacity produces the
            # requested subdued (roughly 30% perceived) decorative presence
            # while keeping the complete camera recognisable at header size.
            painter.setOpacity(0.50)
            painter.drawPixmap(
                self.width() - target_width,
                source_y,
                target_width,
                target_height,
                self._camera_background,
                source_x,
                0,
                source_width,
                source_height,
            )
            painter.setOpacity(1.0)

        # Preserve a dark readable field under the heading.
        readable = QLinearGradient(0, 0, self.width(), 0)
        readable.setColorAt(0.0, QColor(5, 12, 27, 90))
        readable.setColorAt(0.56, QColor(5, 12, 27, 36))
        readable.setColorAt(0.78, QColor(5, 12, 27, 0))
        readable.setColorAt(1.0, QColor(5, 12, 27, 0))
        painter.fillRect(self.rect(), readable)

        painter.restore()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#486fc7"), 1))
        painter.drawRoundedRect(bounds, 10, 10)


class CameraHeroIcon(QWidget):
    """Compact UI camera glyph matching the reference header."""

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(Qt.PenStyle.NoPen)
        glow = QRadialGradient(bounds.center(), bounds.width() * 0.55)
        glow.setColorAt(0.0, QColor(109, 75, 255, 125))
        glow.setColorAt(0.58, QColor(76, 58, 235, 45))
        glow.setColorAt(1.0, QColor(76, 58, 235, 0))
        painter.setBrush(glow)
        painter.drawEllipse(bounds)
        inner = bounds.adjusted(4, 4, -4, -4)
        gradient = QLinearGradient(inner.left(), inner.top(), inner.right(), inner.bottom())
        gradient.setColorAt(0.0, QColor("#5269ff"))
        gradient.setColorAt(0.54, QColor("#7554f4"))
        gradient.setColorAt(1.0, QColor("#a94cf2"))
        painter.setPen(QPen(QColor("#a397ff"), 1))
        painter.setBrush(gradient)
        painter.drawRoundedRect(inner, 8, 8)
        painter.setPen(QPen(QColor("#fbf9ff"), 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        body = inner.adjusted(8, 11, -8, -8)
        painter.drawRoundedRect(body, 4, 4)
        painter.drawEllipse(body.center(), 5.0, 5.0)
        painter.drawLine(body.left() + 5, body.top(), body.left() + 9, body.top() - 4)
        painter.drawLine(body.left() + 9, body.top() - 4, body.left() + 16, body.top() - 4)


class MainWindow(QMainWindow):
    def __init__(self, start_cb: Callable, destroy_cb: Callable):
        super().__init__()
        load_switch_states()
        self._start_cb = start_cb
        self._destroy_cb = destroy_cb

        self.setWindowTitle("Deep-Live-Cam by Syrex — GPEN-1024")
        self.setMinimumSize(1100, 620)
        self.resize(1280, 760)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        # The window is normally mounted inside Studio, so do not reserve a
        # second title-sized strip above its own hero header.
        root_layout.setContentsMargins(10, 2, 10, 8)
        root_layout.setSpacing(10)

        # Page identity belongs to the working area, not to a separate
        # screen.  It mirrors the Studio shell while leaving all existing
        # camera controls and handlers in place below it.
        page_hero = CameraHeroFrame()
        page_hero.setObjectName("cameraHero")
        page_hero.setFixedHeight(76)
        hero_layout = QHBoxLayout(page_hero)
        hero_layout.setContentsMargins(14, 7, 10, 7)
        hero_layout.setSpacing(10)
        hero_icon = CameraHeroIcon()
        hero_icon.setObjectName("cameraHeroIcon")
        hero_icon.setFixedSize(46, 46)
        hero_layout.addWidget(hero_icon, 0, Qt.AlignmentFlag.AlignVCenter)
        hero_text = QVBoxLayout()
        hero_text.setSpacing(1)
        hero_title = QLabel("Камера / Face Swap")
        self.camera_hero_title = hero_title
        hero_title.setObjectName("cameraHeroTitle")
        hero_subtitle = QLabel("Настройка камеры, замена лица и улучшение видео в реальном времени")
        self.camera_hero_subtitle = hero_subtitle
        hero_subtitle.setObjectName("cameraHeroSubtitle")
        hero_text.addWidget(hero_title)
        hero_text.addWidget(hero_subtitle)
        hero_layout.addLayout(hero_text, 1)
        breadcrumb = QLabel("Студия   ›   Камера / Face Swap")
        self.camera_breadcrumb = breadcrumb
        breadcrumb.setObjectName("cameraBreadcrumb")
        hero_layout.addWidget(breadcrumb, 0, Qt.AlignmentFlag.AlignVCenter)
        root_layout.addWidget(page_hero)

        # Main application layout: compact controls on the left, roomy media
        # dashboard on the right. Only the control column scrolls.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter = splitter
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        root_layout.addWidget(splitter, 1)

        # ── left control column ──────────────────────────────────────────
        control_scroll = QScrollArea()
        control_scroll.setObjectName("controlScroll")
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # Do not let wide controls paint under the splitter at narrow widths.
        # A horizontal scrollbar is preferable to clipped fields or borders.
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        control_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        control_scroll.setMinimumWidth(360)
        control_scroll.setMinimumHeight(0)
        control_scroll.setMaximumWidth(800)
        control_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        control_content = QWidget()
        control_content.setObjectName("controlContent")
        left_layout = QVBoxLayout(control_content)
        left_layout.setContentsMargins(2, 2, 4, 2)
        left_layout.setSpacing(6)
        left_layout.addWidget(self._build_options_card())
        left_layout.addWidget(self._build_sliders_card())
        left_layout.addWidget(self._build_advanced_card())
        left_layout.addStretch(1)
        control_scroll.setWidget(control_content)

        # Keep the actions and the active camera selector reachable even when
        # the long controls list is scrolled.  This is layout-only: every
        # existing button and combo retains its original widget and callback.
        controls_column = QWidget()
        controls_column.setMinimumHeight(0)
        controls_layout = QVBoxLayout(controls_column)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)
        controls_layout.addWidget(control_scroll, 1)
        controls_footer = QFrame()
        controls_footer.setObjectName("controlsFooter")
        footer_layout = QVBoxLayout(controls_footer)
        footer_layout.setContentsMargins(6, 6, 6, 6)
        footer_layout.setSpacing(10)
        footer_layout.addLayout(self._build_action_row())
        footer_layout.addWidget(self._build_camera_card())
        # The footer contains two 28px action rows and a 74px camera card.
        # Give it its real height explicitly so Qt never clips the camera
        # card under the action buttons when Studio is vertically constrained.
        controls_footer.setFixedHeight(178)
        controls_layout.addWidget(controls_footer, 0)
        splitter.addWidget(controls_column)

        # ── right media/dashboard column ─────────────────────────────────
        dashboard = QWidget()
        dashboard.setObjectName("dashboard")
        dash = QVBoxLayout(dashboard)
        dash.setContentsMargins(0, 0, 2, 0)
        dash.setSpacing(10)

        # Source / target selection strip.
        media_strip = QHBoxLayout()
        media_strip.setSpacing(10)

        src_card = QGroupBox(_("Source Face"))
        src_card.setObjectName("mediaSourceCard")
        self.src_card = src_card
        src_layout = QVBoxLayout(src_card)
        self.source_label = _make_image_drop(_("Source face"), (280, 180))
        self.source_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        src_layout.addWidget(self.source_label)
        src_buttons = QHBoxLayout()
        self.btn_select_source = QPushButton(_("Select a face"))
        self.btn_select_source.clicked.connect(lambda _checked=False: self._on_select_source(video_only=False))
        self.btn_random_face = QPushButton("🔄")
        self.btn_random_face.setObjectName("secondary")
        self.btn_random_face.setFixedWidth(42)
        self.btn_random_face.clicked.connect(self._on_random_face)
        src_buttons.addWidget(self.btn_select_source, 1)
        src_buttons.addWidget(self.btn_random_face)
        src_layout.addLayout(src_buttons)
        self.btn_select_video = QPushButton("Видео")
        self.btn_select_video.setObjectName("secondary")
        self.btn_select_video.setMinimumHeight(30)
        self.btn_select_video.setToolTip("Выбрать исходное видео; программа выберет лучший кадр лица")
        self.btn_select_video.clicked.connect(lambda _checked=False: self._on_select_source(video_only=True))
        src_layout.addWidget(self.btn_select_video)
        self.btn_full_head = QPushButton("Полная голова (β)")
        self.btn_full_head.setObjectName("secondary")
        self.btn_full_head.setMinimumHeight(30)
        self.btn_full_head.setToolTip("Полная голова из исходного видео: волосы, уши, борода и мимика. Требуется выбрать видео.")
        self.btn_full_head.clicked.connect(self._on_toggle_full_head)
        src_layout.addWidget(self.btn_full_head)
        self.btn_trained_identity = QPushButton("Обученная личность (DFM)")
        self.btn_trained_identity.setObjectName("secondary")
        self.btn_trained_identity.setMinimumHeight(30)
        self.btn_trained_identity.setToolTip("Использовать локально обученную модель личности вместо обычной замены по фото")
        self.btn_trained_identity.clicked.connect(self._on_toggle_trained_identity)
        src_layout.addWidget(self.btn_trained_identity)
        media_strip.addWidget(src_card, 1)

        swap_col = QVBoxLayout()
        swap_col.addStretch(1)
        self.btn_swap = QPushButton("↔")
        self.btn_swap.setObjectName("secondary")
        self.btn_swap.setFixedSize(44, 44)
        self.btn_swap.clicked.connect(self._on_swap_paths)
        swap_col.addWidget(self.btn_swap, alignment=Qt.AlignmentFlag.AlignCenter)
        swap_col.addStretch(1)
        media_strip.addLayout(swap_col)

        tgt_card = QGroupBox(_("Target"))
        tgt_card.setObjectName("mediaTargetCard")
        self.tgt_card = tgt_card
        tgt_layout = QVBoxLayout(tgt_card)
        self.target_label = _make_image_drop(_("Target"), (280, 180))
        self.target_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        tgt_layout.addWidget(self.target_label)
        self.btn_select_target = QPushButton(_("Select a target"))
        self.btn_select_target.clicked.connect(self._on_select_target)
        tgt_layout.addWidget(self.btn_select_target)
        media_strip.addWidget(tgt_card, 1)
        live_card = QGroupBox()
        live_card.setObjectName("liveSwapCard")
        self.live_swap_card = live_card
        live_layout = QVBoxLayout(live_card)
        live_layout.setContentsMargins(12, 8, 12, 10)
        live_layout.setSpacing(8)
        title = QLabel(_("Live Face Swap"))
        self.dashboard_title = title
        title.setObjectName("dashboardTitle")
        live_layout.addWidget(title)
        live_layout.addLayout(media_strip)
        dash.addWidget(live_card)

        # Large visual stage. Live processing itself still uses the proven
        # WebcamPreviewWindow, but this keeps the main UI organized like a
        # creator dashboard and shows the current target when selected.
        preview_card = QGroupBox(_("Preview"))
        preview_card.setObjectName("previewCard")
        self.preview_card = preview_card
        pv_layout = QVBoxLayout(preview_card)
        self.dashboard_preview = QWidget()
        self.dashboard_preview.setObjectName("dashboardPreview")
        self.dashboard_preview.setMinimumHeight(220)
        self.dashboard_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.dashboard_preview_layout = QVBoxLayout(self.dashboard_preview)
        self.dashboard_preview_layout.setContentsMargins(0, 0, 0, 0)
        self.dashboard_preview_label = QLabel(_("Select a target or press LIVE\nto open the real-time preview"))
        self.dashboard_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dashboard_preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.dashboard_preview_layout.addWidget(self.dashboard_preview_label, 1)
        pv_layout.addWidget(self.dashboard_preview, 1)
        dash.addWidget(preview_card, 1)

        # Quick status and live controls.
        quick = QHBoxLayout()
        self._dashboard_fps = QLabel(_("FPS: —"))
        self._dashboard_fps.setObjectName("metricLabel")
        quick.addWidget(self._dashboard_fps)
        quick.addStretch(1)
        self.btn_dashboard_live = QPushButton(_("Live"))
        self.btn_dashboard_live.setObjectName("liveButton")
        self.btn_dashboard_live.clicked.connect(self._on_live)
        quick.addWidget(self.btn_dashboard_live)
        self.btn_dashboard_preview = QPushButton("Preview")
        self.btn_dashboard_preview.setObjectName("secondary")
        self.btn_dashboard_preview.clicked.connect(self._on_toggle_preview)
        quick.addWidget(self.btn_dashboard_preview)
        dash.addLayout(quick)

        # Bottom capture settings wrap into two rows.  The former one-line
        # layout forced the dashboard wider than the window and squeezed the
        # left cards underneath the splitter.
        bottom = QGridLayout()
        bottom.setHorizontalSpacing(10)
        bottom.setVerticalSpacing(5)

        self.lbl_resolution = QLabel(_("Camera resolution:"))
        bottom.addWidget(self.lbl_resolution, 0, 0)
        self.cb_resolution = QComboBox()
        for preset, (width, height) in QUALITY_OPTIONS.items():
            ratio = "4:3" if "4:3" in preset else "16:9"
            self.cb_resolution.addItem(f"{width} × {height}  ·  {ratio}", preset)
        saved_resolution = getattr(modules.globals, "camera_resolution", "720p")
        saved_index = self.cb_resolution.findData(saved_resolution)
        self.cb_resolution.setCurrentIndex(saved_index if saved_index >= 0 else self.cb_resolution.findData("720p"))
        self.cb_resolution.currentIndexChanged.connect(
            lambda _index: self._on_resolution_change(str(self.cb_resolution.currentData() or "720p"))
        )
        self.cb_resolution.setToolTip("Формат захвата и вывода камеры. Применяется при следующем запуске LIVE.")
        bottom.addWidget(self.cb_resolution, 0, 1, 1, 3)

        self.lbl_quality = QLabel(_("Processing quality:"))
        bottom.addWidget(self.lbl_quality, 1, 0)
        self.cb_quality = QComboBox()
        for label, preset in PROCESSING_QUALITY_OPTIONS.items():
            self.cb_quality.addItem(label, preset)
        saved_quality = getattr(modules.globals, "preview_quality", "720p")
        quality_index = self.cb_quality.findData(saved_quality)
        self.cb_quality.setCurrentIndex(quality_index if quality_index >= 0 else self.cb_quality.findData("720p"))
        self.cb_quality.currentIndexChanged.connect(
            lambda _index: self._on_quality_change(str(self.cb_quality.currentData() or "720p"))
        )
        self.cb_quality.setToolTip("Детализация обработки Face Swap. Не меняет разрешение камеры.")
        bottom.addWidget(self.cb_quality, 1, 1)

        self.lbl_language = QLabel(_("Language:"))
        bottom.addWidget(self.lbl_language, 1, 2)
        self.cb_language = QComboBox()
        self.cb_language.addItems(["English", "Russian"])
        self.cb_language.setCurrentText("Russian" if _UI_LANGUAGE == "ru" else "English")
        self.cb_language.currentTextChanged.connect(self._on_language_change)
        self.cb_language.setToolTip("Restart the application after changing language")
        bottom.addWidget(self.cb_language, 1, 3)
        bottom.setColumnStretch(1, 1)
        bottom.setColumnStretch(3, 1)
        dash.addLayout(bottom)

        self._status_label = QLabel("")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dash.addWidget(self._status_label)

        splitter.addWidget(dashboard)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([770, 880])

        # Apply the selected language immediately on first construction.
        self._retranslate_ui()
        QTimer.singleShot(0, self._restore_saved_source_preview)

    def _restore_saved_source_preview(self) -> None:
        """Show the persisted source immediately when the app is reopened."""
        path = modules.globals.source_path
        if isinstance(path, str) and os.path.isfile(path) and is_image(path):
            self.source_label.setPixmap(render_image_preview(path, (280, 180), crop=False))
            self.source_label.setText("")

    def _retranslate_ui(self) -> None:
        """Update visible UI strings immediately when the language changes."""
        # Main title intentionally stays branded and unchanged.
        self.setWindowTitle("Deep-Live-Cam by Syrex — GPEN-1024")
        if _UI_LANGUAGE == "ru":
            self.camera_hero_title.setText("Камера / Face Swap")
            self.camera_hero_subtitle.setText(
                "Настройка камеры, замена лица и улучшение видео в реальном времени"
            )
            self.camera_breadcrumb.setText("Студия   ›   Камера / Face Swap")
        else:
            self.camera_hero_title.setText("Camera / Face Swap")
            self.camera_hero_subtitle.setText(
                "Camera setup, face replacement and real-time video enhancement"
            )
            self.camera_breadcrumb.setText("Studio   ›   Camera / Face Swap")

        # Section titles / labels.
        self.options_card.setTitle(_("Options"))
        self.refinement_card.setTitle(_("Refinement"))
        self.advanced_card.setTitle(_("Advanced"))
        self.camera_card.setTitle(_("Camera"))
        self.src_card.setTitle(_("Source Face"))
        self.tgt_card.setTitle(_("Target"))
        self.preview_card.setTitle(_("Preview"))
        self.dashboard_title.setText(_("Live Face Swap"))

        # Main options.
        self.sw_keep_fps.setText(_("Keep fps"))
        self.sw_keep_audio.setText(_("Keep audio"))
        self.sw_keep_frames.setText(_("Keep frames"))
        self.sw_many_faces.setText(_("Many faces"))
        self.sw_map_faces.setText(_("Map faces"))
        self.sw_show_fps.setText(_("Show FPS"))
        self.sw_virtual_camera.setText(_("Virtual Camera"))
        self.sw_virtual_background.setText(_("Virtual Background"))
        self.sw_smart_fps.setText(_("Smart FPS"))
        self.sw_poisson.setText(_("Poisson Blend"))
        self.sw_color_fix.setText(_("Fix Blueish Cam"))
        self.lbl_enhancer.setText(_("Face Enhancer:"))
        self.btn_select_virtual_background.setText(_("Choose background"))

        # Refinement labels.
        self.lbl_transparency.setText(_("Transparency"))
        self.lbl_sharpness.setText(_("Sharpness"))
        self.lbl_mouth_mask.setText(_("Mouth Mask"))
        self.lbl_mask_feather.setText(_("Edge Softness"))

        # Advanced labels.
        self.sw_color_match.setText(_("Color Match"))
        self.lbl_performance.setText(_("Performance Mode"))
        self.lbl_stabilization.setText(_("Face Stabilization"))
        self.lbl_eye.setText(_("Eye Protection"))
        self.lbl_offset_x.setText(_("Face X"))
        self.lbl_offset_y.setText(_("Face Y"))
        self.lbl_scale.setText(_("Face Scale"))
        self.lbl_rotation.setText(_("Face Rotation"))
        self.lbl_brightness.setText("Яркость" if _UI_LANGUAGE == "ru" else "Brightness")
        self.lbl_contrast.setText("Контраст" if _UI_LANGUAGE == "ru" else "Contrast")
        self.lbl_saturation.setText("Насыщенность" if _UI_LANGUAGE == "ru" else "Saturation")
        self.lbl_gamma.setText("Гамма" if _UI_LANGUAGE == "ru" else "Gamma")
        self.lbl_vibrance.setText("Цифровая интенсивность" if _UI_LANGUAGE == "ru" else "Digital Vibrance")
        self.lbl_texture.setText("Сохранение текстуры" if _UI_LANGUAGE == "ru" else "Texture Preserve")
        self.lbl_target_fps.setText(_("Target FPS"))

        # Buttons.
        self.btn_select_source.setText(_("Select a face"))
        if hasattr(self, "btn_select_video"):
            self.btn_select_video.setText("Видео" if _UI_LANGUAGE == "ru" else "Video")
        if hasattr(self, "btn_full_head"):
            active = bool(getattr(modules.globals, "full_head_mode", False))
            self.btn_full_head.setText(
                ("Полная голова: ВКЛ" if active else "Полная голова (β)")
                if _UI_LANGUAGE == "ru" else
                ("Full head: ON" if active else "Full head (β)")
            )
        if hasattr(self, "btn_trained_identity"):
            active = bool(getattr(modules.globals, "trained_identity_mode", False))
            self.btn_trained_identity.setText(
                ("Обученная личность: ВКЛ" if active else "Обученная личность (DFM)")
                if _UI_LANGUAGE == "ru" else
                ("Trained identity: ON" if active else "Trained identity (DFM)")
            )
        self.btn_select_target.setText(_("Select a target"))
        self.btn_start.setText(_("Start"))
        self.btn_destroy.setText(_("Stop video"))
        self.btn_preview.setText(_("Preview"))
        if hasattr(self, "btn_output"):
            self.btn_output.setText(
                _("Close")
                if _OUTPUT_WINDOW is not None and _OUTPUT_WINDOW.isVisible()
                else _("Output")
            )
        self.btn_live.setText(_("Live"))
        self.btn_dashboard_live.setText(_("Live"))
        self.btn_dashboard_preview.setText(_("Preview"))
        self.lbl_camera.setText(_("Select Camera:"))
        self.lbl_resolution.setText(_("Camera resolution:"))
        self.lbl_quality.setText(_("Processing quality:"))
        self.lbl_language.setText(_("Language:"))
        self.dashboard_preview_label.setText(_("Select a target or press LIVE\nto open the real-time preview"))

        # Preserve internal English keys while translating visible performance choices.
        perf_value = getattr(modules.globals, "performance_mode", "Balanced")
        perf_labels = {
            "Quality": _("Quality"),
            "Balanced": _("Balanced"),
            "Performance": _("Performance"),
        }
        self.cb_performance.blockSignals(True)
        self.cb_performance.clear()
        for key in ("Quality", "Balanced", "Performance"):
            self.cb_performance.addItem(perf_labels[key], userData=key)
        idx = self.cb_performance.findData(perf_value)
        self.cb_performance.setCurrentIndex(max(0, idx))
        self.cb_performance.blockSignals(False)

        # Language selector itself remains bilingual and switches immediately.
        current_lang = "Russian" if _UI_LANGUAGE == "ru" else "English"
        self.cb_language.blockSignals(True)
        self.cb_language.clear()
        self.cb_language.addItem("English")
        self.cb_language.addItem("Русский" if _UI_LANGUAGE == "ru" else "Russian")
        self.cb_language.setCurrentIndex(1 if _UI_LANGUAGE == "ru" else 0)
        self.cb_language.blockSignals(False)
        self.cb_language.setToolTip(
            "Выберите язык интерфейса" if _UI_LANGUAGE == "ru" else "Select interface language"
        )

        # Face enhancer display labels.
        current_enh = self.cb_enhancer.currentData() or self.cb_enhancer.currentText()
        if current_enh in (None, ""):
            current_enh = self.cb_enhancer.currentText()
        enh_items = [
            ("None", _("None")),
            ("GFPGAN", "GFPGAN"),
            ("GPEN-512", "GPEN-512"),
            ("GPEN-256", "GPEN-256"),
            ("GPEN-1024 (slow)", "GPEN-1024 (slow)"),
        ]
        self.cb_enhancer.blockSignals(True)
        self.cb_enhancer.clear()
        for key, label in enh_items:
            self.cb_enhancer.addItem(label, userData=key)
        enh_idx = self.cb_enhancer.findData(current_enh)
        self.cb_enhancer.setCurrentIndex(max(0, enh_idx))
        self.cb_enhancer.blockSignals(False)

        # Source / target placeholders when nothing is loaded.
        if modules.globals.source_path is None:
            self.source_label.setText(_("Source face"))
        if modules.globals.target_path is None:
            self.target_label.setText(_("Target"))



    def _build_image_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(16)

        # Source column
        src_col = QVBoxLayout()
        self.source_label = _make_image_drop(_("Source face"), (200, 200))
        src_col.addWidget(self.source_label, alignment=Qt.AlignmentFlag.AlignCenter)
        src_row = QHBoxLayout()
        self.btn_select_source = QPushButton(_("Select a face"))
        self.btn_select_source.setToolTip(
            _("Choose the source face image to swap onto the target")
        )
        self.btn_select_source.clicked.connect(lambda _checked=False: self._on_select_source(video_only=False))
        self.btn_select_video = QPushButton("Видео")
        self.btn_select_video.setObjectName("secondary")
        self.btn_select_video.setToolTip("Выбрать исходное видео; программа сама возьмёт лучший кадр лица")
        self.btn_select_video.clicked.connect(lambda _checked=False: self._on_select_source(video_only=True))
        self.btn_random_face = QPushButton("🔄")
        self.btn_random_face.setObjectName("secondary")
        self.btn_random_face.setFixedWidth(40)
        self.btn_random_face.setToolTip(
            _("Get a random face from thispersondoesnotexist.com")
        )
        self.btn_random_face.clicked.connect(self._on_random_face)
        src_row.addWidget(self.btn_select_source)
        src_row.addWidget(self.btn_random_face)
        src_col.addLayout(src_row)
        # Keep the video action on its own line: narrow windows previously
        # squeezed it to zero width beside the main source button.
        self.btn_select_video.setMinimumHeight(30)
        src_col.addWidget(self.btn_select_video)

        # Swap button column
        swap_col = QVBoxLayout()
        swap_col.addStretch(1)
        self.btn_swap = QPushButton("↔")
        self.btn_swap.setObjectName("secondary")
        self.btn_swap.setFixedSize(44, 44)
        self.btn_swap.setToolTip(_("Swap source and target images"))
        self.btn_swap.clicked.connect(self._on_swap_paths)
        swap_col.addWidget(self.btn_swap, alignment=Qt.AlignmentFlag.AlignCenter)
        swap_col.addStretch(1)

        # Target column
        tgt_col = QVBoxLayout()
        self.target_label = _make_image_drop(_("Target"), (200, 200))
        tgt_col.addWidget(self.target_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.btn_select_target = QPushButton(_("Select a target"))
        self.btn_select_target.setToolTip(
            _("Choose the target image or video to apply face swap to")
        )
        self.btn_select_target.clicked.connect(self._on_select_target)
        tgt_col.addWidget(self.btn_select_target)

        row.addLayout(src_col)
        row.addLayout(swap_col)
        row.addLayout(tgt_col)
        return row

    # ── options card ─────────────────────────────────────────────────────

    def _build_options_card(self) -> QGroupBox:
        card = QGroupBox(_("Options"))
        card.setObjectName("controlCard")
        self.options_card = card
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(1)

        def make(field, label, tip):
            sw = _Switch(_(label), getattr(modules.globals, field), _(tip))
            sw.toggled.connect(
                lambda v, f=field: (
                    setattr(modules.globals, f, v),
                    save_switch_states(),
                )
            )
            return sw

        self.sw_keep_fps = make("keep_fps", "Keep fps",
                                "Output video keeps the original frame rate")
        self.sw_keep_audio = make("keep_audio", "Keep audio",
                                  "Copy audio track from the source video to output")
        self.sw_keep_frames = make("keep_frames", "Keep frames",
                                   "Keep extracted frames on disk after processing")
        self.sw_many_faces = make("many_faces", "Many faces",
                                  "Swap every detected face, not just the primary one")
        self.sw_poisson = make("poisson_blend", "Poisson Blend",
                               "Blend face edges smoothly using Poisson blending")
        self.sw_color_fix = make("color_correction", "Fix Blueish Cam",
                                 "Fix blue/green color cast from some webcams")
        self.sw_show_fps = make("show_fps", "Show FPS",
                                "Display frames-per-second counter on the live preview")
        self.sw_virtual_camera = make(
            "virtual_camera", "Virtual Camera",
            "Send the processed live camera to OBS Virtual Camera",
        )
        self.sw_virtual_background = make(
            "virtual_background", "Virtual Background",
            "Replace the camera room with the selected image in every live output",
        )
        self.sw_smart_fps = make(
            "smart_fps", "Smart FPS",
            "Keep the background on while automatically reducing its update rate if FPS falls",
        )

        # Map faces is special — closes mapper when toggled off.
        self.sw_map_faces = _Switch(_("Map faces"), modules.globals.map_faces,
                                    _("Manually assign which source face maps to which target face"))
        self.sw_map_faces.toggled.connect(self._on_map_faces_toggled)

        # Layout: 2 columns of switches
        items = [
            self.sw_keep_fps, self.sw_keep_audio,
            self.sw_keep_frames, self.sw_many_faces,
            self.sw_map_faces, self.sw_show_fps,
            self.sw_poisson, self.sw_color_fix,
            self.sw_virtual_camera, self.sw_virtual_background,
            self.sw_smart_fps,
        ]
        for i, w in enumerate(items):
            grid.addWidget(w, i // 2, i % 2)

        # Face enhancer dropdown.  ``items`` may have an odd count (the
        # Virtual Camera switch occupies its own left-hand row), so round up
        # the number of switch rows.  Using floor division here placed the
        # enhancer label on top of that switch and swallowed its mouse clicks.
        enhancer_row = (len(items) + 1) // 2
        enhancer_label = QLabel(_("Face Enhancer:"))
        self.lbl_enhancer = enhancer_label
        grid.addWidget(enhancer_label, enhancer_row, 0)

        self.cb_enhancer = QComboBox()
        self.cb_enhancer.addItems(["None", "GFPGAN", "GPEN-512", "GPEN-256", "GPEN-1024 (slow)"])
        initial = "None"
        if modules.globals.fp_ui.get("face_enhancer", False):
            initial = "GFPGAN"
        elif modules.globals.fp_ui.get("face_enhancer_gpen512", False):
            initial = "GPEN-512"
        elif modules.globals.fp_ui.get("face_enhancer_gpen1024", False):
            initial = "GPEN-1024 (slow)"
        elif modules.globals.fp_ui.get("face_enhancer_gpen256", False):
            initial = "GPEN-256"
        self.cb_enhancer.setCurrentText(initial)
        self.cb_enhancer.currentTextChanged.connect(self._on_enhancer_change)
        self.cb_enhancer.setToolTip(_("Select a face enhancement model (None = no enhancement)"))
        grid.addWidget(self.cb_enhancer, enhancer_row, 1)

        background_row = enhancer_row + 1
        self.btn_select_virtual_background = QPushButton(_("Choose background"))
        self.btn_select_virtual_background.setObjectName("secondary")
        self.btn_select_virtual_background.setToolTip(
            "Выбрать картинку для виртуального фона камеры"
        )
        self.btn_select_virtual_background.clicked.connect(self._on_select_virtual_background)
        grid.addWidget(QLabel(_("Virtual Background") + ":"), background_row, 0)
        grid.addWidget(self.btn_select_virtual_background, background_row, 1)

        return card

    # ── sliders card ─────────────────────────────────────────────────────

    def _build_sliders_card(self) -> QGroupBox:
        card = QGroupBox(_("Refinement"))
        card.setObjectName("controlCard")
        self.refinement_card = card
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(1)

        def slider(min_v, max_v, default, denom, on_change):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(int(min_v * denom), int(max_v * denom))
            s.setValue(int(default * denom))
            s.valueChanged.connect(lambda iv: on_change(iv / denom))
            return s

        # Transparency
        self.lbl_transparency = QLabel(_("Transparency"))
        grid.addWidget(self.lbl_transparency, 0, 0)
        self.s_transparency = slider(0.0, 1.0, modules.globals.opacity, 100, self._on_transparency_change)
        self.s_transparency.setToolTip(
            _("Blend between original and swapped face (0% = original, 100% = fully swapped)")
        )
        grid.addWidget(self.s_transparency, 0, 1)

        # Sharpness
        self.lbl_sharpness = QLabel(_("Sharpness"))
        grid.addWidget(self.lbl_sharpness, 1, 0)
        self.s_sharpness = slider(
            0.0, 5.0, modules.globals.sharpness, 10, self._on_sharpness_change
        )
        self.s_sharpness.setToolTip(_("Sharpen the enhanced face output"))
        grid.addWidget(self.s_sharpness, 1, 1)

        # Mouth mask restores its last selected size on launch.
        self.lbl_mouth_mask = QLabel(_("Mouth Mask"))
        grid.addWidget(self.lbl_mouth_mask, 2, 0)
        self.s_mouth = slider(0.0, 100.0, modules.globals.mouth_mask_size, 1,
                              self._on_mouth_mask_change)
        self.s_mouth.sliderPressed.connect(self._on_mouth_mask_pressed)
        self.s_mouth.sliderReleased.connect(self._on_mouth_mask_released)
        self.s_mouth.setToolTip(
            _("0 = use GPEN mouth, 100 = preserve the swapped mouth after GPEN")
        )
        grid.addWidget(self.s_mouth, 2, 1)

        # Mask Feather
        self.lbl_mask_feather = QLabel(_("Edge Softness"))
        grid.addWidget(self.lbl_mask_feather, 3, 0)
        self.s_mask_feather = slider(0.0, 100.0, getattr(modules.globals, "mask_feather", 50.0), 1, self._on_mask_feather_change)
        self.s_mask_feather.setToolTip(
            _("Softness of the face-mask edge: 0 = crisp, 100 = very soft")
        )
        grid.addWidget(self.s_mask_feather, 3, 1)

        # Geometry profiles switch on the next frame; no model reload and no
        # LIVE restart are required.  They affect only the face alpha mask.
        self.lbl_mask_profile = QLabel("Форма маски")
        grid.addWidget(self.lbl_mask_profile, 4, 0)
        self.cb_mask_profile = QComboBox()
        self.cb_mask_profile.addItem("Компактная", "Tight")
        self.cb_mask_profile.addItem("Лицо + подбородок", "Chin")
        self.cb_mask_profile.addItem("Максимальная", "Full")
        current_profile = str(getattr(modules.globals, "mask_profile", "Chin"))
        current_index = self.cb_mask_profile.findData(current_profile)
        self.cb_mask_profile.setCurrentIndex(current_index if current_index >= 0 else 1)
        self.cb_mask_profile.setToolTip(
            "Переключается сразу: меняет только границу маски, без загрузки модели"
        )
        self.cb_mask_profile.currentIndexChanged.connect(self._on_mask_profile_change)
        grid.addWidget(self.cb_mask_profile, 4, 1)
        return card

    # ── advanced controls ────────────────────────────────────────────────

    def _build_advanced_card(self) -> QGroupBox:
        card = QGroupBox(_("Advanced"))
        card.setObjectName("controlCard")
        self.advanced_card = card
        grid = QGridLayout(card)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(1)

        self.sw_color_match = _Switch(
            _("Color Match"), getattr(modules.globals, "color_match", False),
            _("Match swapped-face colour and lighting to the target")
        )
        self.sw_color_match.toggled.connect(self._on_color_match_toggled)
        grid.addWidget(self.sw_color_match, 0, 0)

        self.lbl_performance = QLabel(_("Performance Mode"))
        grid.addWidget(self.lbl_performance, 0, 1)
        self.cb_performance = QComboBox()
        self.cb_performance.addItems(["Quality", "Balanced", "Performance"])
        self.cb_performance.setCurrentText(getattr(modules.globals, "performance_mode", "Balanced"))
        self.cb_performance.currentTextChanged.connect(self._on_performance_mode_display_change)
        grid.addWidget(self.cb_performance, 0, 2)

        def sld(label, lo, hi, value, denom, handler, tip, row):
            label_widget = QLabel(_(label))
            grid.addWidget(label_widget, row, 0)
            setattr(self, {
                "Face Stabilization": "lbl_stabilization",
                "Eye Protection": "lbl_eye",
                "Face X": "lbl_offset_x",
                "Face Y": "lbl_offset_y",
                "Face Scale": "lbl_scale",
                "Face Rotation": "lbl_rotation",
                "Brightness": "lbl_brightness",
                "Contrast": "lbl_contrast",
                "Saturation": "lbl_saturation",
                "Gamma": "lbl_gamma",
                "Digital Vibrance": "lbl_vibrance",
                "Texture Preserve": "lbl_texture",
            }.get(label, "_unused_label"), label_widget)
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(int(lo * denom), int(hi * denom))
            s.setValue(int(value * denom))
            s.valueChanged.connect(lambda iv: handler(iv / denom))
            s.setToolTip(_(tip))
            grid.addWidget(s, row, 1, 1, 2)
            return s

        self.s_stabilization = sld("Face Stabilization", 0, 90, getattr(modules.globals, "face_stabilization", 0.0), 1, self._on_stabilization_change, "0 = off; higher values smooth small detector movement", 1)
        self.s_eye = sld("Eye Protection", 0, 100, getattr(modules.globals, "eye_protection", 85.0), 1, self._on_eye_protection_change, "Preserve target eye detail; 0 = off", 2)
        self.s_offset_x = sld("Face X", -100, 100, getattr(modules.globals, "face_offset_x", 0.0), 1, self._on_offset_x_change, "Move the pasted face horizontally", 3)
        self.s_offset_y = sld("Face Y", -100, 100, getattr(modules.globals, "face_offset_y", 0.0), 1, self._on_offset_y_change, "Move the pasted face vertically", 4)
        self.s_scale = sld("Face Scale", 0.60, 1.40, getattr(modules.globals, "face_scale", 1.0), 100, self._on_scale_change, "Adjust pasted face size", 5)
        self.s_rotation = sld("Face Rotation", -30, 30, getattr(modules.globals, "face_rotation", 0.0), 1, self._on_rotation_change, "Rotate the pasted face", 6)

        self.s_brightness = sld("Brightness", -100, 100, getattr(modules.globals, "brightness", 0.0), 1, self._on_brightness_change, "Adjust overall image brightness", 7)
        self.s_contrast = sld("Contrast", 0.50, 1.50, getattr(modules.globals, "contrast", 1.0), 100, self._on_contrast_change, "Adjust overall image contrast", 8)
        self.s_saturation = sld("Saturation", 0.0, 2.0, getattr(modules.globals, "saturation", 1.0), 100, self._on_saturation_change, "Adjust overall image saturation", 9)
        self.s_gamma = sld("Gamma", 0.50, 1.80, getattr(modules.globals, "gamma", 1.0), 100, self._on_gamma_change, "Adjust shadows and highlights", 10)
        self.s_vibrance = sld("Digital Vibrance", -100, 100, getattr(modules.globals, "digital_vibrance", 0.0), 1, self._on_vibrance_change, "Boost muted colours without oversaturating skin", 11)
        self.s_texture = sld("Texture Preserve", 0, 100, getattr(modules.globals, "texture_preservation", 0.0), 1, self._on_texture_change, "Restore face texture after GPEN smoothing", 12)
        self.lbl_target_fps = QLabel(_("Target FPS"))
        grid.addWidget(self.lbl_target_fps, 13, 0)
        self.cb_target_fps = QComboBox()
        self.cb_target_fps.addItems(["15", "20", "24", "25", "30", "60"])
        self.cb_target_fps.setCurrentText(str(getattr(modules.globals, "target_fps", 20)))
        self.cb_target_fps.currentTextChanged.connect(self._on_target_fps_change)
        grid.addWidget(self.cb_target_fps, 13, 1, 1, 2)

        self.sw_diagnostics = _Switch(
            "Диагностика FPS", getattr(modules.globals, "show_diagnostics", False),
            "Показывает время детектора, замены и GPEN рядом с FPS."
        )
        self.sw_diagnostics.toggled.connect(self._on_diagnostics_toggled)
        grid.addWidget(self.sw_diagnostics, 14, 0, 1, 3)

        return card

    # ── action row ───────────────────────────────────────────────────────

    def _build_action_row(self) -> QGridLayout:
        # A two-line grid stays readable in a normal-width window.  The old
        # one-line strip squeezed the last actions out of view.
        row = QGridLayout()
        row.setHorizontalSpacing(5)
        row.setVerticalSpacing(5)
        self.cb_profile = QComboBox()
        self.cb_profile.addItem("Качество", userData="Quality")
        self.cb_profile.addItem("Баланс", userData="Balanced")
        self.cb_profile.addItem("Макс. FPS", userData="Max FPS")
        profile_index = self.cb_profile.findData(getattr(modules.globals, "quality_profile", "Quality"))
        self.cb_profile.setCurrentIndex(max(0, profile_index))
        self.cb_profile.setToolTip("Готовые профили качества и скорости")
        self.cb_profile.currentIndexChanged.connect(self._on_profile_changed)

        self.btn_start = QPushButton(_("Start"))
        self.btn_start.setToolTip(_("Begin processing the target image/video with selected face"))
        self.btn_start.clicked.connect(self._on_start)

        self.btn_destroy = QPushButton(_("Stop video"))
        self.btn_destroy.setObjectName("danger")
        self.btn_destroy.setToolTip(_("Stop live video and release camera"))
        self.btn_destroy.clicked.connect(self._on_stop_live)

        self.btn_preview = QPushButton(_("Preview"))
        self.btn_preview.setObjectName("secondary")
        self.btn_preview.setToolTip(_("Show/hide a preview of the processed output"))
        self.btn_preview.clicked.connect(self._on_toggle_preview)

        self.btn_output = QPushButton(_("Output"))
        self.btn_output.setObjectName("secondary")
        self.btn_output.setToolTip(
            "Открыть чистое окно результата для «Захвата окна» в TikTok LIVE Studio"
        )
        self.btn_output.clicked.connect(self._on_toggle_output_window)

        self.btn_snapshot = QPushButton("📷")
        self.btn_snapshot.setObjectName("secondary")
        self.btn_snapshot.setToolTip("Сохранить текущий кадр с заменённым лицом")
        self.btn_snapshot.clicked.connect(self._on_snapshot)

        self.btn_dataset = QPushButton("Датасет")
        self.btn_dataset.setObjectName("secondary")
        self.btn_dataset.setToolTip(
            "Записать необработанное видео с камеры для обучения новой модели лица"
        )
        self.btn_dataset.clicked.connect(self._on_record_dataset)

        self.btn_compare = QPushButton("A/B")
        self.btn_compare.setObjectName("secondary")
        self.btn_compare.setToolTip("Сравнить оригинальную камеру и обработанный результат; виртуальная камера не меняется")
        self.btn_compare.clicked.connect(self._on_toggle_comparison)

        # Preview is already the large main panel on the right; duplicating
        # that control in this compact action strip only wastes a slot.
        actions_top = (self.cb_profile, self.btn_start, self.btn_destroy)
        actions_bottom = (
            self.btn_output, self.btn_snapshot, self.btn_dataset, self.btn_compare,
        )
        for index, widget in enumerate(actions_top):
            row.addWidget(widget, 0, index)
        for index, widget in enumerate(actions_bottom):
            row.addWidget(widget, 1, index)
        for column in range(4):
            row.setColumnStretch(column, 1)
        for _b in (self.btn_start, self.btn_destroy, self.btn_preview, self.btn_output, self.btn_snapshot, self.btn_dataset, self.btn_compare):
            _b.setFixedHeight(28)
        return row

    # ── camera card ──────────────────────────────────────────────────────

    def _build_camera_card(self) -> QGroupBox:
        card = QGroupBox(_("Camera"))
        card.setObjectName("cameraControlCard")
        self.camera_card = card
        card.setMinimumHeight(80)
        card.setMaximumHeight(88)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.lbl_camera = QLabel(_("Select Camera:"))
        layout.addWidget(self.lbl_camera)
        self._camera_indices, self._camera_names = get_available_cameras()

        self.cb_camera = QComboBox()
        self.cb_camera.setFixedHeight(30)
        if not self._camera_names or self._camera_names[0] == "No cameras found":
            self.cb_camera.addItem("No cameras found")
            self.cb_camera.setEnabled(False)
            cam_ok = False
        else:
            self.cb_camera.addItems(self._camera_names)
            saved_camera = getattr(modules.globals, "last_camera_name", None)
            saved_index = self.cb_camera.findText(str(saved_camera)) if saved_camera else -1
            if saved_index >= 0:
                self.cb_camera.setCurrentIndex(saved_index)
            cam_ok = True
        self.cb_camera.setToolTip(_("Select which camera to use for live mode"))
        self.cb_camera.currentTextChanged.connect(self._on_camera_changed)
        layout.addWidget(self.cb_camera, 1)

        self.btn_live = QPushButton(_("Live"))
        self.btn_live.setFixedHeight(30)
        self.btn_live.setEnabled(cam_ok)
        self.btn_live.setToolTip(_("Start real-time face swap using webcam"))
        self.btn_live.clicked.connect(self._on_live)
        layout.addWidget(self.btn_live)

        return card

    # ── slot handlers ────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _on_select_source(self, video_only: bool = False) -> None:
        global _RECENT_SOURCE_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        file_filter = (
            "Source video (*.mp4 *.mov *.mkv *.avi)"
            if video_only else _IMAGE_FILE_FILTER
        )
        path, _filter = QFileDialog.getOpenFileName(
            self, "Выберите исходное видео" if video_only and _UI_LANGUAGE == "ru" else _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            file_filter,
        )
        if path and is_image(path):
            modules.globals.full_head_mode = False
            # A preceding video source may have supplied a stabilised identity
            # embedding.  Keeping it here makes every subsequently selected
            # photo render as that old person even though its preview changes.
            modules.globals.source_profile_embedding = None
            modules.globals.source_profile_from_video = False
            modules.globals.source_path = path
            modules.globals.last_source_media_path = path
            modules.globals.last_source_cache_path = path
            _RECENT_SOURCE_DIR = os.path.dirname(path)
            self.source_label.setPixmap(render_image_preview(path, (280, 180), crop=False))
            self.source_label.setText("")
            save_switch_states()
        elif path and is_video(path):
            # LivePortrait full-head rendering was too unstable for a moving
            # webcam (warped outline, and only ~7 FPS).  A sharp face picked
            # from the source video uses the proven photo face-swap path: it
            # keeps the user's room/background and remains stable in real time.
            update_status("Selecting the sharpest face frame from the video...")
            selected = _choose_source_video_frame(path)
            if selected is None:
                update_status("A face was not found in the selected video")
                return
            best_frame, best_score, source_profile = selected
            source_cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "source_cache"
            )
            os.makedirs(source_cache_dir, exist_ok=True)
            cached_path = os.path.join(source_cache_dir, "video_source.jpg")
            if not cv2.imwrite(cached_path, best_frame):
                update_status("Could not prepare a source frame from the video")
                return
            modules.globals.full_head_mode = False
            modules.globals.source_path = cached_path
            modules.globals.source_profile_embedding = source_profile
            modules.globals.source_profile_from_video = source_profile is not None
            if source_profile is not None:
                np.save(os.path.join(source_cache_dir, "video_source_embedding.npy"), source_profile)
            modules.globals.last_source_media_path = path
            modules.globals.last_source_cache_path = cached_path
            _RECENT_SOURCE_DIR = os.path.dirname(path)
            self.source_label.setPixmap(render_image_preview(cached_path, (280, 180), crop=False))
            self.source_label.setText("")
            update_status(f"A sharp frontal face frame was selected from the video ({best_score:.0%})")
            save_switch_states()
        elif not path:
            return
        else:
            modules.globals.source_path = None
            self.source_label.clear()
            self.source_label.setText(_("Source face"))

    def _on_toggle_trained_identity(self) -> None:
        """Toggle the locally trained DFM identity without replacing normal mode."""
        from modules.processors.frame.dfm_identity import is_available
        if not is_available():
            update_status("Обученная DFM-модель не найдена в папке models")
            return
        enabled = not bool(getattr(modules.globals, "trained_identity_mode", False))
        modules.globals.trained_identity_mode = enabled
        if enabled:
            # LivePortrait's old experimental full-head path has a different
            # renderer and must not run alongside the DFM decoder.
            modules.globals.full_head_mode = False
            modules.globals.map_faces = False
        self._retranslate_ui()
        save_switch_states()
        update_status(
            "Обученная личность включена: модель загрузится при запуске LIVE"
            if enabled else "Обычная замена по фото включена"
        )

    def _on_toggle_full_head(self) -> None:
        """Enable the separate LivePortrait/TensorRT head renderer for a video source."""
        source_video = getattr(modules.globals, "last_source_media_path", None)
        if not (isinstance(source_video, str) and os.path.isfile(source_video) and is_video(source_video)):
            update_status("Сначала выберите исходное видео кнопкой «Видео»")
            return
        enabled = not bool(getattr(modules.globals, "full_head_mode", False))
        modules.globals.full_head_mode = enabled
        if enabled:
            # A DFM face crop and a full generated head are mutually exclusive.
            modules.globals.trained_identity_mode = False
            modules.globals.map_faces = False
        self._retranslate_ui()
        save_switch_states()
        update_status(
            "Полная голова включена: TensorRT подготовит исходное видео при запуске LIVE"
            if enabled else "Обычная замена по фото включена"
        )

    def _on_select_target(self) -> None:
        global _RECENT_TARGET_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an target image or video"),
            _RECENT_TARGET_DIR or "",
            _MEDIA_FILE_FILTER,
        )
        if not path:
            return
        if is_image(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            self.target_label.setPixmap(render_image_preview(path, (280, 180), crop=False))
            self.target_label.setText("")
            self._update_dashboard_preview(path)
        elif is_video(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            pm = render_video_preview(path, (200, 200))
            if pm:
                self.target_label.setPixmap(pm)
                self.target_label.setText("")
                self._update_dashboard_preview(path)
        else:
            modules.globals.target_path = None
            self.target_label.clear()
            self.target_label.setText(_("Target"))

    def _on_random_face(self) -> None:
        if _PREVIEW is not None:
            _PREVIEW.hide()
        try:
            response = requests.get(
                "https://thispersondoesnotexist.com/",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            response.raise_for_status()
            temp_path = os.path.join(tempfile.gettempdir(), "deep_live_cam_random_face.jpg")
            with open(temp_path, "wb") as f:
                f.write(response.content)
            modules.globals.source_path = temp_path
            modules.globals.source_profile_embedding = None
            modules.globals.source_profile_from_video = False
            self.source_label.setPixmap(render_image_preview(temp_path, (280, 180), crop=False))
            self.source_label.setText("")
        except Exception as exc:
            print(f"Failed to fetch random face: {exc}")

    def _on_swap_paths(self) -> None:
        global _RECENT_SOURCE_DIR, _RECENT_TARGET_DIR
        sp = modules.globals.source_path
        tp = modules.globals.target_path
        if not (sp and tp and is_image(sp) and is_image(tp)):
            return
        modules.globals.source_path, modules.globals.target_path = tp, sp
        # The new source is an ordinary image, not the cached video identity.
        modules.globals.source_profile_embedding = None
        modules.globals.source_profile_from_video = False
        _RECENT_SOURCE_DIR = os.path.dirname(tp)
        _RECENT_TARGET_DIR = os.path.dirname(sp)
        if _PREVIEW is not None:
            _PREVIEW.hide()
        self.source_label.setPixmap(render_image_preview(tp, (280, 180), crop=False))
        self.target_label.setPixmap(render_image_preview(sp, (280, 180), crop=False))
        self._update_dashboard_preview(sp)
        self.source_label.setText("")
        self.target_label.setText("")

    def _update_dashboard_preview(self, path: str) -> None:
        try:
            if is_image(path):
                pm = render_image_preview(path, (900, 520))
            elif is_video(path):
                pm = render_video_preview(path, (900, 520))
            else:
                pm = None
            if pm is not None:
                self.dashboard_preview_label.setPixmap(pm)
                self.dashboard_preview_label.setText("")
        except Exception as exc:
            print(f"Dashboard preview update failed: {exc}")

    def _on_map_faces_toggled(self, value: bool) -> None:
        modules.globals.map_faces = value
        save_switch_states()
        if not value:
            close_mapper_window()

    def _on_enhancer_change(self, choice: str) -> None:
        key_map = {
            "None": None,
            "GFPGAN": "face_enhancer",
            "GPEN-512": "face_enhancer_gpen512",
            "GPEN-256": "face_enhancer_gpen256",
            "GPEN-1024 (slow)": "face_enhancer_gpen1024",
        }
        for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512", "face_enhancer_gpen1024"):
            _update_tumbler(key, False)
        selected = key_map.get(choice)
        if selected:
            _update_tumbler(selected, True)
        save_switch_states()

    def _on_transparency_change(self, value: float) -> None:
        modules.globals.opacity = value
        modules.globals.face_swapper_enabled = value > 0.0
        save_switch_states()
        pct = int(value * 100)
        if pct == 0:
            modules.globals.fp_ui["face_enhancer"] = False
            update_status("Transparency set to 0% - Face swapping disabled.")
        elif pct == 100:
            modules.globals.face_swapper_enabled = True
            update_status("Transparency set to 100%.")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"Transparency set to {pct}%")

    def _on_sharpness_change(self, value: float) -> None:
        modules.globals.sharpness = value
        save_switch_states()
        update_status(f"Sharpness set to {value:.1f}")

    def _on_mouth_mask_change(self, value: float) -> None:
        modules.globals.mouth_mask_size = value
        modules.globals.mouth_mask = value > 0
        if value <= 0:
            modules.globals.show_mouth_mask_box = False
        save_switch_states()

    def _on_mask_feather_change(self, value: float) -> None:
        modules.globals.mask_feather = max(0.0, min(100.0, float(value)))
        save_switch_states()
        update_status(f"Edge Softness set to {value:.0f}")

    def _on_mask_profile_change(self, _index: int) -> None:
        profile = str(self.cb_mask_profile.currentData() or "Chin")
        modules.globals.mask_profile = profile
        save_switch_states()
        update_status("Форма маски изменена — применяется на следующем кадре")

    def _on_mouth_mask_pressed(self) -> None:
        if modules.globals.mouth_mask_size > 0:
            modules.globals.show_mouth_mask_box = True

    def _on_mouth_mask_released(self) -> None:
        modules.globals.show_mouth_mask_box = False

    def _on_color_match_toggled(self, value: bool) -> None:
        modules.globals.color_match = bool(value)
        save_switch_states()
        update_status("Color Match " + ("enabled" if value else "disabled"))

    def _on_stabilization_change(self, value: float) -> None:
        modules.globals.face_stabilization = max(0.0, min(90.0, float(value)))
        save_switch_states()

    def _on_eye_protection_change(self, value: float) -> None:
        modules.globals.eye_protection = max(0.0, min(100.0, float(value)))
        save_switch_states()

    def _on_offset_x_change(self, value: float) -> None:
        modules.globals.face_offset_x = max(-100.0, min(100.0, float(value)))
        save_switch_states()

    def _on_offset_y_change(self, value: float) -> None:
        modules.globals.face_offset_y = max(-100.0, min(100.0, float(value)))
        save_switch_states()

    def _on_scale_change(self, value: float) -> None:
        modules.globals.face_scale = max(0.60, min(1.40, float(value)))
        save_switch_states()

    def _on_rotation_change(self, value: float) -> None:
        modules.globals.face_rotation = max(-30.0, min(30.0, float(value)))
        save_switch_states()

    def _on_brightness_change(self, value: float) -> None:
        modules.globals.brightness = max(-100.0, min(100.0, float(value)))
        save_switch_states()

    def _on_contrast_change(self, value: float) -> None:
        modules.globals.contrast = max(0.50, min(1.50, float(value)))
        save_switch_states()

    def _on_saturation_change(self, value: float) -> None:
        modules.globals.saturation = max(0.0, min(2.0, float(value)))
        save_switch_states()

    def _on_gamma_change(self, value: float) -> None:
        modules.globals.gamma = max(0.50, min(1.80, float(value)))
        save_switch_states()

    def _on_vibrance_change(self, value: float) -> None:
        modules.globals.digital_vibrance = max(-100.0, min(100.0, float(value)))
        save_switch_states()

    def _on_texture_change(self, value: float) -> None:
        modules.globals.texture_preservation = max(0.0, min(100.0, float(value)))
        save_switch_states()

    def _on_diagnostics_toggled(self, value: bool) -> None:
        modules.globals.show_diagnostics = bool(value)
        save_switch_states()

    def _on_toggle_comparison(self) -> None:
        if _WEBCAM_PREVIEW is None:
            update_status("Start LIVE before using A/B comparison")
            return
        modules.globals.preview_original = not bool(
            getattr(modules.globals, "preview_original", False)
        )
        self.btn_compare.setText("После" if modules.globals.preview_original else "A/B")
        update_status("A/B: original camera" if modules.globals.preview_original else "A/B: processed result")

    def _on_performance_mode_display_change(self, value: str) -> None:
        key = self.cb_performance.currentData()
        self._on_performance_mode_change(str(key or value))

    def _on_performance_mode_change(self, value: str) -> None:
        modules.globals.performance_mode = value if value in ("Quality", "Balanced", "Performance") else "Balanced"
        if modules.globals.performance_mode == "Performance":
            # Restore the proven LIVE preset: the light GPEN-256 pass is part
            # of the 22–24 FPS reference output.  Larger GFPGAN/GPEN models
            # remain off in this mode, but selecting Performance must not
            # silently turn the visible GPEN-256 selection into "None".
            modules.globals.fp_ui["face_enhancer"] = False
            modules.globals.fp_ui["face_enhancer_gpen512"] = False
            modules.globals.fp_ui["face_enhancer_gpen1024"] = False
            if not modules.globals.fp_ui.get("face_enhancer_gpen256", False):
                modules.globals.fp_ui["face_enhancer_gpen256"] = True
            # Keep GPEN-256 in the same ONNX Runtime CUDA context as the
            # detector and swapper.  The optional Torch/TensorRT engine uses
            # a separate CUDA stream; on this setup it produced long global
            # synchronisation stalls even though its average time was low.
            modules.globals.use_gpen256_tensorrt = False
            if hasattr(self, "cb_enhancer"):
                self.cb_enhancer.blockSignals(True)
                self.cb_enhancer.setCurrentText("GPEN-256")
                self.cb_enhancer.blockSignals(False)
        save_switch_states()
        suffix = " — GPEN-256 включён, тяжёлые улучшатели отключены" if modules.globals.performance_mode == "Performance" else ""
        update_status(f"Performance mode: {_(modules.globals.performance_mode)}{suffix}")

    def _on_target_fps_change(self, value: str) -> None:
        try:
            modules.globals.target_fps = int(value)
        except ValueError:
            modules.globals.target_fps = 20
        save_switch_states()

    def _on_quality_change(self, value: str) -> None:
        if value in QUALITY_OPTIONS:
            modules.globals.preview_quality = value
            save_switch_states()
            update_status(f"{_('Processing quality:')} {value}. Изменено качество Face Swap, не разрешение камеры.")

    def _on_resolution_change(self, value: str) -> None:
        if value in QUALITY_OPTIONS:
            modules.globals.camera_resolution = value
            save_switch_states()
            width, height = QUALITY_OPTIONS[value]
            update_status(f"{_('Camera resolution:')} {width} × {height}. Перезапустите LIVE для применения.")

    def _on_camera_changed(self, value: str) -> None:
        if value and value != "No cameras found":
            modules.globals.last_camera_name = value
            save_switch_states()

    def _on_profile_changed(self, _index: int) -> None:
        profile = str(self.cb_profile.currentData() or "Quality")
        profiles = {
            "Quality": {
                "performance_mode": "Quality", "enhancer": "GPEN-256", "preview_quality": "1080p",
                "gpen256_tensorrt": False,
                "target_fps": 25, "sharpness": .8, "mouth_mask_size": 100.,
                "mask_feather": 18., "face_stabilization": 60., "poisson_blend": False,
            },
            "Balanced": {
                "performance_mode": "Balanced", "enhancer": "GPEN-256", "preview_quality": "720p",
                "gpen256_tensorrt": False,
                "target_fps": 25, "sharpness": .6, "mouth_mask_size": 80.,
                "mask_feather": 18., "face_stabilization": 45., "poisson_blend": False,
            },
            "Max FPS": {
                # This is the verified LIVE reference preset: it retains the
                # lightweight GPEN-256 finish (about 18 ms) while avoiding the
                # much heavier 1080p and larger-enhancer paths.
                "performance_mode": "Performance", "enhancer": "GPEN-256", "preview_quality": "720p",
                "gpen256_tensorrt": False,
                "target_fps": 30, "sharpness": .2, "mouth_mask_size": 0.,
                "mask_feather": 42., "face_stabilization": 25., "poisson_blend": False,
                "color_match": True,
            },
        }
        values = profiles[profile]
        modules.globals.quality_profile = profile
        modules.globals.performance_mode = values["performance_mode"]
        modules.globals.preview_quality = values["preview_quality"]
        modules.globals.target_fps = values["target_fps"]
        modules.globals.sharpness = values["sharpness"]
        modules.globals.mouth_mask_size = values["mouth_mask_size"]
        modules.globals.mouth_mask = values["mouth_mask_size"] > 0
        modules.globals.mask_feather = values["mask_feather"]
        modules.globals.face_stabilization = values["face_stabilization"]
        modules.globals.poisson_blend = values["poisson_blend"]
        modules.globals.full_head_coverage = True
        modules.globals.color_match = bool(values.get("color_match", True))
        modules.globals.use_gpen256_tensorrt = bool(values.get("gpen256_tensorrt", False))
        enhancer_map = {
            "None": None,
            "GFPGAN": "face_enhancer",
            "GPEN-256": "face_enhancer_gpen256",
            "GPEN-512": "face_enhancer_gpen512",
            "GPEN-1024 (slow)": "face_enhancer_gpen1024",
        }
        for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512", "face_enhancer_gpen1024"):
            modules.globals.fp_ui[key] = key == enhancer_map[values["enhancer"]]
        for widget, value in (
            (self.cb_performance, values["performance_mode"]), (self.cb_target_fps, str(values["target_fps"])),
            (self.cb_quality, values["preview_quality"]), (self.cb_enhancer, values["enhancer"]),
            (self.s_sharpness, int(values["sharpness"] * 10)), (self.s_mouth, int(values["mouth_mask_size"])),
            (self.s_mask_feather, int(values["mask_feather"])), (self.s_stabilization, int(values["face_stabilization"])),
        ):
            widget.blockSignals(True)
            if isinstance(widget, QSlider):
                widget.setValue(value)
            elif widget is self.cb_quality:
                index = widget.findData(value)
                widget.setCurrentIndex(index if index >= 0 else widget.findData("720p"))
            else:
                widget.setCurrentText(value)
            widget.blockSignals(False)
        self.sw_poisson._checkbox.blockSignals(True)
        self.sw_poisson._checkbox.setChecked(values["poisson_blend"])
        self.sw_poisson._checkbox.blockSignals(False)
        get_frame_processors_modules(modules.globals.frame_processors)
        save_switch_states()
        update_status(f"Profile applied: {profile}")

    def _on_snapshot(self) -> None:
        if _WEBCAM_PREVIEW is None:
            update_status("Start LIVE before taking a snapshot")
            return
        frame = _WEBCAM_PREVIEW.get_last_processed_frame()
        if frame is None:
            update_status("Waiting for the first processed frame")
            return
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "captures")
        try:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"face-swap-{time.strftime('%Y%m%d-%H%M%S')}.png")
            if not cv2.imwrite(path, frame):
                raise OSError("image writer failed")
            update_status(f"Snapshot saved: {path}")
        except OSError as exc:
            update_status(f"Could not save snapshot: {exc}")

    def _on_record_dataset(self) -> None:
        """Open an isolated raw-camera recorder for DFM training material.

        It never sends frames to the virtual camera or through the face swap
        pipeline.  Keeping recording separate makes the collected footage a
        trustworthy target dataset and avoids competing for one camera handle.
        """
        global _DATASET_RECORDER
        if _WEBCAM_PREVIEW is not None:
            update_status("Сначала нажми «Стоп видео»: камера занята LIVE-режимом")
            return
        idx = self.cb_camera.currentIndex()
        if idx < 0 or idx >= len(self._camera_indices):
            update_status("Камера недоступна")
            return
        if _DATASET_RECORDER is not None and _DATASET_RECORDER.isVisible():
            _DATASET_RECORDER.raise_()
            _DATASET_RECORDER.activateWindow()
            return
        _DATASET_RECORDER = DatasetRecorderDialog(
            self._camera_indices[idx],
            getattr(modules.globals, "camera_resolution", "720p"),
            parent=self,
        )
        _DATASET_RECORDER.show()

    def _on_language_change(self, value: str) -> None:
        global _UI_LANGUAGE
        _UI_LANGUAGE = "ru" if value in ("Russian", "Русский") else "en"
        save_switch_states()
        self._retranslate_ui()

    def _on_start(self) -> None:
        if _MAPPER is not None and _MAPPER.isVisible():
            update_status("Please complete pop-up or close it.")
            return
        if modules.globals.map_faces:
            modules.globals.source_target_map = []
            if is_image(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_image()
            elif is_video(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_video()
            if modules.globals.source_target_map:
                _open_mapper_dialog(self._start_cb, modules.globals.source_target_map)
            else:
                update_status("No faces found in target")
        else:
            self._select_output_and_start()

    def _select_output_and_start(self) -> None:
        global _RECENT_OUTPUT_DIR
        if is_image(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save image output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.png"),
                _IMAGE_FILE_FILTER,
            )
        elif is_video(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save video output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.mp4"),
                _VIDEO_FILE_FILTER,
            )
        else:
            return
        if path:
            modules.globals.output_path = path
            _RECENT_OUTPUT_DIR = os.path.dirname(path)
            self._start_cb()

    def _on_toggle_preview(self) -> None:
        if _PREVIEW is None:
            return
        if _PREVIEW.isVisible():
            _PREVIEW.hide()
        elif modules.globals.source_path and modules.globals.target_path:
            _PREVIEW.init_for_target()
            _PREVIEW.refresh_frame(0)
            _PREVIEW.show()

    def _on_select_virtual_background(self) -> None:
        """Choose a still image used behind the live camera subject."""
        initial = os.path.dirname(
            str(getattr(modules.globals, "virtual_background_path", "") or "")
        )
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Выберите фон" if _UI_LANGUAGE == "ru" else "Choose background",
            initial,
            _IMAGE_FILE_FILTER,
        )
        if not path or not is_image(path):
            return
        modules.globals.virtual_background_path = path
        modules.globals.virtual_background = True
        self.sw_virtual_background.setChecked(True)
        save_switch_states()
        update_status("Виртуальный фон выбран. Он применится к следующему кадру LIVE.")

    def _on_toggle_output_window(self) -> None:
        """Show a clean top-level result window for screen/window capture.

        This does not create a camera device and does not change the virtual
        camera path.  The window is updated only from the existing GUI timer,
        so it adds no inference, detection, or copying work to the live worker.
        """
        global _OUTPUT_WINDOW, _OUTPUT_WINDOW_ACTIVE
        if _OUTPUT_WINDOW is not None and _OUTPUT_WINDOW.isVisible():
            _OUTPUT_WINDOW.close()
            return
        _OUTPUT_WINDOW = OutputWindow()
        _OUTPUT_WINDOW_ACTIVE = True
        _OUTPUT_WINDOW.show()
        _OUTPUT_WINDOW.raise_()
        _OUTPUT_WINDOW.activateWindow()
        self.btn_output.setText(_("Close"))
        if _WEBCAM_PREVIEW is None:
            update_status("Окно вывода готово. Нажми LIVE — в нём появится обработанное изображение.")
        else:
            update_status("Окно вывода открыто. В TikTok выбери «Захват окна» → DLC Output.")

    def _on_stop_live(self) -> None:
        """Stop only the webcam workers; keep the main application open."""
        global _WEBCAM_PREVIEW
        if _WEBCAM_PREVIEW is None:
            update_status("Live video is not running")
            return
        try:
            _WEBCAM_PREVIEW.close()
            update_status("Live video stopped. Camera released.")
        except Exception as exc:
            update_status(f"Could not stop live video: {exc}")
        finally:
            _WEBCAM_PREVIEW = None

    def _warm_live_pipeline(self) -> None:
        """Complete GPU first-use work before the live preview is visible.

        ONNX Runtime allocates CUDA buffers and selects kernels on the first
        real detector frame.  Performing that work after the preview opens
        produces a conspicuous 2–6 FPS burst even though steady state is fast.
        """
        from modules.face_analyser import get_face_analyser

        update_status("Подготовка моделей GPU…")
        get_face_analyser()
        quality = getattr(modules.globals, "preview_quality", "720p")
        capture_quality = getattr(modules.globals, "camera_resolution", "720p")
        capture_width, capture_height = QUALITY_OPTIONS.get(
            capture_quality, QUALITY_OPTIONS["720p"]
        )
        # Match WebcamPreviewWindow's negotiated capture ceiling and the
        # processing scale exactly, so the first visible inference needs no
        # new CUDA allocations.
        capture_width, capture_height = min(capture_width, 1280), min(capture_height, 720)
        scale = _quality_process_scale(capture_width, capture_height, quality)
        warm_width = max(64, int(round(capture_width * scale)))
        warm_height = max(64, int(round(capture_height * scale)))
        warm_frame = np.zeros((warm_height, warm_width, 3), dtype=np.uint8)
        # A detection pass allocates the detector's actual GPU buffers.  A
        # blank frame intentionally has no face, so it cannot affect mapping
        # or source/target state.
        detect_one_face_fast(warm_frame)

        if modules.globals.fp_ui.get("face_enhancer_gpen256", False):
            from modules.processors.frame.face_enhancer_gpen256 import get_enhancer
            get_enhancer()
        elif modules.globals.fp_ui.get("face_enhancer_gpen512", False):
            from modules.processors.frame.face_enhancer_gpen512 import get_enhancer
            get_enhancer()
        elif modules.globals.fp_ui.get("face_enhancer", False):
            from modules.processors.frame.face_enhancer import get_face_enhancer
            get_face_enhancer()

    def _on_live(self) -> None:
        idx = self.cb_camera.currentIndex()
        if idx < 0 or idx >= len(self._camera_indices):
            update_status("No camera available")
            return
        camera_index = self._camera_indices[idx]
        if _LIVE_MAPPER is not None and _LIVE_MAPPER.isVisible():
            update_status("Source x Target Mapper is already open.")
            _LIVE_MAPPER.raise_()
            return
        if not modules.globals.map_faces:
            if modules.globals.source_path is None:
                update_status("Please select a source image first")
                return
            from modules.face_analyser import get_face_analyser
            get_face_analyser()
            if not getattr(modules.globals, "full_head_mode", False):
                from modules.processors.frame.face_swapper import get_face_swapper
                get_face_swapper()
            self._warm_live_pipeline()
            _open_webcam_preview(
                camera_index,
                getattr(modules.globals, "camera_resolution", "720p"),
                host=self.dashboard_preview,
                host_layout=self.dashboard_preview_layout,
                placeholder=self.dashboard_preview_label,
            )
        else:
            modules.globals.source_target_map = []
            _open_live_mapper_dialog(camera_index, modules.globals.source_target_map)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep the left controls readable in a normal window, while allowing the
        # right dashboard/preview to expand freely when the window is maximized.
        if hasattr(self, "_splitter"):
            h = max(620, self.height())
            left = 470 if h < 760 else 500
            if not self._splitter.isVisible():
                return
            sizes = self._splitter.sizes()
            if len(sizes) == 2:
                if abs(sizes[0] - left) > 12:
                    self._splitter.setSizes([left, max(240, self._splitter.width() - left)])

    def closeEvent(self, event):
        global _WEBCAM_PREVIEW
        if _WEBCAM_PREVIEW is not None:
            try:
                _WEBCAM_PREVIEW.close()
            except Exception:
                pass
            _WEBCAM_PREVIEW = None
        # Treat OS-level close as Destroy click
        self._destroy_cb()
        event.accept()


def _update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, refresh frame processors so
    # toggling enhancers takes effect immediately.
    if _WEBCAM_PREVIEW is not None and _WEBCAM_PREVIEW.isVisible():
        get_frame_processors_modules(modules.globals.frame_processors)


# ─── preview window (still-image / video scrub) ──────────────────────────


class DatasetRecorderDialog(QDialog):
    """Record raw, consented camera footage for an isolated DFM dataset.

    It has no face-swap, enhancer, virtual-camera or network path. The user
    explicitly starts and stops recording; closing always releases the camera.
    """

    def __init__(self, camera_index: int, quality: str = "720p", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Запись датасета лица")
        self.resize(820, 600)
        self.setModal(False)
        self._writer = None
        self._output_path: Optional[str] = None
        self._recording_started_at: Optional[float] = None
        self._recorded_frames = 0

        layout = QVBoxLayout(self)
        self._image_label = QLabel("Подключение к камере…")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(640, 360)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)
        self._status = QLabel("Необработанная камера. Запись не попадёт в OBS и не пройдёт через замену лица.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        controls = QHBoxLayout()
        self._record_button = QPushButton("Начать запись")
        self._record_button.setObjectName("primary")
        self._record_button.clicked.connect(self._toggle_recording)
        controls.addWidget(self._record_button)
        self._close_button = QPushButton("Закрыть")
        self._close_button.setObjectName("secondary")
        self._close_button.clicked.connect(self.close)
        controls.addWidget(self._close_button)
        layout.addLayout(controls)

        self._cap = VideoCapturer(camera_index)
        req_w, req_h = QUALITY_OPTIONS.get(quality, QUALITY_OPTIONS["720p"])
        req_w, req_h = min(req_w, 1280), min(req_h, 720)
        if not self._cap.start(req_w, req_h, 30):
            self._status.setText("Не удалось открыть камеру. Закрой LIVE/OBS и попробуй ещё раз.")
            self._record_button.setEnabled(False)
            QTimer.singleShot(0, self.close)
            return
        fps = max(1.0, float(self._cap.actual_fps or 30.0))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(max(1, min(33, int(1000.0 / fps))))
        self._status.setText(
            "Готово. Нажми «Начать запись», затем 5–10 минут медленно поворачивай голову "
            "влево/вправо, вверх/вниз и говори. Видео сохранится только на этом ПК."
        )

    def _tick(self) -> None:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._status.setText("Камера перестала передавать кадры")
            self._stop_recording()
            return
        if self._writer is not None:
            self._writer.write(frame)
            self._recorded_frames += 1
            elapsed = max(0.0, time.time() - (self._recording_started_at or time.time()))
            self._status.setText(
                f"Идёт запись: {elapsed:05.1f} сек • кадров: {self._recorded_frames}. "
                "Нажми «Остановить и сохранить», когда закончишь."
            )
        preview = fit_image_to_size(frame, self._image_label.width(), self._image_label.height())
        self._image_label.setPixmap(_bgr_to_qpixmap(preview))

    def _toggle_recording(self) -> None:
        if self._writer is None:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "dataset_recordings")
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"target-{time.strftime('%Y%m%d-%H%M%S')}.mp4")
            width, height = self._cap.actual_width, self._cap.actual_height
            fps = max(10.0, min(60.0, float(self._cap.actual_fps or 30.0)))
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise OSError("MP4 video writer is unavailable")
            self._writer = writer
            self._output_path = output_path
            self._recording_started_at = time.time()
            self._recorded_frames = 0
            self._record_button.setText("Остановить и сохранить")
        except OSError as exc:
            self._status.setText(f"Не удалось начать запись: {exc}")

    def _stop_recording(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.release()
        finally:
            self._writer = None
        duration = self._recorded_frames / max(1.0, float(self._cap.actual_fps or 30.0))
        saved_path = self._output_path or ""
        self._record_button.setText("Начать новую запись")
        self._status.setText(
            f"Сохранено {duration:.1f} сек: {saved_path}. "
            "Если записал 5–10 минут, закрой окно — я подготовлю кадры."
        )
        update_status(f"Dataset recording saved: {saved_path}")

    def closeEvent(self, event) -> None:
        self._stop_recording()
        try:
            self._timer.stop()
        except AttributeError:
            pass
        try:
            self._cap.release()
        except Exception:
            pass
        global _DATASET_RECORDER
        if _DATASET_RECORDER is self:
            _DATASET_RECORDER = None
        event.accept()


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("Preview"))
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self.refresh_frame)
        layout.addWidget(self._slider)

    def init_for_target(self) -> None:
        if is_image(modules.globals.target_path):
            self._slider.hide()
        elif is_video(modules.globals.target_path):
            total = get_video_frame_total(modules.globals.target_path)
            self._slider.setRange(0, max(0, total - 1))
            self._slider.setValue(0)
            self._slider.show()

    def refresh_frame(self, frame_number: int = 0) -> None:
        if not (modules.globals.source_path and modules.globals.target_path):
            return
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        from modules.processors.frame.core import get_frame_processors_modules as _gfpm
        for fp in _gfpm(modules.globals.frame_processors):
            temp_frame = fp.process_frame(
                get_one_face(imread_unicode(modules.globals.source_path)), temp_frame
            )
        # Fit to current widget size while preserving aspect ratio.
        h, w = temp_frame.shape[:2]
        bound_w = min(PREVIEW_MAX_WIDTH, max(self.width(), PREVIEW_DEFAULT_WIDTH))
        bound_h = min(PREVIEW_MAX_HEIGHT, max(self.height(), PREVIEW_DEFAULT_HEIGHT))
        ratio = min(bound_w / w, bound_h / h)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        temp_frame = cv2.resize(temp_frame, new_size, interpolation=cv2.INTER_LANCZOS4)
        self._image_label.setPixmap(_bgr_to_qpixmap(temp_frame))
        update_status("Processing succeed!")


# ─── webcam preview window ───────────────────────────────────────────────


class _CaptureWorker(QThread):
    """Reads frames from the camera into a bounded queue. Drops on overflow."""

    def __init__(self, cap, capture_queue: queue.Queue, stop_event: threading.Event):
        super().__init__()
        self._cap = cap
        self._queue = capture_queue
        self._stop = stop_event

    def run(self) -> None:
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._stop.set()
                break
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass


class _ProcessingWorker(QThread):
    """Pulls raw frames, runs detect/swap/enhance, pushes processed frames."""

    metricsChanged = Signal(float, float, float, float, float)

    def __init__(self, capture_queue, processed_queue, stop_event, camera_fps: float):
        super().__init__()
        self._cq = capture_queue
        self._pq = processed_queue
        self._stop = stop_event
        self._fps = camera_fps
        self._virtual_camera_output: Optional[VirtualCameraPublisher] = None
        self._virtual_camera_failed = False
        self._virtual_camera_announced = False
        self._virtual_background = VirtualBackground()
        self._virtual_background_error_announced = False
        self._smart_fps_low_windows = 0
        self._smart_fps_high_windows = 0

    def _apply_virtual_background(self, frame: np.ndarray) -> np.ndarray:
        """Apply the user-selected portrait matte before any live output."""
        if not getattr(modules.globals, "virtual_background", False):
            self._virtual_background_error_announced = False
            return frame
        background_path = getattr(modules.globals, "virtual_background_path", None)
        if not isinstance(background_path, str) or not os.path.isfile(background_path):
            if not self._virtual_background_error_announced:
                update_status("Виртуальный фон не найден. Выберите изображение в настройках.")
                self._virtual_background_error_announced = True
            return frame
        result = self._virtual_background.apply(frame, background_path)
        if self._virtual_background.last_error and not self._virtual_background_error_announced:
            update_status(f"Виртуальный фон недоступен: {self._virtual_background.last_error}")
            self._virtual_background_error_announced = True
        return result

    def _close_virtual_camera(self) -> None:
        if self._virtual_camera_output is not None:
            try:
                self._virtual_camera_output.close()
            except Exception:
                pass
            self._virtual_camera_output = None

    def _publish_virtual_camera(self, bgr_frame: np.ndarray, target_fps: int) -> None:
        """Queue the newest output frame without blocking live processing."""
        if not getattr(modules.globals, "virtual_camera", False):
            self._close_virtual_camera()
            self._virtual_camera_failed = False
            self._virtual_camera_announced = False
            return
        if self._virtual_camera_failed:
            return
        try:
            if self._virtual_camera_output is None:
                self._virtual_camera_output = VirtualCameraPublisher()
            self._virtual_camera_output.submit(bgr_frame, target_fps)
            device_name = self._virtual_camera_output.device_name
            if device_name and not self._virtual_camera_announced:
                update_status(f"Virtual camera ready: {device_name}")
                self._virtual_camera_announced = True
        except Exception as exc:
            self._close_virtual_camera()
            self._virtual_camera_failed = True
            update_status(
                "Virtual camera unavailable. Install OBS Virtual Camera, then restart. "
                f"({exc})"
            )

    def run(self) -> None:
        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        source_image = None
        last_source_key = None
        last_full_head_mode = None
        full_head_processor = None
        prev_time = time.time()
        fps_update_interval = 0.5
        frame_count = 0
        fps = 0.0
        process_count = 0
        det_count = 0
        cached_target_face = None
        cached_many_faces = None
        gpen256_interval = 1

        while not self._stop.is_set():
            try:
                frame = self._cq.get(timeout=0.05)
            except queue.Empty:
                continue

            t_total = time.perf_counter()
            t_detection = 0.0
            t_swap = 0.0
            t_gpen = 0.0
            t_other = 0.0

            temp_frame = frame
            if modules.globals.live_mirror:
                t0 = time.perf_counter()
                temp_frame = gpu_flip(temp_frame, 1)
                t_other += time.perf_counter() - t0

            # Enforce the selected Quality at the actual processing stage. This
            # is necessary for webcams such as Iriun that accept 1280x960 even
            # when a different requested capture mode is ignored by the driver.
            original_live_h, original_live_w = temp_frame.shape[:2]
            quality_name = getattr(modules.globals, "preview_quality", "720p")
            perf_mode = getattr(modules.globals, "performance_mode", "Balanced")
            # Always enhance the current face. Reusing a previous enhanced
            # result or skipping the pass visibly degrades facial texture.
            gpen256_interval = 1
            # Performance mode uses a smaller processing frame to target the 24 FPS
            # camera rate. Quality/Balanced keep the selected processing quality.
            if perf_mode == "Performance":
                q_scale = min(0.50, _quality_process_scale(original_live_w, original_live_h, "360p"))
            else:
                q_scale = _quality_process_scale(original_live_w, original_live_h, quality_name)
            if q_scale < 0.999:
                t0 = time.perf_counter()
                proc_w = max(64, int(round(original_live_w * q_scale)))
                proc_h = max(64, int(round(original_live_h * q_scale)))
                temp_frame = cv2.resize(temp_frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                t_other += time.perf_counter() - t0

            # A/B is preview-only: copying the unprocessed frame is done only
            # while the user is actually comparing. The virtual camera always
            # receives ``temp_frame`` after the complete processing pipeline.
            preview_original = (
                temp_frame.copy()
                if getattr(modules.globals, "preview_original", False)
                else None
            )

            # Filled after Face Swap, before GPEN.  This lets us preserve the
            # swapped model's current mouth movement rather than putting the
            # original camera teeth/braces back over the generated face.
            mouth_source_frame = None

            if not modules.globals.map_faces:
                full_head_mode = bool(getattr(modules.globals, "full_head_mode", False))
                source_key = (
                    getattr(modules.globals, "last_source_media_path", None)
                    if full_head_mode else modules.globals.source_path
                )
                if source_key and (
                    source_key != last_source_key
                    or full_head_mode != last_full_head_mode
                ):
                    last_source_key = source_key
                    last_full_head_mode = full_head_mode
                    if full_head_mode:
                        try:
                            from modules.processors.frame.full_head_trt import FullHeadTRT
                            full_head_processor = FullHeadTRT(source_key)
                            source_image = None
                            update_status("Full-head GPU source prepared")
                        except Exception as exc:
                            full_head_processor = None
                            update_status(f"Full-head GPU unavailable: {exc}")
                    else:
                        full_head_processor = None
                        source_image = get_one_face(imread_unicode(modules.globals.source_path))
                        profile = getattr(modules.globals, "source_profile_embedding", None)
                        if source_image is not None and profile is not None:
                            try:
                                profile = np.asarray(profile, dtype=np.float32).reshape(-1)
                                if profile.size == source_image.normed_embedding.size and np.all(np.isfinite(profile)):
                                    # InsightFace derives normed_embedding from this
                                    # field, so the regular fast swapper receives a
                                    # stable multi-frame identity profile.
                                    source_image.embedding = profile
                            except (AttributeError, ValueError):
                                pass

                det_count += 1
                # Use the detector's current landmarks on every frame. This
                # keeps the swapped face locked to real head motion and avoids
                # the visible mask jitter caused by extrapolating keypoints.
                t0 = time.perf_counter()
                raw_faces = []
                if modules.globals.many_faces:
                    detected_many = detect_many_faces_fast(temp_frame)
                    if detected_many:
                        raw_faces = list(detected_many)
                    cached_target_face = None
                else:
                    detected_one = detect_one_face_fast(temp_frame)
                    raw_faces = [detected_one] if detected_one is not None else []
                    cached_many_faces = None

                # Fast detection skips the 2d106 landmark model, but the mouth
                # mask / eye protection need it. Attach landmarks before applying
                # geometry controls so the prepared copy retains the originals.
                use_adaptive_mask = (
                    getattr(modules.globals, "adaptive_mask", False)
                    and getattr(modules.globals, "performance_mode", "Balanced") != "Performance"
                )
                if (modules.globals.mouth_mask or use_adaptive_mask) and raw_faces:
                    ensure_landmarks(temp_frame, raw_faces)

                prepared_faces = []
                for face_index, raw_face in enumerate(raw_faces):
                    if raw_face is not None:
                        prepared_faces.append(
                            prepare_target_face(raw_face, temp_frame.shape, track_key=face_index)
                        )

                if modules.globals.many_faces:
                    cached_many_faces = prepared_faces
                    cached_target_face = None
                else:
                    cached_target_face = prepared_faces[0] if prepared_faces else None
                    cached_many_faces = None

                t_detection += time.perf_counter() - t0

                cached_faces = None
                if cached_many_faces:
                    cached_faces = cached_many_faces
                elif cached_target_face is not None:
                    cached_faces = [cached_target_face]

                full_head_applied = False
                if full_head_processor is not None and cached_target_face is not None:
                    try:
                        t0 = time.perf_counter()
                        temp_frame = full_head_processor.process(temp_frame, cached_target_face)
                        t_swap += time.perf_counter() - t0
                        full_head_applied = True
                    except Exception as exc:
                        full_head_processor = None
                        update_status(f"Full-head frame fallback: {exc}")

                # LIVE PIPELINE ORDER:
                # 1) Face swap first.
                # 2) Other non-enhancer processors.
                # 3) GPEN/GFPGAN last, so the enhancer sees the already-swapped face.
                #
                # The previous order could run GPEN before the swapper depending on
                # frame_processors order. GPEN changes pixels but the cached target
                # Face geometry remains from the original frame, which can produce
                # a displaced rectangular face overlay.
                for fp in frame_processors:
                    if full_head_applied:
                        continue
                    if fp.NAME == "DLC.FACE-SWAPPER":
                        swapped_bboxes = []
                        if modules.globals.many_faces and cached_many_faces:
                            result = temp_frame.copy()
                            for t_face in cached_many_faces:
                                t0 = time.perf_counter()
                                result = fp.swap_face(source_image, t_face, result)
                                t_swap += time.perf_counter() - t0
                                if hasattr(t_face, "bbox") and t_face.bbox is not None:
                                    swapped_bboxes.append(t_face.bbox.astype(int))
                            temp_frame = result
                        elif cached_target_face is not None:
                            t0 = time.perf_counter()
                            temp_frame = fp.swap_face(
                                source_image, cached_target_face, temp_frame
                            )
                            t_swap += time.perf_counter() - t0
                            if (
                                hasattr(cached_target_face, "bbox")
                                and cached_target_face.bbox is not None
                            ):
                                swapped_bboxes.append(cached_target_face.bbox.astype(int))
                        t0 = time.perf_counter()
                        temp_frame = fp.apply_post_processing(temp_frame, swapped_bboxes)
                        t_swap += time.perf_counter() - t0

                    elif fp.NAME not in (
                        "DLC.FACE-ENHANCER",
                        "DLC.FACE-ENHANCER-GPEN256",
                        "DLC.FACE-ENHANCER-GPEN512",
                    ):
                        t0 = time.perf_counter()
                        temp_frame = fp.process_frame(source_image, temp_frame)
                        t_other += time.perf_counter() - t0

                # Preserve the already-swapped current mouth before GPEN. The
                # copy is made only when the Mouth Mask control is active.
                # It is a frame-local result, never a previous video frame.
                if not full_head_applied and (
                    getattr(modules.globals, "mouth_mask", False)
                    or float(getattr(modules.globals, "mouth_mask_size", 0.0)) > 0.0
                ):
                    mouth_source_frame = temp_frame.copy()

                # Enhancers intentionally run AFTER the swap.
                # Keep the cached target face: swap does not change its geometry.
                for fp in frame_processors:
                    if full_head_applied:
                        continue
                    if fp.NAME == "DLC.FACE-ENHANCER":
                        if (
                            modules.globals.fp_ui["face_enhancer"]
                            and perf_mode != "Performance"
                        ):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN256":
                        if (
                            modules.globals.fp_ui.get("face_enhancer_gpen256", False)
                            and process_count % gpen256_interval == 0
                        ):
                            t0 = time.perf_counter()
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                            t_gpen += time.perf_counter() - t0
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN512":
                        if (
                            modules.globals.fp_ui.get("face_enhancer_gpen512", False)
                            and perf_mode != "Performance"
                        ):
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                    elif fp.NAME == "DLC.FACE-ENHANCER-GPEN1024":
                        if (
                            modules.globals.fp_ui.get("face_enhancer_gpen1024", False)
                            and perf_mode != "Performance"
                            and process_count % 2 == 0
                        ):
                            t0 = time.perf_counter()
                            temp_frame = fp.process_frame(
                                None, temp_frame, detected_faces=cached_faces
                            )
                            t_gpen += time.perf_counter() - t0
            # Final inner-mouth restore happens AFTER GPEN. It keeps the face
            # swapper's current mouth expression and avoids GPEN darkening it.
            if mouth_source_frame is not None and cached_faces:
                for face_index, target_face in enumerate(cached_faces):
                    try:
                        temp_frame = restore_target_mouth(
                            temp_frame, mouth_source_frame, target_face, track_key=face_index
                        )
                    except Exception as exc:
                        print(f"[MOUTH MASK] restore failed: {exc}", flush=True)

            # Applied last so the controls affect the exact image sent to OBS.
            # At neutral defaults this returns the original array immediately.
            temp_frame = _apply_output_color_adjustments(temp_frame)
            temp_frame = self._apply_virtual_background(temp_frame)

            # Keep the selected processing resolution. The preview widget will
            # scale it for display, but the frame itself remains at the chosen quality.
            total_ms = (time.perf_counter() - t_total) * 1000.0
            if not hasattr(self, "_profile_acc"):
                self._profile_acc = {"n": 0, "det": 0.0, "swap": 0.0,
                                     "gpen": 0.0, "other": 0.0, "total": 0.0,
                                     "last": time.time()}
            pa = self._profile_acc
            pa["n"] += 1
            pa["det"] += t_detection * 1000.0
            pa["swap"] += t_swap * 1000.0
            pa["gpen"] += t_gpen * 1000.0
            pa["other"] += t_other * 1000.0
            pa["total"] += total_ms
            nowp = time.time()
            if nowp - pa["last"] >= 1.0:
                n = max(1, pa["n"])
                avg_total = pa["total"] / n
                avg_fps = 1000.0 / max(1.0, avg_total)
                avg_det = pa["det"] / n
                avg_swap = pa["swap"] / n
                avg_gpen = pa["gpen"] / n
                # Smart FPS changes only virtual-background cadence. It needs
                # two consecutive measurements before changing anything, so a
                # one-off GPU spike never makes the subject edge flicker.
                if getattr(modules.globals, "smart_fps", True) and getattr(modules.globals, "virtual_background", False):
                    minimum = max(10, int(getattr(modules.globals, "smart_fps_minimum", 18)))
                    current_interval = max(1, int(getattr(modules.globals, "virtual_background_interval", 2)))
                    if avg_fps < minimum:
                        self._smart_fps_low_windows += 1
                        self._smart_fps_high_windows = 0
                        if self._smart_fps_low_windows >= 2 and current_interval < 6:
                            modules.globals.virtual_background_interval = current_interval + 1
                            self._smart_fps_low_windows = 0
                    elif avg_fps >= minimum + 3:
                        self._smart_fps_high_windows += 1
                        self._smart_fps_low_windows = 0
                        if self._smart_fps_high_windows >= 4 and current_interval > 2:
                            modules.globals.virtual_background_interval = current_interval - 1
                            self._smart_fps_high_windows = 0
                    else:
                        self._smart_fps_low_windows = 0
                        self._smart_fps_high_windows = 0
                else:
                    self._smart_fps_low_windows = 0
                    self._smart_fps_high_windows = 0
                print(
                    "[PROFILE] "
                    f"total={avg_total:.1f}ms | "
                    f"detect={avg_det:.1f}ms | "
                    f"swap={avg_swap:.1f}ms | "
                    f"GPEN={avg_gpen:.1f}ms | "
                    f"other={pa['other']/n:.1f}ms | frames={n}",
                    flush=True
                )
                self.metricsChanged.emit(
                    avg_fps, avg_det, avg_swap,
                    avg_gpen, avg_total,
                )
                pa.update({"n": 0, "det": 0.0, "swap": 0.0, "gpen": 0.0,
                           "other": 0.0, "total": 0.0, "last": nowp})

            process_count += 1
            # Target FPS is a soft limiter. It never delays a frame when the
            # pipeline is already slower than the requested target.
            target_fps = max(1, int(getattr(modules.globals, "target_fps", 20)))
            frame_budget = 1.0 / target_fps
            elapsed = time.perf_counter() - t_total
            if elapsed < frame_budget:
                time.sleep(frame_budget - elapsed)
            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            # The capture window must not inherit a diagnostic FPS overlay.
            # This shallow cost is paid only while that optional window is open.
            clean_output_frame = temp_frame.copy() if _OUTPUT_WINDOW_ACTIVE else temp_frame

            if modules.globals.show_fps:
                cv2.putText(
                    temp_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )

            self._publish_virtual_camera(temp_frame, target_fps)

            # The capture window is intentionally fed by the existing GUI
            # timer rather than a second worker.
            try:
                self._pq.put_nowait((temp_frame, preview_original, clean_output_frame))
            except queue.Full:
                try:
                    self._pq.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._pq.put_nowait((temp_frame, preview_original, clean_output_frame))
                except queue.Full:
                    pass

        self._close_virtual_camera()


class OutputWindow(QWidget):
    """A clean, ordinary window for TikTok/OBS window capture.

    It is deliberately not another camera or virtual device.  The UI timer
    sends it the already-processed frame that the live preview has received,
    so opening it does not add face-detection or model inference work.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLC Output")
        self.resize(720, 1280)
        self.setMinimumSize(320, 480)
        self.setStyleSheet("background: #000;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._image_label = QLabel("Ожидание LIVE…")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("color: #b8b8b8; background: #000;")
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._image_label, 1)

    def update_frame(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0 or not self.isVisible():
            return
        display = fit_image_to_widget(
            frame, max(1, self._image_label.width()), max(1, self._image_label.height())
        )
        self._image_label.setPixmap(_bgr_to_qpixmap(display))
        self._image_label.setText("")

    def closeEvent(self, event) -> None:
        global _OUTPUT_WINDOW, _OUTPUT_WINDOW_ACTIVE
        _OUTPUT_WINDOW_ACTIVE = False
        if _OUTPUT_WINDOW is self:
            _OUTPUT_WINDOW = None
        if _MAIN is not None and hasattr(_MAIN, "btn_output"):
            _MAIN.btn_output.setText(_("Output"))
        event.accept()


class WebcamPreviewWindow(QWidget):
    def __init__(self, camera_index: int, quality: str = "720p", parent=None):
        super().__init__(parent)
        # These attributes must exist before opening the device.  A failed
        # camera start schedules ``close()``, which used to call closeEvent
        # before _stop_event had been created and left Unity Capture in a
        # half-closed state.
        self._stop_event = threading.Event()
        self._capture_worker = None
        self._processing_worker = None
        self._timer = None
        self._cap = None
        self._last_processed_frame: Optional[np.ndarray] = None
        # When embedded in the dashboard, this widget behaves like a normal
        # child and fills the Preview host. Only standalone callers get a
        # floating window.
        if parent is None:
            self.setWindowTitle("Live Preview")
            self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._cap = VideoCapturer(camera_index)
        req_w, req_h = QUALITY_OPTIONS.get(quality, QUALITY_OPTIONS["720p"])
        if not self._cap.start(req_w, req_h, 60):
            # Fallback keeps the live camera usable if a device rejects the requested mode.
            if not self._cap.start(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT, 60):
                update_status(_("Failed to start camera"))
                QTimer.singleShot(0, self.close)
                return

        camera_fps = self._cap.actual_fps
        print(
            f"[webcam] Camera running at {self._cap.actual_width}x"
            f"{self._cap.actual_height}@{camera_fps:.0f}fps"
        )

        self._capture_queue: queue.Queue = queue.Queue(maxsize=2)
        self._processed_queue: queue.Queue = queue.Queue(maxsize=2)
        self._capture_worker = _CaptureWorker(
            self._cap, self._capture_queue, self._stop_event
        )
        self._processing_worker = _ProcessingWorker(
            self._capture_queue, self._processed_queue, self._stop_event, camera_fps
        )
        self._processing_worker.metricsChanged.connect(self._on_metrics)
        self._capture_worker.start()
        self._processing_worker.start()

        # Poll at ~2x camera fps so we never block but also don't burn CPU.
        poll_ms = max(1, min(16, int(500 / max(camera_fps, 1))))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(poll_ms)

    def _tick(self) -> None:
        if self._stop_event.is_set():
            self.close()
            return
        try:
            queued_frame = self._processed_queue.get_nowait()
        except queue.Empty:
            return
        clean_output_frame = None
        if isinstance(queued_frame, tuple) and len(queued_frame) == 3:
            processed_frame, original_frame, clean_output_frame = queued_frame
        elif isinstance(queued_frame, tuple):
            processed_frame, original_frame = queued_frame
        else:
            # Compatibility with frames queued by an already-running worker.
            processed_frame, original_frame = queued_frame, None
        self._last_processed_frame = processed_frame.copy()
        if _OUTPUT_WINDOW is not None:
            try:
                _OUTPUT_WINDOW.update_frame(clean_output_frame if clean_output_frame is not None else processed_frame)
            except RuntimeError:
                # The user may close the capture window while the GUI timer is
                # handling the previous frame.
                pass
        bgr_frame = (
            original_frame
            if getattr(modules.globals, "preview_original", False)
            and original_frame is not None
            else processed_frame
        )
        bgr_frame = fit_image_to_widget(bgr_frame, self.width(), self.height())
        self._image_label.setPixmap(_bgr_to_qpixmap(bgr_frame))

    def _on_metrics(self, fps: float, detect_ms: float, swap_ms: float,
                    gpen_ms: float, total_ms: float) -> None:
        """Update the dashboard from the GUI thread; no processing work here."""
        if _MAIN is None or not hasattr(_MAIN, "_dashboard_fps"):
            return
        if getattr(modules.globals, "show_diagnostics", False):
            _MAIN._dashboard_fps.setText(
                f"FPS: {fps:.1f}  •  D {detect_ms:.0f}  S {swap_ms:.0f}  G {gpen_ms:.0f} ms"
            )
        else:
            smart = ""
            if getattr(modules.globals, "smart_fps", False) and getattr(modules.globals, "virtual_background", False):
                smart = f"  •  Smart фон: 1/{getattr(modules.globals, 'virtual_background_interval', 2)}"
            _MAIN._dashboard_fps.setText(f"FPS: {fps:.1f}{smart}")

    def get_last_processed_frame(self) -> Optional[np.ndarray]:
        return self._last_processed_frame.copy() if self._last_processed_frame is not None else None

    def closeEvent(self, event) -> None:
        self._stop_event.set()
        try:
            if self._timer is not None:
                self._timer.stop()
        except Exception:
            pass
        for worker in (self._capture_worker, self._processing_worker):
            if worker is None:
                continue
            try:
                worker.wait(2000)
            except Exception:
                pass
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        global _WEBCAM_PREVIEW
        if _WEBCAM_PREVIEW is self:
            _WEBCAM_PREVIEW = None
        # Restore the dashboard placeholder when the embedded live widget stops.
        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "dashboard_preview_label"):
            try:
                parent.dashboard_preview_label.show()
            except Exception:
                pass
        event.accept()


def _open_webcam_preview(
    camera_index: int,
    quality: str = "720p",
    host=None,
    host_layout=None,
    placeholder=None,
) -> None:
    global _WEBCAM_PREVIEW
    if _WEBCAM_PREVIEW is not None:
        try:
            _WEBCAM_PREVIEW.close()
        except Exception:
            pass
        _WEBCAM_PREVIEW = None

    if host is not None:
        if placeholder is not None:
            placeholder.hide()
        _WEBCAM_PREVIEW = WebcamPreviewWindow(camera_index, quality, parent=host)
        if host_layout is None:
            host_layout = host.layout()
        if host_layout is not None:
            host_layout.addWidget(_WEBCAM_PREVIEW, 1)
        _WEBCAM_PREVIEW.show()
        return

    # Backward-compatible standalone mode for any existing non-dashboard caller.
    _WEBCAM_PREVIEW = WebcamPreviewWindow(camera_index, quality)
    _WEBCAM_PREVIEW.show()


# ─── mapper dialogs (image/video + live) ────────────────────────────────


def _make_thumb(cv2_img: np.ndarray) -> QPixmap:
    rgb = gpu_cvt_color(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize(
        (MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE), Image.LANCZOS
    )
    return _pil_to_qpixmap(image)


class MapperDialog(QDialog):
    """Source × Target mapper for image / video processing."""

    def __init__(self, start_cb: Callable, mapping: list):
        super().__init__(_MAIN)
        self._start_cb = start_cb
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_WIDTH, POPUP_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_submit = QPushButton(_("Submit"))
        btn_submit.clicked.connect(self._on_submit)
        layout.addWidget(btn_submit, alignment=Qt.AlignmentFlag.AlignCenter)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        for item in self._map:
            row = item["id"]
            btn = QPushButton(_("Select source image"))
            btn.setFixedWidth(200)
            btn.clicked.connect(lambda _c, n=row: self._select_source(n))
            grid.addWidget(btn, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px solid #555;")
            grid.addWidget(tgt_label, row, 3)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_source(self, row: int) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row]["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_submit(self) -> None:
        if has_valid_map():
            self.accept()
            _MAIN._select_output_and_start()
        else:
            self.set_status("Atleast 1 source with target is required!")


class LiveMapperDialog(QDialog):
    """Source × Target mapper for live webcam mode."""

    def __init__(self, camera_index: int, mapping: list):
        super().__init__(_MAIN)
        self._camera_index = camera_index
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_LIVE_WIDTH, POPUP_LIVE_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        for text, slot in (
            (_("Add"), self._on_add),
            (_("Clear"), self._on_clear),
            (_("Submit"), self._on_submit),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        for item in self._map:
            row = item["id"]
            btn_s = QPushButton(_("Select source image"))
            btn_s.setFixedWidth(200)
            btn_s.clicked.connect(lambda _c, n=row: self._select_face(n, "source"))
            grid.addWidget(btn_s, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            btn_t = QPushButton(_("Select target image"))
            btn_t.setFixedWidth(200)
            btn_t.clicked.connect(lambda _c, n=row: self._select_face(n, "target"))
            grid.addWidget(btn_t, row, 3)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(tgt_label, row, 4)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_face(self, row: int, kind: str) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row][kind] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_add(self) -> None:
        add_blank_map()
        self._rebuild()
        self.set_status("Please provide mapping!")

    def _on_clear(self) -> None:
        for item in self._map:
            item.pop("source", None)
            item.pop("target", None)
        self._rebuild()
        self.set_status("All mappings cleared!")

    def _on_submit(self) -> None:
        if has_valid_map():
            simplify_maps()
            self.set_status("Mappings successfully submitted!")
            self.accept()
            if _MAIN is not None:
                _open_webcam_preview(
                    self._camera_index,
                    getattr(modules.globals, "camera_resolution", "720p"),
                    host=_MAIN.dashboard_preview,
                    host_layout=_MAIN.dashboard_preview_layout,
                    placeholder=_MAIN.dashboard_preview_label,
                )
            else:
                _open_webcam_preview(self._camera_index)
        else:
            self.set_status("At least 1 source with target is required!")


def _open_mapper_dialog(start_cb: Callable, mapping: list) -> None:
    global _MAPPER
    close_mapper_window()
    _MAPPER = MapperDialog(start_cb, mapping)
    _MAPPER.show()


def _open_live_mapper_dialog(camera_index: int, mapping: list) -> None:
    global _LIVE_MAPPER
    close_mapper_window()
    _LIVE_MAPPER = LiveMapperDialog(camera_index, mapping)
    _LIVE_MAPPER.show()


def close_mapper_window() -> None:
    global _MAPPER, _LIVE_MAPPER
    if _MAPPER is not None:
        _MAPPER.close()
        _MAPPER = None
    if _LIVE_MAPPER is not None:
        _LIVE_MAPPER.close()
        _LIVE_MAPPER = None


# ─── entry point ─────────────────────────────────────────────────────────


class _Window:
    """Thin wrapper exposing .mainloop() for core.py compatibility."""

    def __init__(self, app: QApplication, main_window: MainWindow):
        self._app = app
        self._main = main_window

    def mainloop(self) -> None:
        self._main.show()
        self._app.exec()


def init(
    start: Callable[[], None], destroy: Callable[[], None], lang: str
) -> _Window:
    global _APP, _MAIN, _PREVIEW, _LANG, _BRIDGE

    _LANG = LanguageManager(lang)
    if QApplication.instance() is None:
        _APP = QApplication(sys.argv)
    else:
        _APP = QApplication.instance()
    _APP.setStyleSheet(QSS)

    _BRIDGE = _UIBridge()
    _MAIN = MainWindow(start, destroy)
    _PREVIEW = PreviewWindow()

    # Route status updates onto the UI thread regardless of caller.
    _BRIDGE.statusChanged.connect(_MAIN.set_status)

    return _Window(_APP, _MAIN)
