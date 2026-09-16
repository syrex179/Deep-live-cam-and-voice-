"""Reference-style PySide6 shell for Deep Live Studio.

The shell is deliberately a UI layer: the established camera and RVC programs
remain separate processes and this screen changes their real saved settings or
opens the corresponding module.  It never substitutes synthetic devices.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import psutil
from PySide6.QtCore import QPoint, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient, QWindow
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QStackedWidget, QVBoxLayout, QWidget,
)

from deep_live_studio import (
    CAMERA_STATE_PATH, DEFAULT_SCENES, ROOT, VOICE_ROOT, _read_json,
    _running, _start_camera, _start_voice, _stop, _write_json,
)


STYLE = """
* { font-family: 'Segoe UI'; color: #edf2ff; font-size: 13px; }
/* Keep every host viewport opaque.  The Camera module is built in the same
   QApplication, so an unstyled Qt viewport otherwise falls back to the light
   Windows palette after the module has been opened once. */
QMainWindow, QWidget#studioRoot, QStackedWidget, QStackedWidget > QWidget,
QScrollArea, QScrollArea > QWidget > QWidget { background: #050914; }
QFrame#topbar { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0b1223,stop:.55 #0a1120,stop:1 #0d1730); border-bottom: 1px solid #263b62; }
QFrame#sidebar { background: transparent; border: none; }
QFrame#card { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #101c31,stop:1 #0b1424); border: 1px solid #294575; border-radius: 14px; }
QFrame#inner { background: #101c31; border: 1px solid #29436e; border-radius: 10px; }
QLabel#brand { font-size: 25px; font-weight: 800; color: #f3f5ff; }
QLabel#tagline { font-size: 10px; font-weight: 700; letter-spacing: 2px; color: #bbc9ee; }
QLabel#title { font-size: 18px; font-weight: 800; color: #f4f6ff; }
QLabel#eyebrow { color: #bca8ff; font-size: 11px; font-weight: 700; letter-spacing: 6px; }
QLabel#section { color: #b9c9ef; font-weight: 700; }
QLabel#muted { color: #91a0be; }
QLabel#metric { background: #101a2d; border: 1px solid #29456f; border-radius: 10px; padding: 7px 13px; min-width: 58px; }
QLabel#green { color: #34e69b; font-weight: 700; }
QLabel#preview { background: #050912; border: 1px solid #34496a; border-radius: 7px; color: #8090ad; }
QPushButton { background: #16253d; border: 1px solid #38557f; border-radius: 8px; min-height: 30px; padding: 5px 13px; font-weight: 600; }
QPushButton:hover { background: #213657; border-color: #8c6dff; }
QPushButton#primary { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #6f3cf2,stop:1 #9b45ff); border-color: #b78bff; }
QPushButton#primary:hover { background: #854cf7; }
QPushButton#success { background: #0d9f59; border-color: #4be99d; }
QPushButton#assetNav { background: transparent; border: none; padding: 0; margin: 0; min-height: 54px; max-height: 54px; }
QPushButton#assetNav:hover { background: transparent; }
QPushButton#window { background: #111c2e; border: 1px solid #2c3c56; min-width: 32px; max-width: 32px; min-height: 30px; padding: 0; }
QPushButton#window:hover { background: #2a3a59; }
QComboBox, QLineEdit { background: #0b1423; border: 1px solid #30435f; border-radius: 6px; min-height: 29px; padding: 0 8px; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView { background: #111c2e; border: 1px solid #3b4f73; selection-background-color: #6338e8; }
QSlider::groove:horizontal { height: 5px; background: #283a58; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #7848ff; border-radius: 2px; }
QSlider::handle:horizontal { background: #f2f4ff; border: 2px solid #9b76ff; width: 12px; height: 12px; margin: -5px 0; border-radius: 7px; }
QCheckBox { spacing: 7px; color: #d9e1f6; }
QCheckBox::indicator { width: 28px; height: 16px; border-radius: 8px; background: #34455f; }
QCheckBox::indicator:checked { background: #7140f6; }
QScrollArea { border: none; }
QScrollBar:vertical { background: #070d19; width: 9px; }
QScrollBar::handle:vertical { background: #34435c; min-height: 35px; border-radius: 4px; }
"""


class HeroFrame(QFrame):
    """Paint a hero artwork proportionally, cropping only its outer edges."""

    def __init__(self, artwork: Path) -> None:
        super().__init__()
        self.setObjectName("hero")
        self._artwork = QPixmap(str(artwork))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(0, 0, self.width() - 1, self.height() - 1)
        clip = QPainterPath()
        clip.addRoundedRect(bounds, 16, 16)
        painter.save()
        painter.setClipPath(clip)
        if not self._artwork.isNull():
            # Keep the source aspect ratio: fill the short banner by cropping,
            # never by squeezing faces, microphones or the waveform.
            scaled = self._artwork.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap((self.width() - scaled.width()) // 2, (self.height() - scaled.height()) // 2, scaled)
        shade = QLinearGradient(0, 0, self.width(), 0)
        shade.setColorAt(0.00, QColor(3, 8, 20, 34))
        shade.setColorAt(0.36, QColor(3, 8, 20, 108))
        shade.setColorAt(0.64, QColor(3, 8, 20, 98))
        shade.setColorAt(1.00, QColor(3, 8, 20, 28))
        painter.fillRect(self.rect(), shade)
        painter.restore()
        painter.setPen(QPen(QColor("#6b5ae7"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(bounds, 16, 16)


class StudioBrandIdentity(QWidget):
    """Premium top-bar identity: rendered title plus the ribbon artwork."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(350, 56)
        source = QPixmap(str(ROOT / "assets" / "studio-ribbon-logo-v2.png"))
        # The art has a navy background matching the top bar.  Crop only its
        # generous export padding so the glossy ribbon reads at header size.
        self._logo = source.copy(220, 230, 810, 810) if not source.isNull() else source

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self._logo.isNull():
            mark = self._logo.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap(0, (self.height() - mark.height()) // 2, mark)

        text_x = 62
        title_font = QFont("Segoe UI", 22, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor("#fbfcff"))
        painter.drawText(text_x, 27, "Deep Live ")
        studio_x = text_x + painter.fontMetrics().horizontalAdvance("Deep Live ")
        studio_gradient = QLinearGradient(studio_x, 0, studio_x + 76, 0)
        studio_gradient.setColorAt(0.0, QColor("#d37bff"))
        studio_gradient.setColorAt(0.52, QColor("#a442ff"))
        studio_gradient.setColorAt(1.0, QColor("#6f58ff"))
        painter.setPen(QPen(studio_gradient, 1))
        painter.drawText(studio_x, 27, "Studio")

        tagline_font = QFont("Segoe UI", 9, QFont.Weight.DemiBold)
        tagline_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.75)
        painter.setFont(tagline_font)
        painter.setPen(QColor("#b5c5eb"))
        painter.drawText(text_x, 45, "AI FACE & VOICE REALTIME")


class Ring(QWidget):
    def __init__(self, label: str, color: str) -> None:
        super().__init__()
        self.label, self.color, self.value = label, QColor(color), 0
        # A ring must preserve a square drawing area; allowing the horizontal
        # layout to stretch it turns the circular indicators into ovals.
        self.setFixedSize(112, 104)

    def set_value(self, value: float) -> None:
        self.value = max(0, min(100, int(value)))
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = min(self.width() - 30, self.height() - 35)
        left = (self.width() - size) / 2
        box = QRectF(left, 5, size, size)
        pen = QPen(QColor("#202e48"), 7)
        painter.setPen(pen)
        painter.drawArc(box, 0, 360 * 16)
        pen.setColor(self.color)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(box, 90 * 16, -int(360 * 16 * self.value / 100))
        painter.setPen(QColor("#eff3ff"))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, f"{self.value}%")
        painter.setPen(QColor("#b8c4df"))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(self.rect().adjusted(0, 64, 0, 0), Qt.AlignmentFlag.AlignHCenter, self.label)


def _paint_line_icon(painter: QPainter, kind: str, box: QRectF, color: QColor, width: float = 2.0) -> None:
    """Draw small, font-independent launcher icons."""
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    x, y, w, h = box.x(), box.y(), box.width(), box.height()
    if kind == "home":
        painter.drawLine(QPoint(int(x + w * .16), int(y + h * .48)), QPoint(int(x + w * .5), int(y + h * .18)))
        painter.drawLine(QPoint(int(x + w * .5), int(y + h * .18)), QPoint(int(x + w * .84), int(y + h * .48)))
        painter.drawRoundedRect(QRectF(x + w * .25, y + h * .45, w * .5, h * .38), 2, 2)
        painter.drawLine(QPoint(int(x + w * .50), int(y + h * .83)), QPoint(int(x + w * .50), int(y + h * .61)))
    elif kind == "camera":
        painter.drawRoundedRect(QRectF(x + w * .14, y + h * .34, w * .72, h * .45), 3, 3)
        painter.drawRoundedRect(QRectF(x + w * .38, y + h * .22, w * .25, h * .14), 2, 2)
        painter.drawEllipse(QRectF(x + w * .38, y + h * .43, w * .25, h * .25))
    elif kind == "face":
        for x1, y1, x2, y2 in ((.16,.39,.16,.20),(.16,.20,.34,.20),(.84,.39,.84,.20),(.84,.20,.66,.20),(.16,.61,.16,.80),(.16,.80,.34,.80),(.84,.61,.84,.80),(.84,.80,.66,.80)):
            painter.drawLine(QPoint(int(x+w*x1),int(y+h*y1)), QPoint(int(x+w*x2),int(y+h*y2)))
        painter.drawEllipse(QRectF(x + w * .39, y + h * .34, w * .22, h * .22))
        painter.drawArc(QRectF(x + w * .34, y + h * .49, w * .32, h * .24), 25 * 16, 130 * 16)
    elif kind == "microphone":
        painter.drawRoundedRect(QRectF(x + w * .38, y + h * .15, w * .24, h * .49), 6, 6)
        painter.drawArc(QRectF(x + w * .27, y + h * .39, w * .46, h * .36), 0, -180 * 16)
        painter.drawLine(QPoint(int(x+w*.5),int(y+h*.75)), QPoint(int(x+w*.5),int(y+h*.90)))
        painter.drawLine(QPoint(int(x+w*.33),int(y+h*.90)), QPoint(int(x+w*.67),int(y+h*.90)))
    elif kind == "output":
        painter.drawEllipse(QRectF(x + w * .42, y + h * .43, w * .15, h * .15))
        # Symmetric broadcast waves, matching the reference rather than a
        # one-sided speaker symbol.
        painter.drawArc(QRectF(x + w * .24, y + h * .28, w * .52, h * .44), 118 * 16, 124 * 16)
        painter.drawArc(QRectF(x + w * .11, y + h * .15, w * .78, h * .70), 127 * 16, 106 * 16)
        painter.drawArc(QRectF(x + w * .24, y + h * .28, w * .52, h * .44), -62 * 16, 124 * 16)
        painter.drawArc(QRectF(x + w * .11, y + h * .15, w * .78, h * .70), -53 * 16, 106 * 16)
    elif kind == "scenes":
        painter.setBrush(QColor(color))
        for dx, dy in ((.12,.15),(.57,.15),(.12,.60),(.57,.60)):
            painter.drawRoundedRect(QRectF(x + w*dx, y + h*dy, w*.31, h*.31), 1.8, 1.8)
        painter.setBrush(Qt.BrushStyle.NoBrush)
    elif kind == "models":
        top, right, middle, left = (QPoint(int(x+w*.5),int(y+h*.12)), QPoint(int(x+w*.82),int(y+h*.30)), QPoint(int(x+w*.5),int(y+h*.48)), QPoint(int(x+w*.18),int(y+h*.30)))
        painter.drawLine(top, right); painter.drawLine(right, middle); painter.drawLine(middle, left); painter.drawLine(left, top)
        painter.drawLine(QPoint(int(x+w*.18),int(y+h*.30)), QPoint(int(x+w*.18),int(y+h*.66)))
        painter.drawLine(QPoint(int(x+w*.82),int(y+h*.30)), QPoint(int(x+w*.82),int(y+h*.66)))
        painter.drawLine(QPoint(int(x+w*.18),int(y+h*.66)), QPoint(int(x+w*.5),int(y+h*.85)))
        painter.drawLine(QPoint(int(x+w*.82),int(y+h*.66)), QPoint(int(x+w*.5),int(y+h*.85)))
        painter.drawLine(QPoint(int(x+w*.5),int(y+h*.48)), QPoint(int(x+w*.5),int(y+h*.85)))
    elif kind == "settings":
        # Compact cog with distinct teeth, matching the reference glyph.
        cx, cy = x + w * .5, y + h * .5
        painter.setPen(QPen(color, width * 1.35, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for step in range(8):
            angle = step * math.pi / 4.0
            dx, dy = math.cos(angle), math.sin(angle)
            painter.drawLine(
                QPoint(int(cx + dx * w * .27), int(cy + dy * h * .27)),
                QPoint(int(cx + dx * w * .40), int(cy + dy * h * .40)),
            )
        painter.setBrush(QColor(color))
        painter.drawEllipse(QRectF(x + w*.25, y + h*.25, w*.50, h*.50))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#0d1a31"))
        painter.drawEllipse(QRectF(x + w*.43, y + h*.43, w*.14, h*.14))
        painter.setBrush(Qt.BrushStyle.NoBrush)
    elif kind == "logs":
        painter.drawRoundedRect(QRectF(x+w*.25,y+h*.12,w*.5,h*.76), 2, 2)
        painter.drawLine(QPoint(int(x+w*.61), int(y+h*.12)), QPoint(int(x+w*.75), int(y+h*.27)))
        painter.drawLine(QPoint(int(x+w*.61), int(y+h*.12)), QPoint(int(x+w*.61), int(y+h*.27)))
        for yy in (.35,.52,.69): painter.drawLine(QPoint(int(x+w*.35),int(y+h*yy)), QPoint(int(x+w*.65),int(y+h*yy)))
    elif kind == "about":
        painter.drawEllipse(QRectF(x+w*.13,y+h*.10,w*.74,h*.74))
        painter.drawPoint(QPoint(int(x+w*.5),int(y+h*.31)))
        painter.drawLine(QPoint(int(x+w*.5),int(y+h*.43)), QPoint(int(x+w*.5),int(y+h*.68)))


class VectorIcon(QWidget):
    def __init__(self, kind: str, color: str, size: int = 28) -> None:
        super().__init__()
        self.kind, self.color = kind, QColor(color)
        self.setFixedSize(size, size)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        _paint_line_icon(painter, self.kind, QRectF(self.rect()).adjusted(2, 2, -2, -2), self.color, 2.0)


class IconNavButton(QPushButton):
    _ACCENTS = {
        "home": "#b184ff", "camera": "#79afff", "microphone": "#c18aff",
        "scenes": "#75afff", "models": "#bd96ff", "output": "#78baff",
        "settings": "#83b9ff", "logs": "#c295ff", "about": "#7bb8ff",
    }

    def __init__(self, text: str, kind: str) -> None:
        super().__init__(text)
        self.kind = kind

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        active = self.isChecked()
        hovered = self.underMouse()
        accent = QColor(self._ACCENTS.get(self.kind, "#a9c8ff"))
        tile = QRectF(9, (self.height() - 36) / 2, 36, 36)
        fill = QLinearGradient(tile.topLeft(), tile.bottomRight())
        if active:
            fill.setColorAt(0, QColor("#7d47ff"))
            fill.setColorAt(1, QColor("#4d24c5"))
            tile_border = QColor("#ba91ff")
            icon_color = QColor("#ffffff")
        elif hovered:
            fill.setColorAt(0, accent.darker(220))
            fill.setColorAt(1, QColor("#0c1a33"))
            tile_border = accent.lighter(125)
            icon_color = accent.lighter(145)
        else:
            fill.setColorAt(0, accent.darker(300))
            fill.setColorAt(1, QColor("#09152a"))
            tile_border = accent.darker(165)
            icon_color = accent
        painter.setPen(QPen(tile_border, 1.1))
        painter.setBrush(fill)
        painter.drawRoundedRect(tile, 11, 11)
        if active:
            painter.setPen(QPen(QColor("#f4eaff"), 3.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPoint(3, 10), QPoint(3, self.height() - 10))
        _paint_line_icon(painter, self.kind, tile.adjusted(7.5, 7.5, -7.5, -7.5), icon_color, 1.85)


class AssetNavButton(QPushButton):
    """A real navigation button which uses the supplied PNG only as its icon."""

    def __init__(self, text: str, asset: Path) -> None:
        super().__init__("")
        self.setAccessibleName(text)
        self.setToolTip(text)
        self.setObjectName("assetNav")
        self.setCheckable(True)
        self.setFixedHeight(54)
        self._text = text
        self._pixmap = QPixmap(str(asset))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        active = self.isChecked()
        hovered = self.underMouse()

        if active:
            fill = QLinearGradient(rect.topLeft(), rect.topRight())
            fill.setColorAt(0, QColor("#6d2fff"))
            fill.setColorAt(1, QColor("#8a42ff"))
            painter.setPen(QPen(QColor(209, 177, 255, 190), 1.0))
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, 11, 11)
            # A restrained glow is painted outside the card's content.
            painter.setPen(QPen(QColor(126, 60, 255, 80), 2.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect.adjusted(-.5, -.5, .5, .5), 11, 11)
        elif hovered:
            painter.setPen(QPen(QColor(75, 65, 145, 150), 1.0))
            painter.setBrush(QColor("#101a33"))
            painter.drawRoundedRect(rect, 11, 11)
        else:
            painter.setPen(QPen(QColor(30, 55, 95, 64), 1.0))
            painter.setBrush(QColor("#091224"))
            painter.drawRoundedRect(rect, 11, 11)

        if not self._pixmap.isNull():
            icon = self._pixmap.scaled(
                32, 32, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(12, (self.height() - icon.height()) // 2, icon)

        painter.setPen(QColor("#ffffff") if active else QColor("#c8d7f3"))
        font = QFont("Segoe UI", 10, QFont.Weight.DemiBold if active else QFont.Weight.Medium)
        font.setPixelSize(15)
        painter.setFont(font)
        painter.drawText(QRectF(56, 0, self.width() - 66, self.height()), Qt.AlignmentFlag.AlignVCenter, self._text)


class SidebarFrame(QFrame):
    """The sidebar shell is painted behind the supplied, real button PNGs."""

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(self.rect()).adjusted(.5, .5, -.5, -.5)
        clip = QPainterPath()
        clip.addRoundedRect(bounds, 16, 16)

        painter.save()
        painter.setClipPath(clip)
        base = QLinearGradient(0, 0, self.width(), self.height())
        base.setColorAt(0, QColor("#070b19"))
        base.setColorAt(.52, QColor("#060b18"))
        base.setColorAt(1, QColor("#070c1a"))
        painter.fillRect(self.rect(), base)
        painter.restore()

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(80, 120, 220, 90), 1.0))
        painter.drawRoundedRect(bounds, 16, 16)


class SyrexFooter(QWidget):
    """Responsive brand card rendered as real UI, not an embedded image."""

    def __init__(self, asset: Path) -> None:
        super().__init__()
        self.setObjectName("syrexFooter")
        self._pixmap = QPixmap(str(asset))
        self.setMinimumHeight(189)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if not self._pixmap.isNull():
            painter.drawPixmap(self.rect(), self._pixmap)
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(.5, .5, -.5, -.5)
        panel = QLinearGradient(rect.topLeft(), rect.bottomRight())
        panel.setColorAt(0.0, QColor("#080c23"))
        panel.setColorAt(.50, QColor("#07152a"))
        panel.setColorAt(1.0, QColor("#071127"))
        painter.setPen(QPen(QColor("#7849e9"), 1.2))
        painter.setBrush(panel)
        painter.drawRoundedRect(rect, 15, 15)
        clip = QPainterPath(); clip.addRoundedRect(rect, 15, 15)
        painter.save(); painter.setClipPath(clip)
        violet = QRadialGradient(-self.width() * .08, self.height() * .70, self.width() * .72)
        violet.setColorAt(0.0, QColor(131, 38, 255, 130)); violet.setColorAt(.46, QColor(92, 27, 211, 52)); violet.setColorAt(1.0, QColor(22, 10, 60, 0))
        painter.fillRect(self.rect(), violet)
        blue = QRadialGradient(self.width() * 1.04, self.height() * .30, self.width() * .70)
        blue.setColorAt(0.0, QColor(0, 121, 255, 150)); blue.setColorAt(.46, QColor(15, 82, 207, 50)); blue.setColorAt(1.0, QColor(5, 25, 62, 0))
        painter.fillRect(self.rect(), blue)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        curve = QPainterPath(QPoint(0, int(self.height() * .84)))
        curve.cubicTo(self.width() * .20, self.height() * .52, self.width() * .43, self.height() * .63, self.width() * .67, self.height() * .39)
        painter.setPen(QPen(QColor(143, 53, 255, 180), 1.15)); painter.drawPath(curve)
        curve2 = QPainterPath(QPoint(int(self.width() * .50), self.height()))
        curve2.cubicTo(self.width() * .67, self.height() * .62, self.width() * .91, self.height() * .65, self.width(), self.height() * .31)
        painter.setPen(QPen(QColor(30, 126, 255, 185), 1.05)); painter.drawPath(curve2)
        painter.restore()

        logo_font = QFont("Segoe UI", 31, QFont.Weight.Bold)
        logo_font.setItalic(True)
        painter.setFont(logo_font)
        gradient = QLinearGradient(self.width() * .42, 17, self.width() * .60, 54)
        gradient.setColorAt(0, QColor("#9b7bff"))
        gradient.setColorAt(.52, QColor("#7b45ef"))
        gradient.setColorAt(1, QColor("#368eff"))
        painter.setPen(QPen(QColor(83, 35, 172, 130), 2.4))
        painter.drawText(QRectF(1, 14, self.width(), 43), Qt.AlignmentFlag.AlignHCenter, "S")
        painter.setPen(QPen(gradient, 1))
        painter.drawText(QRectF(0, 13, self.width(), 43), Qt.AlignmentFlag.AlignHCenter, "S")

        painter.setPen(QColor("#d3e2ff"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        painter.drawText(QRectF(0, 55, self.width(), 17), Qt.AlignmentFlag.AlignHCenter, "by Syrex")
        painter.setPen(QColor("#a7c4f5"))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Medium))
        painter.drawText(QRectF(0, 78, self.width(), 27), Qt.AlignmentFlag.AlignHCenter, "Real People.\nNew Possibilities.")
        painter.setPen(QPen(QColor("#2f6dbb"), 1))
        painter.drawLine(QPoint(25, 120), QPoint(self.width() - 25, 120))
        painter.setPen(QColor("#98b6ee"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Medium))
        painter.drawText(QRectF(0, 127, self.width(), 17), Qt.AlignmentFlag.AlignHCenter, "v1.0.0")


class SidebarFooterCard(QFrame):
    """Footer built from Qt widgets instead of a footer screenshot."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("sidebarFooter")
        self.setFixedHeight(180)
        self.setStyleSheet("""
            QFrame#sidebarFooter {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #0b1024, stop:.55 #081426, stop:1 #091226);
                border: 1px solid #6541e9; border-radius: 14px;
            }
            QLabel { background: transparent; border: none; }
            QLabel#footerLogo { color: #9b55ff; font-size: 30px; font-weight: 700; font-style: italic; }
            QLabel#footerBy { color: #d8e4ff; font-size: 13px; font-weight: 600; }
            QLabel#footerCopy { color: #a8c0ed; font-size: 11px; }
            QLabel#footerRule { background: #29528b; max-height: 1px; min-height: 1px; }
            QLabel#footerVersion { color: #9ab8ee; font-size: 11px; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 13, 18, 12)
        layout.setSpacing(2)
        logo = QLabel("S"); logo.setObjectName("footerLogo"); logo.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        by = QLabel("by Syrex"); by.setObjectName("footerBy"); by.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        copy = QLabel("Real People.\nNew Possibilities."); copy.setObjectName("footerCopy"); copy.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        rule = QLabel(); rule.setObjectName("footerRule")
        version = QLabel("v1.0.0"); version.setObjectName("footerVersion"); version.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(logo)
        layout.addWidget(by)
        layout.addSpacing(8)
        layout.addWidget(copy)
        layout.addSpacing(10)
        layout.addWidget(rule)
        layout.addSpacing(8)
        layout.addWidget(version)


class ModuleGlyph(QWidget):
    """Premium camera/microphone glyphs used in the launcher cards."""

    def __init__(self, kind: str, accent: str) -> None:
        super().__init__()
        self.kind = kind
        self.accent = QColor(accent)
        self.setFixedSize(62, 62)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        box = self.rect().adjusted(1, 1, -1, -1)
        fill = QLinearGradient(box.topLeft(), box.bottomRight())
        fill.setColorAt(0, self.accent)
        fill.setColorAt(1, QColor("#18254b"))
        painter.setPen(QPen(QColor("#b69aff"), 1))
        painter.setBrush(fill)
        painter.drawRoundedRect(box, 16, 16)

        pen = QPen(QColor("#f4f7ff"), 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self.kind == "camera":
            painter.drawRoundedRect(17, 25, 28, 19, 4, 4)
            painter.drawRoundedRect(24, 20, 12, 7, 2, 2)
            painter.drawEllipse(26, 29, 11, 11)
            painter.drawPoint(42, 29)
        else:
            painter.drawRoundedRect(25, 17, 12, 24, 6, 6)
            painter.drawArc(20, 25, 22, 23, 0, -180 * 16)
            painter.drawLine(31, 48, 31, 53)
            painter.drawLine(24, 53, 38, 53)


class DragBar(QFrame):
    def __init__(self, window: QMainWindow) -> None:
        super().__init__()
        self.window = window
        self._origin: QPoint | None = None
        self.setObjectName("topbar")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._origin is not None and event.buttons() & Qt.MouseButton.LeftButton:
            now = event.globalPosition().toPoint()
            self.window.move(self.window.pos() + now - self._origin)
            self._origin = now
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._origin = None
        super().mouseReleaseEvent(event)


def _top_level_window_for_process(pid: int) -> int | None:
    """Return the main visible window for a process on Windows."""
    if os.name != "nt":
        return None
    windows: list[int] = []
    user32 = ctypes.windll.user32
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    @callback_type
    def visit(hwnd, _data):
        owner = user32.GetWindow(hwnd, 4)  # GW_OWNER
        process_id = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        # The RVC module deliberately starts hidden so it never flashes as a
        # separate application before Studio attaches it to this page.
        if process_id.value == pid and not owner:
            windows.append(int(hwnd))
            return False
        return True

    user32.EnumWindows(visit, 0)
    return windows[0] if windows else None


class EmbeddedModulePage(QWidget):
    """Hosts a real module window inside a Studio page instead of beside it."""
    def __init__(self, title: str, description: str) -> None:
        super().__init__()
        self._handle: int | None = None
        self._foreign_window: QWindow | None = None
        self._container: QWidget | None = None
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(8)
        self._heading = QLabel(title)
        self._heading.setObjectName("title")
        self.layout.addWidget(self._heading)
        self.placeholder = QLabel(description + "\n\nНажми кнопку запуска в Studio — интерфейс появится здесь.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setObjectName("preview")
        self.layout.addWidget(self.placeholder, 1)

    def _set_embedded_layout(self, embedded: bool) -> None:
        """Remove the shell-only heading once a real module is mounted."""
        self._heading.setVisible(not embedded)
        if embedded:
            self.layout.setContentsMargins(0, 0, 0, 0)
            self.layout.setSpacing(0)
        else:
            self.layout.setContentsMargins(6, 6, 6, 6)
            self.layout.setSpacing(8)

    def attach(self, hwnd: int) -> bool:
        if self._handle == hwnd and self._container is not None:
            return True
        foreign = QWindow.fromWinId(hwnd)
        if foreign is None:
            return False
        if self._container is not None:
            self.layout.removeWidget(self._container)
            self._container.deleteLater()
        self.placeholder.hide()
        self._set_embedded_layout(True)
        self._foreign_window = foreign
        self._container = QWidget.createWindowContainer(foreign, self)
        self._container.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.layout.addWidget(self._container, 1)
        if os.name == "nt":
            # Qt normally reparents a foreign QWindow itself.  Some PySide6
            # builds leave it top-level, so enforce the parent relationship
            # for the isolated RVC process as well.
            user32 = ctypes.windll.user32
            user32.SetParent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            user32.SetParent.restype = ctypes.c_void_p
            user32.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
            user32.GetWindowLongPtrW.restype = ctypes.c_longlong
            user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong]
            user32.SetWindowLongPtrW.restype = ctypes.c_longlong
            style = user32.GetWindowLongPtrW(ctypes.c_void_p(hwnd), -16)
            style = (style & ~0x80000000) | 0x40000000  # WS_POPUP -> WS_CHILD
            user32.SetWindowLongPtrW(ctypes.c_void_p(hwnd), -16, style)
            user32.SetParent(ctypes.c_void_p(hwnd), ctypes.c_void_p(int(self._container.winId())))
            user32.ShowWindow(ctypes.c_void_p(hwnd), 5)  # SW_SHOW
        self._handle = hwnd
        return True

    def attach_widget(self, widget: QWidget) -> None:
        """Place a native Studio-owned module widget in this page."""
        if self._container is not None:
            self.layout.removeWidget(self._container)
            self._container.deleteLater()
            self._container = None
        self.placeholder.hide()
        self._set_embedded_layout(True)
        widget.setParent(self)
        widget.setWindowFlags(Qt.WindowType.Widget)
        self.layout.addWidget(widget, 1)
        self._container = widget
        self._handle = None

    def reset(self) -> None:
        if self._container is not None:
            self.layout.removeWidget(self._container)
            self._container.deleteLater()
        self._container = None
        self._foreign_window = None
        self._handle = None
        self._set_embedded_layout(False)
        self.placeholder.show()


class StudioShell(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Deep Live Studio")
        self.setMinimumSize(1280, 800)
        self.resize(1536, 1024)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setStyleSheet(STYLE)
        self._gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="studio-metrics")
        self._gpu_future: Future[str] | None = None
        self._gpu_text = "GPU: —"
        self._last_gpu_request = 0.0
        self._camera_module_ready = False
        self._voice_module_ready = False
        self._voice_module_loading = False
        self.camera_state = _read_json(CAMERA_STATE_PATH, {})
        self.voice_config_path = VOICE_ROOT / "configs" / "config.json"
        self.voice_state = _read_json(self.voice_config_path, {})

        root = QWidget()
        root.setObjectName("studioRoot")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(4, 2, 4, 4)
        outer.setSpacing(6)
        outer.addWidget(self._build_topbar())
        body = QHBoxLayout()
        body.setSpacing(10)
        body.addWidget(self._build_sidebar())
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_dashboard())
        self.camera_module_page = EmbeddedModulePage("Камера / Face Swap", "Рабочее окно Deep Live Cam ещё не запущено.")
        self.pages.addWidget(self.camera_module_page)
        self.voice_module_page = EmbeddedModulePage("Голос (RVC)", "Рабочее окно Deep Live Voice ещё не запущено.")
        self.pages.addWidget(self.voice_module_page)
        self.pages.addWidget(self._build_scenes_page())
        self.pages.addWidget(self._build_models_page())
        self.pages.addWidget(self._build_output_page())
        self.pages.addWidget(self._build_settings_page())
        self.pages.addWidget(self._build_logs_page())
        self.pages.addWidget(self._simple_page("О программе", "Deep Live Studio объединяет управление Deep Live Cam и RVC Voice в одном окне.", "Открыть папку", self._open_root))
        body.addWidget(self.pages, 1)
        outer.addLayout(body, 1)
        outer.addWidget(self._build_statusbar())
        self.setCentralWidget(root)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_metrics)
        self.timer.start(5000)
        self._refresh_metrics()
    def _build_topbar(self) -> QWidget:
        bar = DragBar(self)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 6, 12, 6)
        layout.addWidget(StudioBrandIdentity())
        layout.addStretch(1)
        self.top_cpu, self.top_gpu, self.top_ram, self.top_vram = [QLabel("—") for _ in range(4)]
        for label in (self.top_cpu, self.top_gpu, self.top_ram, self.top_vram):
            label.setObjectName("metric")
            layout.addWidget(label)
        self.ready = QLabel("●  Модули остановлены")
        self.ready.setObjectName("green")
        self.ready.setStyleSheet("background:#0d2530; border:1px solid #254957; border-radius:8px; padding:9px 12px;")
        layout.addWidget(self.ready)
        for text, callback in (("—", self.showMinimized), ("□", self._toggle_max), ("×", self.close)):
            button = QPushButton(text)
            button.setObjectName("window")
            button.clicked.connect(callback)
            layout.addWidget(button)
        return bar

    def _build_sidebar(self) -> QWidget:
        sidebar = SidebarFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(195)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        asset_dir = ROOT / "assets" / "sidebar-clean"
        labels = [
            ("Студия", "studio_home.png", 0), ("Камера", "camera.png", 1), ("Голос", "voice.png", 2),
            ("Сцены", "scenes.png", 3), ("Модели", "models.png", 4), ("OBS / Вывод", "obs_output.png", 5),
            ("Настройки", "settings.png", 6), ("Логи", "logs.png", 7), ("О программе", "about.png", 8),
        ]
        self.nav: list[QPushButton] = []
        for text, asset_name, index in labels:
            button = AssetNavButton(text, asset_dir / asset_name)
            button.clicked.connect(lambda _checked=False, i=index: self._select_page(i))
            layout.addWidget(button)
            self.nav.append(button)
        self.nav[0].setChecked(True)
        layout.addStretch(1)
        layout.addSpacing(3)
        layout.addWidget(SidebarFooterCard())
        return sidebar

    def _card(self, title: str, icon: str = "") -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 11, 12, 12)
        layout.setSpacing(8)
        label = QLabel(f"{icon}  {title}".strip())
        label.setObjectName("title")
        layout.addWidget(label)
        return card, layout

    def _build_dashboard(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        outer = QVBoxLayout(content); outer.setContentsMargins(16, 2, 16, 14); outer.setSpacing(16)
        hero = HeroFrame(ROOT / "assets" / "studio-hero-ai-v2.png")
        hero.setFixedHeight(208)
        hero_box = QVBoxLayout(hero); hero_box.setContentsMargins(24, 13, 24, 12); hero_box.setSpacing(1)
        welcome = QLabel("Д О Б Р О   П О Ж А Л О В А Т Ь   В"); welcome.setObjectName("eyebrow"); welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome.setStyleSheet("font-size:8px; font-weight:700; letter-spacing:5px; color:#d5caff; background:transparent;")
        heading = QLabel("Deep Live <span style='color:#b054ff'>Studio</span>"); heading.setAlignment(Qt.AlignmentFlag.AlignCenter); heading.setStyleSheet("font-size:36px; font-weight:800; color:#f8f9ff; background:transparent;")
        sub = QLabel("AI FACE & VOICE REALTIME\nЕдиный центр управления камерой, заменой лица и голосом.\nБольше возможностей. Реальное время. Без границ."); sub.setAlignment(Qt.AlignmentFlag.AlignCenter); sub.setStyleSheet("color:#d2daf3; font-size:11px; line-height:135%; background:transparent;")
        pill = QLabel("Технологии, которые сближают твои идеи"); pill.setAlignment(Qt.AlignmentFlag.AlignCenter); pill.setStyleSheet("color:#d6caff; font-size:10px; border:1px solid #7861db; border-radius:12px; padding:4px 14px; background:rgba(11,20,50,170);")
        hero_box.addWidget(welcome); hero_box.addWidget(heading); hero_box.addWidget(sub); hero_box.addWidget(pill, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addWidget(hero)
        products = QHBoxLayout(); products.setSpacing(16)
        products.addWidget(self._launcher_card("▣", "Камера / Face Swap", "Управление камерой, заменой лица\nи виртуальным видео в реальном времени.", ["Реалистичная замена лица", "Виртуальная камера (OBS / Unity)", "Поддержка GPEN, DFM и других моделей", "Гибкие настройки и улучшение качества"], "Открыть камеру   →", lambda: self._select_page(1), "#7538f4"), 1)
        products.addWidget(self._launcher_card("♩", "Голос / RVC", "Управление микрофоном, RVC\nи преобразованием голоса.", ["Высокое качество звука", "Поддержка .pth и .index", "Настройка голоса в реальном времени", "Низкая задержка", "Работает с любыми приложениями"], "Открыть голос   →", lambda: self._select_page(2), "#216df3"), 1)
        outer.addLayout(products)
        quick_row = QHBoxLayout(); quick_row.addStretch(1)
        quick = QPushButton("ϟ   Быстрый запуск"); quick.setObjectName("primary"); quick.setMinimumWidth(230); quick.clicked.connect(self._launch_all); quick_row.addWidget(quick)
        start_all = QPushButton("▶   Запустить всё"); start_all.setMinimumWidth(185); start_all.clicked.connect(self._launch_all); quick_row.addWidget(start_all); quick_row.addStretch(1)
        outer.addLayout(quick_row)
        bottom = QHBoxLayout(); bottom.setSpacing(14)
        status, status_box = self._card("Статус модулей", "◈")
        # These are dashboard summaries, not a second work area.  Keep both
        # summary cards compact so their content sits directly under the title.
        status.setFixedHeight(166)
        status_row = QHBoxLayout(); status_row.setSpacing(8)
        self.dashboard_module_status: dict[str, QLabel] = {}
        for key, icon, name, color in (("camera", "camera", "Камера", "#a463ff"), ("face", "face", "Face Swap", "#a463ff"), ("voice", "microphone", "Голос (RVC)", "#36a7ff"), ("output", "output", "OBS / Вывод", "#58a4ff")):
            tile = QFrame(); tile.setObjectName("inner"); tile.setMinimumHeight(108); box=QVBoxLayout(tile); box.setContentsMargins(8, 7, 8, 7)
            lab=VectorIcon(icon, color, 31); box.addWidget(lab, 0, Qt.AlignmentFlag.AlignHCenter)
            text=QLabel(name); text.setAlignment(Qt.AlignmentFlag.AlignCenter); box.addWidget(text)
            ready=QLabel("●  Проверка…"); ready.setObjectName("green"); ready.setAlignment(Qt.AlignmentFlag.AlignCenter); box.addWidget(ready)
            self.dashboard_module_status[key] = ready; status_row.addWidget(tile)
        status_box.addLayout(status_row); bottom.addWidget(status, 1)
        self.camera_badge = QLabel(); self.camera_status = QLabel(); self.face_status = QLabel(); self.voice_status = QLabel(); self.output_status = QLabel()
        bottom.addWidget(self._build_launcher_resources(), 1)
        outer.addLayout(bottom)
        scroll.setWidget(content)
        return scroll

    def _launcher_card(self, icon: str, title: str, description: str, benefits: list[str], action: str, callback, color: str) -> QWidget:
        card = QFrame(); card.setObjectName("card"); box = QVBoxLayout(card); box.setContentsMargins(25, 20, 25, 16); box.setSpacing(8)
        background_file = ROOT / "assets" / ("studio-camera-card-bg-v1.png" if "Камера" in title else "studio-voice-card-bg-v1.png")
        if background_file.is_file():
            card.setStyleSheet(f"QFrame#card {{ border-image: url('{background_file.as_posix()}') 0 0 0 0 stretch stretch; border:1px solid #496caa; border-radius:14px; }}")
        head=QHBoxLayout(); head.setSpacing(12)
        glyph=ModuleGlyph("camera" if "Камера" in title else "microphone", color); head.addWidget(glyph)
        texts=QVBoxLayout(); texts.setSpacing(2)
        name=QLabel(title); name.setStyleSheet("font-size:21px;font-weight:800;"); texts.addWidget(name)
        desc=QLabel(description); desc.setObjectName("muted"); desc.setWordWrap(True); texts.addWidget(desc)
        head.addLayout(texts, 1)
        badge = QLabel("◉  РЕАЛЬНОЕ ВРЕМЯ" if "Камера" in title else "◉  ВЫСОКОЕ КАЧЕСТВО"); badge.setFixedWidth(132); badge.setAlignment(Qt.AlignmentFlag.AlignCenter); badge.setStyleSheet(f"color:{color}; background:#0a1730; border:1px solid {color}; border-radius:10px; padding:5px 6px; font-size:9px; font-weight:700;"); head.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        box.addLayout(head)
        body = QHBoxLayout(); body.setSpacing(4)
        left = QVBoxLayout(); left.setSpacing(7)
        for benefit in benefits:
            item=QLabel("✓   " + benefit); item.setWordWrap(True); item.setStyleSheet("color:#d4ddf4; padding:2px; background:transparent;"); left.addWidget(item)
        left.addStretch(1); body.addLayout(left, 3); body.addStretch(2); box.addLayout(body, 1)
        button=QPushButton(action); button.setObjectName("primary"); button.setMinimumHeight(40); button.setStyleSheet(f"QPushButton {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {color},stop:1 #8650ff); border:1px solid #bb9bff; border-radius:9px; font-size:14px; font-weight:800; }} QPushButton:hover {{ background:#8e57ff; }}"); button.clicked.connect(callback); box.addWidget(button)
        return card

    def _build_launcher_resources(self) -> QWidget:
        card, layout = self._card("Системные ресурсы", "◉")
        card.setFixedHeight(166)
        row = QHBoxLayout(); row.setSpacing(10)
        self.cpu_ring = Ring("CPU", "#8357ff")
        self.gpu_ring = Ring("GPU", "#8357ff")
        self.ram_ring = Ring("RAM", "#3c81ff")
        self.vram_ring = Ring("VRAM", "#d24cff")
        for ring in (self.cpu_ring, self.gpu_ring, self.ram_ring, self.vram_ring):
            row.addWidget(ring)
        layout.addLayout(row)
        return card

    def _launch_all(self) -> None:
        self._show_camera_module()
        QTimer.singleShot(100, self._show_voice_module)
        self.bottom_status.setText("●  Модули подготовлены внутри Deep Live Studio")

    def _build_camera_card(self) -> QWidget:
        card, layout = self._card("Камера / Превью", "▣")
        preview = QLabel("Предпросмотр камеры")
        preview.setObjectName("preview")
        preview.setMinimumHeight(225)
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        source = self.camera_state.get("last_source_cache_path")
        if isinstance(source, str) and Path(source).is_file():
            pixmap = QPixmap(source)
            if not pixmap.isNull():
                preview.setPixmap(pixmap.scaled(520, 245, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(preview)
        bar = QHBoxLayout()
        self.camera_badge = QLabel("● Камера остановлена")
        self.camera_badge.setObjectName("green")
        self.camera_format = QLabel("LIVE • локальная камера")
        self.camera_format.setObjectName("muted")
        bar.addWidget(self.camera_badge)
        bar.addStretch(1)
        bar.addWidget(self.camera_format)
        layout.addLayout(bar)
        note = QLabel("Полные настройки и запуск камеры находятся сразу во вкладке «Камера».")
        note.setObjectName("muted")
        layout.addWidget(note)
        form = QGridLayout()
        for row, (label, key, values) in enumerate((
            ("Камера", "last_camera_name", [str(self.camera_state.get("last_camera_name") or "По умолчанию")]),
            ("Разрешение", "preview_quality", ["720p", "1080p", "1440p"]),
            ("FPS", "target_fps", ["15", "20", "25", "30"]),
            ("Устройство", "virtual_camera", ["OBS Virtual Camera", "Обычный вывод"]),
        )):
            form.addWidget(QLabel(label), row, 0)
            combo = QComboBox()
            combo.addItems(values)
            current = str(self.camera_state.get(key, values[0]))
            combo.setCurrentText(current if current in values else values[0])
            if key == "virtual_camera":
                combo.setCurrentText("OBS Virtual Camera" if self.camera_state.get(key, False) else "Обычный вывод")
                combo.currentTextChanged.connect(lambda value: self._save_camera("virtual_camera", value == "OBS Virtual Camera"))
            elif key == "target_fps":
                combo.currentTextChanged.connect(lambda value: self._save_camera("target_fps", int(value)))
            else:
                combo.currentTextChanged.connect(lambda value, field=key: self._save_camera(field, value))
            form.addWidget(combo, row, 1)
        layout.addLayout(form)
        switches = QHBoxLayout()
        switches.addWidget(self._toggle("Зеркальное отображение", "live_mirror"))
        switches.addWidget(self._toggle("Показывать превью", "show_fps"))
        layout.addLayout(switches)
        return card

    def _build_face_card(self) -> QWidget:
        card, layout = self._card("Live Face Swap", "☻")
        header = QHBoxLayout()
        header.addStretch(1)
        header.addWidget(QLabel("Модель"))
        model = QComboBox()
        model.addItems(["GPEN-256 (рекомендуется)", "GPEN-512", "GPEN-1024 (медленно)", "Без улучшения"])
        model.setCurrentText("GPEN-256 (рекомендуется)")
        model.currentTextChanged.connect(self._choose_enhancer)
        header.addWidget(model)
        layout.addLayout(header)
        rows = QHBoxLayout()
        source_box = QFrame(); source_box.setObjectName("inner")
        source_layout = QVBoxLayout(source_box)
        source_layout.addWidget(QLabel("Исходное лицо"))
        self.source_preview = QLabel("Выберите лицо")
        self.source_preview.setObjectName("preview")
        self.source_preview.setFixedSize(180, 115)
        self.source_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_source_preview()
        source_layout.addWidget(self.source_preview)
        select = QPushButton("Выбрать лицо"); select.setObjectName("primary"); select.clicked.connect(self._select_source)
        source_layout.addWidget(select)
        video = QPushButton("Видео"); video.clicked.connect(self._select_source_video); source_layout.addWidget(video)
        full_head = QPushButton("Полная голова (β)")
        full_head.clicked.connect(lambda: self._save_camera("full_head_mode", True))
        source_layout.addWidget(full_head)
        dfm = QPushButton("Обученная личность (DFM)")
        dfm.clicked.connect(lambda: self._save_camera("trained_identity_mode", True))
        source_layout.addWidget(dfm)
        rows.addWidget(source_box)
        target_box = QFrame(); target_box.setObjectName("inner")
        target_layout = QVBoxLayout(target_box)
        target_layout.addWidget(QLabel("Цель"))
        target_preview = QLabel("▧\n\nПеретащите изображение\nили выберите файл\n\nJPG, PNG, WEBP")
        target_preview.setObjectName("preview")
        target_preview.setMinimumSize(190, 155)
        target_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        target_layout.addWidget(target_preview)
        target = QPushButton("Выбрать цель"); target.setObjectName("primary"); target.clicked.connect(self._select_target)
        target_layout.addWidget(target)
        rows.addWidget(target_box)
        controls = QFrame(); controls.setObjectName("inner")
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(QLabel("Параметры лица"))
        controls_layout.addWidget(self._slider("Сходство", "opacity", 0, 100, int(float(self.camera_state.get("opacity", 1)) * 100), lambda n: n / 100))
        controls_layout.addWidget(self._slider("Резкость", "sharpness", 0, 10, int(float(self.camera_state.get("sharpness", 0)) * 10), lambda n: n / 10))
        controls_layout.addWidget(self._slider("Маска", "mouth_mask_size", 0, 100, int(float(self.camera_state.get("mouth_mask_size", 0))), int))
        controls_layout.addWidget(self._slider("Размытие границ", "mask_feather", 0, 50, int(float(self.camera_state.get("mask_feather", 18))), int))
        controls_layout.addWidget(self._value_toggle("Стабилизация лица", "face_stabilization", 60.0))
        controls_layout.addWidget(self._value_toggle("Защита глаз", "eye_protection", 85.0))
        controls_layout.addWidget(self._toggle("Сопоставление цвета", "color_match"))
        controls_layout.addWidget(self._toggle("Виртуальный фон", "virtual_background"))
        rows.addWidget(controls, 1)
        layout.addLayout(rows)
        self.face_fps = QLabel("Полные параметры Face Swap доступны сразу во вкладке «Камера».")
        self.face_fps.setObjectName("muted")
        layout.addWidget(self.face_fps)
        return card

    def _build_voice_card(self) -> QWidget:
        card, layout = self._card("Голос (RVC)", "♩")
        rows = QHBoxLayout()
        devices = QFrame(); devices.setObjectName("inner")
        devices_layout = QVBoxLayout(devices)
        devices_layout.addWidget(QLabel("Аудиоустройство"))
        for label, key in (("Тип", "sg_hostapi"), ("Вход", "sg_input_device"), ("Выход", "sg_output_device")):
            line = QLineEdit(str(self.voice_state.get(key, "Не выбрано")))
            line.editingFinished.connect(lambda field=key, edit=line: self._save_voice(field, edit.text()))
            devices_layout.addWidget(QLabel(label)); devices_layout.addWidget(line)
        devices_layout.addWidget(QLabel("Управление и запуск — сразу во вкладке «Голос»."))
        rows.addWidget(devices)
        models = QFrame(); models.setObjectName("inner")
        models_layout = QVBoxLayout(models)
        models_layout.addWidget(QLabel("Загрузка модели"))
        self.pth_line = QLineEdit(Path(str(self.voice_state.get("pth_path", ""))).name)
        self.index_line = QLineEdit(Path(str(self.voice_state.get("index_path", ""))).name)
        for text, edit, callback in (("Файл модели (.pth)", self.pth_line, self._select_pth), ("Файл индекса (.index)", self.index_line, self._select_index)):
            models_layout.addWidget(QLabel(text))
            row = QHBoxLayout(); row.addWidget(edit); choose = QPushButton("▣"); choose.clicked.connect(callback); row.addWidget(choose); models_layout.addLayout(row)
        models_layout.addWidget(QLabel("Модель голоса"))
        rows.addWidget(models)
        settings = QFrame(); settings.setObjectName("inner")
        settings_layout = QVBoxLayout(settings)
        settings_layout.addWidget(QLabel("Настройки голоса"))
        settings_layout.addWidget(self._voice_slider("Громкость", "rms_mix_rate", 0, 100, int(float(self.voice_state.get("rms_mix_rate", 0)) * 100), lambda n: n / 100))
        settings_layout.addWidget(self._voice_slider("Высота (Pitch)", "pitch", -24, 24, int(self.voice_state.get("pitch", 0)), int))
        settings_layout.addWidget(self._voice_slider("Индекс", "index_rate", 0, 100, int(float(self.voice_state.get("index_rate", 0)) * 100), lambda n: n / 100))
        rows.addWidget(settings)
        layout.addLayout(rows)
        return card

    def _build_resources_card(self) -> QWidget:
        card, layout = self._card("Системные ресурсы", "◉")
        row = QHBoxLayout()
        self.cpu_ring = Ring("CPU", "#8357ff")
        self.gpu_ring = Ring("GPU", "#8357ff")
        self.ram_ring = Ring("RAM", "#3c81ff")
        self.vram_ring = Ring("VRAM", "#d24cff")
        for ring in (self.cpu_ring, self.gpu_ring, self.ram_ring, self.vram_ring): row.addWidget(ring)
        layout.addLayout(row)
        status_box = QFrame(); status_box.setObjectName("inner")
        status_layout = QVBoxLayout(status_box)
        status_layout.addWidget(QLabel("Статус модулей"))
        self.camera_status = QLabel(); self.face_status = QLabel("●  Face Swap                         Готов")
        self.voice_status = QLabel(); self.output_status = QLabel("●  OBS / Вывод                       Готов")
        for item in (self.camera_status, self.face_status, self.voice_status, self.output_status):
            item.setObjectName("green"); status_layout.addWidget(item)
        layout.addWidget(status_box)
        return card

    def _build_output_card(self) -> QWidget:
        card, layout = self._card("Предпросмотр / Вывод", "▧")
        row = QHBoxLayout()
        for label, values, key in (("Видео вывод", ["OBS Virtual Camera", "Обычный вывод"], "virtual_camera"), ("Разрешение", ["1280 × 720", "1920 × 1080"], "preview_quality"), ("FPS", ["20", "25", "30"], "target_fps")):
            column = QVBoxLayout(); column.addWidget(QLabel(label)); combo = QComboBox(); combo.addItems(values)
            if key == "virtual_camera": combo.setCurrentIndex(0 if self.camera_state.get(key, False) else 1)
            combo.currentIndexChanged.connect(lambda _value, field=key, box=combo: self._save_camera(field, box.currentIndex() == 0 if field == "virtual_camera" else box.currentText()))
            column.addWidget(combo); row.addLayout(column)
        row.addStretch(1)
        obs = QPushButton("Открыть OBS"); obs.clicked.connect(self._open_obs)
        cam = QPushButton("Открыть Deep Live Cam"); cam.clicked.connect(self._open_camera)
        row.addWidget(obs); row.addWidget(cam)
        layout.addLayout(row)
        return card

    def _simple_page(self, title: str, description: str, action: str, callback) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(20, 20, 20, 20)
        card, inner = self._card(title)
        text = QLabel(description); text.setObjectName("muted"); text.setWordWrap(True); inner.addWidget(text)
        button = QPushButton(action); button.setObjectName("primary"); button.clicked.connect(callback); inner.addWidget(button)
        layout.addWidget(card); layout.addStretch(1); return page

    def _build_scenes_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(20, 20, 20, 20)
        card, inner = self._card("Сцены", "☷")
        inner.addWidget(QLabel("Сцены сохраняют реальные параметры камеры. Для уже работающей камеры потребуется перезапуск."))
        for name, scene in DEFAULT_SCENES.items():
            button = QPushButton(f"{name} — {scene['description']}"); button.clicked.connect(lambda _checked=False, n=name: self._apply_scene(n)); inner.addWidget(button)
        layout.addWidget(card); layout.addStretch(1); return page

    def _build_models_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(20, 20, 20, 20)
        card, inner = self._card("Модели", "▦")
        pth = self.voice_state.get("pth_path") or "Модель RVC не выбрана"
        index = self.voice_state.get("index_path") or "Индекс RVC не выбран"
        for title, value in (("RVC .pth", pth), ("RVC .index", index), ("Face enhancer", "GPEN-256 / GPEN-512 / GPEN-1024")):
            row = QLabel(f"{title}:  {value}"); row.setWordWrap(True); inner.addWidget(row)
        open_models = QPushButton("Открыть папку моделей"); open_models.clicked.connect(self._open_voice_models); inner.addWidget(open_models)
        layout.addWidget(card); layout.addStretch(1); return page

    def _build_output_page(self) -> QWidget:
        return self._simple_page("OBS / Вывод", "OBS Virtual Camera отправляет готовый кадр в Telegram, Discord, OBS и другие приложения. Включение сохраняется в реальных настройках камеры.", "Открыть камеру", self._open_camera)

    def _build_settings_page(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(20, 20, 20, 20)
        card, inner = self._card("Настройки", "⚙")
        inner.addWidget(self._toggle("Умный FPS", "smart_fps"))
        inner.addWidget(self._toggle("Виртуальная камера", "virtual_camera"))
        inner.addWidget(self._toggle("Виртуальный фон", "virtual_background"))
        note = QLabel("Настройки сохраняются сразу. Если камера уже запущена, перезапусти её, чтобы применить параметры старта.")
        note.setObjectName("muted"); note.setWordWrap(True); inner.addWidget(note)
        layout.addWidget(card); layout.addStretch(1); return page

    def _build_logs_page(self) -> QWidget:
        return self._simple_page("Логи", "Логи камеры и голоса хранятся локально. Ошибки не отправляются в сеть.", "Открыть папку логов", self._open_logs)

    def _build_statusbar(self) -> QWidget:
        bar = QFrame(); bar.setObjectName("topbar")
        layout = QHBoxLayout(bar); layout.setContentsMargins(14, 4, 14, 4)
        layout.addWidget(QLabel("▶  Deep Live Studio v1.0.0  |  by Syrex"))
        layout.addStretch(1)
        self.bottom_status = QLabel("●  Все системы готовы")
        self.bottom_status.setObjectName("green")
        layout.addWidget(self.bottom_status)
        layout.addStretch(1)
        folder = QPushButton("▣  Открыть папку"); folder.clicked.connect(self._open_root)
        layout.addWidget(folder)
        return bar

    def _toggle(self, text: str, key: str, truthy=None):
        from PySide6.QtWidgets import QCheckBox
        check = QCheckBox(text)
        value = self.camera_state.get(key, False)
        check.setChecked(bool(truthy(value) if truthy else value))
        check.toggled.connect(lambda enabled, field=key: self._save_camera(field, enabled))
        return check

    def _value_toggle(self, text: str, key: str, on_value: float):
        """A visible on/off switch for a numeric camera refinement control."""
        from PySide6.QtWidgets import QCheckBox
        check = QCheckBox(text)
        check.setChecked(float(self.camera_state.get(key, 0.0) or 0.0) > 0.0)
        check.toggled.connect(lambda enabled, field=key, value=on_value: self._save_camera(field, value if enabled else 0.0))
        return check

    def _slider(self, label: str, key: str, low: int, high: int, current: int, convert):
        box = QWidget(); row = QHBoxLayout(box); row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label)); slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(low, high); slider.setValue(max(low, min(high, current)))
        value = QLabel(str(slider.value())); value.setFixedWidth(33); value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        slider.valueChanged.connect(lambda amount: (value.setText(str(amount)), self._save_camera(key, convert(amount))))
        row.addWidget(slider, 1); row.addWidget(value); return box

    def _voice_slider(self, label: str, key: str, low: int, high: int, current: int, convert):
        box = QWidget(); row = QHBoxLayout(box); row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label)); slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(low, high); slider.setValue(max(low, min(high, current)))
        value = QLabel(str(slider.value())); value.setFixedWidth(30)
        slider.valueChanged.connect(lambda amount: (value.setText(str(amount)), self._save_voice(key, convert(amount))))
        row.addWidget(slider, 1); row.addWidget(value); return box

    def _save_camera(self, key: str, value) -> None:
        self.camera_state[key] = value
        _write_json(CAMERA_STATE_PATH, self.camera_state)
        self.bottom_status.setText("●  Настройки камеры сохранены")

    def _save_voice(self, key: str, value) -> None:
        self.voice_state[key] = value
        _write_json(self.voice_config_path, self.voice_state)
        self.bottom_status.setText("●  Настройки голоса сохранены — перезапусти голосовой модуль")

    def _select_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите исходное лицо", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self._save_camera("last_source_media_path", path); self._save_camera("last_source_cache_path", path)
            self._update_source_preview()

    def _select_source_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите исходное видео", "", "Video (*.mp4 *.mkv *.avi *.mov)")
        if path:
            self._save_camera("last_source_media_path", path)
            self._save_camera("full_head_mode", True)

    def _select_target(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите цель", "", "Media (*.png *.jpg *.jpeg *.webp *.mp4 *.mkv)")
        if path: self._save_camera("target_path", path)

    def _select_pth(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите RVC модель", str(VOICE_ROOT), "RVC model (*.pth)")
        if path: self.pth_line.setText(Path(path).name); self._save_voice("pth_path", path)

    def _select_index(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите индекс", str(VOICE_ROOT), "RVC index (*.index)")
        if path: self.index_line.setText(Path(path).name); self._save_voice("index_path", path)

    def _update_source_preview(self) -> None:
        path = self.camera_state.get("last_source_cache_path") or self.camera_state.get("last_source_media_path")
        if isinstance(path, str) and Path(path).is_file():
            pixmap = QPixmap(path)
            if not pixmap.isNull(): self.source_preview.setPixmap(pixmap.scaled(self.source_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)); self.source_preview.setText("")

    def _choose_enhancer(self, label: str) -> None:
        selected = {"GPEN-256 (рекомендуется)": "face_enhancer_gpen256", "GPEN-512": "face_enhancer_gpen512", "GPEN-1024 (медленно)": "face_enhancer_gpen1024"}.get(label)
        self.camera_state["fp_ui"] = {"face_enhancer": False, "face_enhancer_gpen256": selected == "face_enhancer_gpen256", "face_enhancer_gpen512": selected == "face_enhancer_gpen512", "face_enhancer_gpen1024": selected == "face_enhancer_gpen1024"}
        _write_json(CAMERA_STATE_PATH, self.camera_state)

    def _apply_scene(self, name: str) -> None:
        self.camera_state.update(DEFAULT_SCENES[name]["camera"])
        _write_json(CAMERA_STATE_PATH, self.camera_state)
        self.bottom_status.setText(f"●  Сцена «{name}» сохранена")

    def _show_camera_module(self) -> None:
        # This loader is scheduled by the camera route.  It must not select a
        # page itself: a queued load must never override a later navigation.
        if self._camera_module_ready:
            self.bottom_status.setText("●  Камера работает внутри Deep Live Studio")
            return
        # The established camera UI is a PySide widget too.  Building it in
        # this QApplication avoids a second top-level Deep Live Camera window.
        try:
            from modules import core as camera_core
            from modules import globals as camera_globals
            from modules import ui as camera_ui
            previous_argv = list(sys.argv)
            try:
                sys.argv = [previous_argv[0], "--execution-provider", "cuda"]
                camera_core.parse_args()
            finally:
                sys.argv = previous_argv
            if not camera_core.pre_check():
                raise RuntimeError("проверка компонентов камеры не пройдена")
            for processor in camera_core.get_frame_processors_modules(camera_globals.frame_processors):
                if not processor.pre_check():
                    raise RuntimeError(f"не готов процессор {processor.NAME}")
            camera_core.limit_resources()
            camera_ui.init(camera_core.start, lambda: camera_core.destroy(False), camera_globals.lang)
            camera_widget = camera_ui._MAIN
            if camera_widget is None:
                raise RuntimeError("интерфейс камеры не создан")
            # `camera_ui.init()` historically installs its QSS on the shared
            # QApplication.  Studio and Camera now share that QApplication in
            # order to embed a real Camera widget, so keep the camera theme
            # local to its own widget and immediately restore the host theme.
            # This prevents the white dashboard gutters and lets the Studio
            # shell retain its navy palette after Camera has been opened.
            camera_widget.setStyleSheet(camera_ui.QSS)
            self.setStyleSheet(STYLE)
            self.camera_module_page.attach_widget(camera_widget)
            self._camera_module_ready = True
            self.bottom_status.setText("●  Камера работает внутри Deep Live Studio")
        except Exception as exc:
            self.bottom_status.setText(f"●  Не удалось встроить камеру: {exc}")

    def _stop_camera(self) -> None:
        _stop("camera")
        if self._camera_module_ready and self.camera_module_page._container is not None:
            try:
                self.camera_module_page._container.close()
            except RuntimeError:
                pass
            self._camera_module_ready = False
        self.camera_module_page.reset()
        self.bottom_status.setText("●  Камера остановлена")

    def _show_voice_module(self) -> None:
        # This method is usually scheduled by ``_select_page(2)``.  Do not
        # change the selected page while the voice UI is loading: otherwise a
        # queued load could pull the user back from another sidebar page.
        if self._voice_module_ready or self._voice_module_loading:
            return
        self._voice_module_loading = True
        try:
            module_file = VOICE_ROOT / "ui_reference.py"
            spec = importlib.util.spec_from_file_location("studio_voice_reference", module_file)
            if spec is None or spec.loader is None:
                raise RuntimeError("не найден ui_reference.py")
            reference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reference)
            voice_ui = reference.MainWindow()
            voice_ui.pth_path.setText(str(self.voice_state.get("pth_path", "")))
            voice_ui.index_path.setText(str(self.voice_state.get("index_path", "")))
            voice_ui.input_noise.setChecked(bool(self.voice_state.get("I_noise_reduce", False)))
            voice_ui.output_noise.setChecked(bool(self.voice_state.get("O_noise_reduce", False)))
            voice_ui.model_path_changed.connect(lambda extension, path: self._save_voice(f"{extension}_path", path))
            voice_ui.device_changed.connect(self._save_voice_devices)
            voice_ui.conversion_started.connect(lambda: self._start_voice_engine(voice_ui))
            voice_ui.conversion_stopped.connect(self._stop_voice_engine)
            voice_ui.microphone_test_requested.connect(lambda: self._request_voice_command("mic_test"))
            self.voice_module_page.attach_widget(voice_ui)
            self._voice_module_ready = True
            self.bottom_status.setText("●  Все системы готовы")
        except Exception as exc:
            self.bottom_status.setText(f"●  Не удалось загрузить Deep Live Voice: {exc}")
        finally:
            self._voice_module_loading = False

    def _save_voice_devices(self, input_device: str, output_device: str, hostapi: str) -> None:
        self._save_voice("sg_input_device", input_device)
        self._save_voice("sg_output_device", output_device)
        self._save_voice("sg_hostapi", hostapi)

    def _request_voice_command(self, action: str) -> None:
        command_path = VOICE_ROOT / "configs" / "studio_command.json"
        _write_json(command_path, {"id": time.time_ns(), "action": action})

    def _start_voice_engine(self, voice_ui) -> None:
        self._save_voice("I_noise_reduce", voice_ui.input_noise.isChecked())
        self._save_voice("O_noise_reduce", voice_ui.output_noise.isChecked())
        self._save_voice("pth_path", voice_ui.pth_path.text())
        self._save_voice("index_path", voice_ui.index_path.text())
        _stop("voice")
        error = _start_voice()
        if error:
            self.bottom_status.setText("●  " + error)
            return
        self.bottom_status.setText("●  Запускаю RVC-конвертацию…")
        QTimer.singleShot(4500, lambda: self._request_voice_command("start"))

    def _stop_voice_engine(self) -> None:
        self._request_voice_command("stop")
        self.bottom_status.setText("●  RVC-конвертация остановлена")

    def _embed_module(self, kind: str, page: EmbeddedModulePage, remaining: int) -> None:
        candidates = []
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                command = " ".join(process.info.get("cmdline") or []).lower()
                if kind == "camera" and "run.py" in command and str(ROOT).lower() in command:
                    candidates.append(process.info["pid"])
                if kind == "voice" and "realtime_gui.py" in command and str(VOICE_ROOT).lower() in command:
                    candidates.append(process.info["pid"])
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        for pid in candidates:
            hwnd = _top_level_window_for_process(pid)
            if hwnd is not None and page.attach(hwnd):
                if kind == "voice":
                    self._voice_module_loading = False
                self.bottom_status.setText("●  Модуль работает внутри Deep Live Studio")
                return
        if remaining > 0:
            QTimer.singleShot(700, lambda: self._embed_module(kind, page, remaining - 1))
        else:
            if kind == "voice":
                self._voice_module_loading = False
            self.bottom_status.setText("●  Не удалось встроить окно модуля. Перезапусти Studio.")

    def _open_camera(self) -> None: self._select_page(1)
    def _open_obs(self) -> None: self.bottom_status.setText("●  В приложении выбери устройство OBS Virtual Camera")
    def _open_root(self) -> None: subprocess.Popen(["explorer", str(ROOT)])
    def _open_voice_models(self) -> None: subprocess.Popen(["explorer", str(VOICE_ROOT)])
    def _open_logs(self) -> None: subprocess.Popen(["explorer", str(ROOT / "logs")])

    def _select_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        for number, button in enumerate(self.nav): button.setChecked(number == index)
        if index == 1 and not self._camera_module_ready:
            QTimer.singleShot(0, self._show_camera_module)
        elif index == 2 and not self._voice_module_ready and not self._voice_module_loading:
            QTimer.singleShot(0, self._show_voice_module)

    def _toggle_max(self) -> None:
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def _refresh_metrics(self) -> None:
        # The embedded Camera shares this process and CUDA driver with Studio.
        # During LIVE, avoid process enumeration and use the lightweight NVML
        # reader below; unlike spawning nvidia-smi it has no console process
        # or driver-wide command invocation to contend with CUDA inference.
        if self._camera_module_ready:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            self.top_cpu.setText(f"CPU\n{cpu:.0f}%")
            self.top_ram.setText(f"RAM\n{memory.used / 1024**3:.1f} / {memory.total / 1024**3:.1f} GB")
            self.cpu_ring.set_value(cpu)
            self.ram_ring.set_value(memory.percent)
            if self._gpu_future is not None and self._gpu_future.done():
                try:
                    self._gpu_text = self._gpu_future.result()
                except Exception:
                    self._gpu_text = "GPU\n—|VRAM\n—|0|0"
                self._gpu_future = None
            if self._gpu_future is None and time.monotonic() - self._last_gpu_request > 5:
                self._gpu_future = self._gpu_executor.submit(self._read_gpu)
                self._last_gpu_request = time.monotonic()
            parts = self._gpu_text.split("|")
            self.top_gpu.setText(parts[0] if parts else "GPU\n—")
            self.top_vram.setText(parts[1] if len(parts) > 1 else "VRAM\n—")
            try:
                gpu, vram = [int(value) for value in parts[2:4]]
            except (ValueError, TypeError):
                gpu, vram = 0, 0
            self.gpu_ring.set_value(gpu)
            self.vram_ring.set_value(vram)
            self.ready.setText("●  Камера работает — лёгкий мониторинг активен")
            return
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        self.top_cpu.setText(f"CPU\n{cpu:.0f}%")
        self.top_ram.setText(f"RAM\n{memory.used / 1024**3:.1f} / {memory.total / 1024**3:.1f} GB")
        self.cpu_ring.set_value(cpu); self.ram_ring.set_value(memory.percent)
        if self._gpu_future is not None and self._gpu_future.done():
            try: self._gpu_text = self._gpu_future.result()
            except Exception: self._gpu_text = "GPU: —"
            self._gpu_future = None
        if self._gpu_future is None and time.monotonic() - self._last_gpu_request > 10:
            self._gpu_future = self._gpu_executor.submit(self._read_gpu); self._last_gpu_request = time.monotonic()
        parts = self._gpu_text.split("|")
        self.top_gpu.setText(parts[0] if parts else "GPU\n—")
        self.top_vram.setText(parts[1] if len(parts) > 1 else "VRAM\n—")
        try:
            gpu, vram = [int(value) for value in parts[2:4]]
        except (ValueError, TypeError): gpu, vram = 0, 0
        self.gpu_ring.set_value(gpu); self.vram_ring.set_value(vram)
        cam, voice = _running("camera"), _running("voice")
        self.camera_badge.setText("● Камера запущена" if cam else "● Камера остановлена")
        self.camera_status.setText("●  Камера                         Работает" if cam else "●  Камера                         Отключена")
        self.voice_status.setText("●  Голос (RVC)                   Работает" if voice else "●  Голос (RVC)                   Отключён")
        if hasattr(self, "dashboard_module_status"):
            state = self.dashboard_module_status
            state["camera"].setText("●  Работает" if cam else "●  Готов")
            state["face"].setText("●  Активен" if cam else "●  Готов")
            state["voice"].setText("●  Работает" if voice else "●  Готов")
            state["output"].setText("●  Включён" if self.camera_state.get("virtual_camera", False) else "●  Готов")
        modules_ready = "●  Модули активны" if cam or voice else "●  Все системы готовы"
        self.ready.setText(modules_ready)

    @staticmethod
    def _read_gpu() -> str:
        """Read GPU state through the NVML DLL without starting nvidia-smi."""
        try:
            nvml = ctypes.WinDLL("nvml.dll")
            # nvmlUtilization_t and nvmlMemory_t.  The final ``reserved``
            # field is present in modern NVML; the first three fields are
            # compatible with older drivers too.
            class _Utilization(ctypes.Structure):
                _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
            class _Memory(ctypes.Structure):
                _fields_ = [
                    ("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong), ("reserved", ctypes.c_ulonglong),
                ]
            if nvml.nvmlInit_v2() != 0:
                return "GPU\n—|VRAM\n—|0|0"
            try:
                handle = ctypes.c_void_p()
                if nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
                    return "GPU\n—|VRAM\n—|0|0"
                utilization = _Utilization()
                memory = _Memory()
                if nvml.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(utilization)) != 0:
                    return "GPU\n—|VRAM\n—|0|0"
                if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
                    return "GPU\n—|VRAM\n—|0|0"
                used = memory.used / 1024**3
                total = max(0.01, memory.total / 1024**3)
                return f"GPU\n{utilization.gpu}%|VRAM\n{used:.1f} / {total:.1f} GB|{int(utilization.gpu)}|{int(used / total * 100)}"
            finally:
                nvml.nvmlShutdown()
        except (AttributeError, OSError, ValueError):
            pass
        return "GPU\n—|VRAM\n—|0|0"

    def closeEvent(self, event) -> None:
        self._gpu_executor.shutdown(wait=False, cancel_futures=True); event.accept()


def run_studio() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = StudioShell(); window.show()
    return app.exec()
