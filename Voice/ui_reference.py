import sys
import math
import time
import os
import ctypes
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal, QRectF, QPointF, QSize
from PySide6.QtGui import (
    QColor, QPainter, QPainterPath, QPen, QBrush, QLinearGradient,
    QFont, QFontMetrics, QIcon, QPixmap
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QToolButton, QLineEdit, QComboBox, QCheckBox, QRadioButton,
    QSlider, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout,
    QVBoxLayout, QHBoxLayout, QScrollArea, QFileDialog, QListWidget,
    QListWidgetItem, QStackedWidget, QSizePolicy, QButtonGroup,
    QMessageBox, QAbstractSpinBox
)

try:
    from PySide6.QtMultimedia import (
        QAudioFormat, QAudioDevice, QMediaDevices, QAudioSource
    )
    MULTIMEDIA_AVAILABLE = True
except ImportError:
    MULTIMEDIA_AVAILABLE = False


APP_QSS = r"""
* {
    font-family: "Segoe UI", "Arial", sans-serif;
    color: #e9ecf7;
}
QMainWindow, QWidget#root, QScrollArea, QScrollArea > QWidget > QWidget {
    background: #090d16;
}
QFrame#topbar {
    background: #0d1220;
    border-bottom: 1px solid #222d43;
}
QFrame#sidebar {
    background: #0c1220;
    border: 1px solid #202c42;
    border-radius: 14px;
}
QFrame#panel, QGroupBox#panel {
    background: #111927;
    border: 1px solid #273650;
    border-radius: 12px;
}
QGroupBox#panel {
    /* Keep the caption inside the card instead of splitting its border. */
    margin-top: 0px;
    padding: 28px 12px 12px 12px;
}
QGroupBox#panel::title {
    subcontrol-origin: padding;
    subcontrol-position: top left;
    left: 0px;
    top: 4px;
    padding: 0;
    background: transparent;
    color: #9baac5;
    font-size: 11px;
    font-weight: 600;
}
QLabel#title {
    font-size: 20px;
    font-weight: 700;
    color: #f4f5ff;
}
QLabel#subtitle, QLabel#muted {
    color: #8491aa;
    font-size: 11px;
}
QLabel#field {
    color: #b7c1d6;
    font-size: 11px;
}
QLabel#brandTitle {
    color: #f4f2ff;
    font-size: 27px;
    font-weight: 800;
}
QLabel#brandBy {
    color: #b9bed0;
    font-size: 21px;
}
QLabel#brandBy span {
    color: #8964ff;
}
QLabel#brandQuote {
    color: #9e8fc9;
    font-size: 11px;
    font-style: italic;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #151f31;
    border: 1px solid #30415e;
    border-radius: 7px;
    min-height: 29px;
    padding: 0 9px;
    color: #e9ecf7;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #8061ff;
}
/* Numeric fields are direct-entry controls: click the value, type, Enter.
   Hiding steppers keeps the compact rows clean and prevents accidental clicks. */
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 0px;
    height: 0px;
    border: none;
    image: none;
}
QComboBox::drop-down {
    width: 28px;
    border: 0;
}
QComboBox QAbstractItemView {
    background: #151f31;
    border: 1px solid #435579;
    selection-background-color: #5e43c8;
}
QPushButton, QToolButton {
    background: #18243a;
    border: 1px solid #354967;
    border-radius: 7px;
    padding: 7px 11px;
    color: #e8ebf7;
}
QPushButton:hover, QToolButton:hover {
    background: #23314c;
    border-color: #8768ff;
}
QPushButton:pressed, QToolButton:pressed {
    background: #342566;
}
QPushButton#primary {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #6745ee, stop:1 #8a5dff);
    border: 1px solid #a38aff;
    font-weight: 700;
}
QPushButton#primary:hover {
    background: #7c58ff;
}
QPushButton#stop_button {
    background: #19253a;
}
/* Sidebar in the reference is a calm list, not a stack of outlined buttons. */
QPushButton#home_button, QPushButton#model_button, QPushButton#profiles_button,
QPushButton#hotkeys_button, QPushButton#mixer_button, QPushButton#effects_button,
QPushButton#logs_button, QPushButton#settings_button, QPushButton#about_button {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    color: #b7c2dc;
    text-align: left;
    padding: 0 12px;
    font-size: 11px;
}
QPushButton#home_button:hover, QPushButton#model_button:hover, QPushButton#profiles_button:hover,
QPushButton#hotkeys_button:hover, QPushButton#mixer_button:hover, QPushButton#effects_button:hover,
QPushButton#logs_button:hover, QPushButton#settings_button:hover, QPushButton#about_button:hover {
    background: #151d31;
    color: #f3f2ff;
}
QPushButton#home_button:checked, QPushButton#model_button:checked, QPushButton#profiles_button:checked,
QPushButton#hotkeys_button:checked, QPushButton#mixer_button:checked, QPushButton#effects_button:checked,
QPushButton#logs_button:checked, QPushButton#settings_button:checked, QPushButton#about_button:checked {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #6f4ff1,stop:1 #3e2b8a);
    border: 1px solid #785cff;
    color: white;
    font-weight: 700;
}
QGroupBox#hero_card {
    background: #101426;
    border: 1px solid #6d48dd;
    border-radius: 12px;
}
QGroupBox#hero_card::title { background: #101426; }
QCheckBox#monitor_toggle::indicator, QCheckBox#convert_toggle::indicator {
    width: 30px;
    height: 16px;
    border-radius: 8px;
    background: #35425b;
    border: 0;
}
QCheckBox#monitor_toggle::indicator:checked, QCheckBox#convert_toggle::indicator:checked {
    background: #7456f8;
}
QPushButton#reset_button {
    background: #151f31;
}
QPushButton#sidebarButton {
    text-align: left;
    padding: 10px 12px;
    border: 1px solid transparent;
    background: transparent;
    color: #9eabc2;
}
QPushButton#sidebarButton:hover {
    background: #17223a;
    border-color: #30415e;
    color: #f0f1ff;
}
QPushButton#sidebarButton:checked {
    background: #30235e;
    border-color: #7054d8;
    color: #ffffff;
}
QPushButton#linkButton {
    background: #151e31;
    border: 1px solid #34445f;
    text-align: left;
}
QCheckBox, QRadioButton {
    spacing: 7px;
    color: #b9c3d8;
    font-size: 11px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 13px;
    height: 13px;
}
QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked {
    background: #101827;
    border: 1px solid #52627e;
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background: #7657ff;
    border: 1px solid #a18aff;
}
QRadioButton::indicator {
    border-radius: 7px;
}
QRadioButton::indicator:checked {
    background: #7657ff;
    border: 3px solid #101827;
}
QSlider::groove:horizontal {
    height: 5px;
    background: #34435e;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #7657ff;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    margin: -5px 0;
    background: #edf0fa;
    border: 2px solid #927aff;
    border-radius: 8px;
}
QListWidget {
    background: #101827;
    border: 1px solid #293952;
    border-radius: 8px;
    outline: none;
}
QListWidget::item {
    padding: 8px;
    border-bottom: 1px solid #202d43;
}
QListWidget::item:selected {
    background: #29214c;
    border-left: 2px solid #8a6aff;
}
QScrollBar:vertical {
    background: #0c1220;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #34435d;
    border-radius: 5px;
    min-height: 30px;
}
QToolTip {
    background: #1b263b;
    color: white;
    border: 1px solid #7657ff;
    padding: 5px;
}
"""


class DropZone(QFrame):
    filesDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("drop_zone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(112)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet("""
            QFrame#drop_zone {
                background: #101827;
                border: 1px dashed #5b6d8d;
                border-radius: 9px;
            }
            QFrame#drop_zone:hover {
                background: #151e31;
                border-color: #8c6dff;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(2)
        for text, style in (
            ("⇧", "font-size:30px;color:#8969ff;font-weight:800;"),
            ("Перетащите файлы модели сюда", "font-size:13px;font-weight:700;"),
            ("или выберите файлы вручную", "color:#8e9ab0;font-size:11px;"),
            ("Поддерживаемые форматы: .pth, .index", "color:#66758e;font-size:10px;"),
        ):
            label = QLabel(text)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(style)
            layout.addWidget(label)

    def dragEnterEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        valid = any(
            Path(url.toLocalFile()).suffix.lower() in {".pth", ".index"}
            for url in event.mimeData().urls()
        )
        event.acceptProposedAction() if valid else event.ignore()

    def dropEvent(self, event):
        files = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if Path(url.toLocalFile()).suffix.lower() in {".pth", ".index"}
        ]
        if files:
            self.filesDropped.emit(files)
            event.acceptProposedAction()
        else:
            event.ignore()


class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("input_waveform")
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.samples = [0.0] * 120
        self.level = 0.0
        self.running = False
        self.external_audio = False
        self._phase = 0.0
        self._source = None
        self._device = None
        self._audio_buffer = bytearray()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._fallback_tick)

    def set_running(self, enabled):
        self.running = bool(enabled)
        if self.running or self.external_audio:
            if not self._timer.isActive():
                self._timer.start(80)
        else:
            self._timer.stop()
        self.update()

    def set_audio_level(self, level):
        self.level = max(0.0, min(1.0, float(level)))

    def feed_samples(self, samples):
        values = list(samples)[-120:]
        if values:
            self.external_audio = True
            if not self._timer.isActive():
                self._timer.start(80)
            self.samples = (self.samples + values)[-120:]
            self.level = min(1.0, sum(abs(x) for x in values) / len(values) * 2.0)
            self.update()

    def attach_audio_source(self, source, device=None):
        self.stop_audio_source()
        self._source = source
        self._device = device
        if source is not None:
            source.readyRead.connect(self._read_audio)
            source.start()

    def stop_audio_source(self):
        if self._source is not None:
            try:
                self._source.stop()
            except RuntimeError:
                pass
        self._source = None

    def _read_audio(self):
        if self._source is None:
            return
        try:
            data = self._source.bytesAvailable()
            raw = self._source.read(data)
            if raw:
                self._audio_buffer.extend(bytes(raw))
                if len(self._audio_buffer) > 4096:
                    chunk = self._audio_buffer[-4096:]
                    self._audio_buffer.clear()
                    samples = []
                    for i in range(0, len(chunk) - 1, 2):
                        value = int.from_bytes(chunk[i:i + 2], "little", signed=True) / 32768.0
                        samples.append(value)
                    self.feed_samples(samples)
        except RuntimeError:
            pass

    def _fallback_tick(self):
        if self.external_audio:
            self.update()
            return
        if not self.running:
            # Do not animate a fabricated waveform before the callback has
            # supplied microphone data. A moving graph here falsely suggests
            # that the audio route is already active.
            return
        self._phase += 0.18
        # Visual fallback only until an external audio engine feeds real samples.
        value = (
            0.18 * math.sin(self._phase * 1.3)
            + 0.10 * math.sin(self._phase * 0.47)
        ) * (0.35 + self.level)
        self.samples = (self.samples + [value])[-120:]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, QColor("#0d1420"))

        center = rect.height() * .46
        usable_right = rect.width() - 48
        painter.setPen(QPen(QColor("#1c293d"), 1))
        for fraction in (.22, .46, .70):
            painter.drawLine(12, int(rect.height() * fraction), usable_right, int(rect.height() * fraction))
        count = min(72, len(self.samples))
        step = max(4.0, (usable_right - 24) / max(1, count - 1))
        amp = min(56.0, rect.height() * .34)
        for i in range(count):
            source = int(i * (len(self.samples) - 1) / max(1, count - 1))
            value = self.samples[source]
            height = max(3.0, abs(value) * amp * 2.0)
            x = 14 + i * step
            gradient = QLinearGradient(x, center - height / 2, x, center + height / 2)
            gradient.setColorAt(0, QColor("#B394FF"))
            gradient.setColorAt(1, QColor("#6E50F3"))
            painter.setPen(QPen(gradient, 2.2, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(x, center - height / 2), QPointF(x, center + height / 2))

        painter.setPen(QColor("#9ca9c0"))
        painter.setFont(QFont("Segoe UI", 9))
        level_text = "Громкость: —" if not self.external_audio else f"Громкость: {20 * math.log10(max(self.level, 1e-5)):.0f} dB"
        painter.drawText(14, rect.height() - 12, level_text)
        painter.drawText(max(14, rect.width() - 258), rect.height() - 12, "Частота: 44100 Hz")
        painter.drawText(max(14, rect.width() - 142), rect.height() - 12, "Каналы: Стерео")
        painter.setPen(QColor("#64728c"))
        for value, y in (("0", .22), ("-20", .38), ("-40", .54), ("-60", .70)):
            painter.drawText(rect.width() - 37, int(rect.height() * y), value)


class SliderRow(QWidget):
    valueChanged = Signal(float)

    def __init__(self, label, minimum, maximum, value, decimals=0, parent=None):
        super().__init__(parent)
        self.scale = 10 ** decimals
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(round(minimum * self.scale), round(maximum * self.scale))
        self.slider.setValue(round(value * self.scale))
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(minimum, maximum)
        self.spin.setValue(value)
        self.spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spin.setFixedWidth(78)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(10)
        layout.setColumnStretch(1, 1)
        name = QLabel(label)
        name.setObjectName("field")
        layout.addWidget(name, 0, 0)
        layout.addWidget(self.slider, 0, 1)
        layout.addWidget(self.spin, 0, 2)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _from_slider(self, value):
        number = value / self.scale
        self.spin.blockSignals(True)
        self.spin.setValue(number)
        self.spin.blockSignals(False)
        self.valueChanged.emit(number)

    def _from_spin(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(round(value * self.scale))
        self.slider.blockSignals(False)
        self.valueChanged.emit(float(value))


class SidebarButton(QPushButton):
    def __init__(self, text, object_name, parent=None):
        super().__init__(text, parent)
        self.setObjectName(object_name)
        self.setCheckable(True)
        self.setMinimumHeight(38)
        self.setCursor(Qt.PointingHandCursor)


class ToggleCheckBox(QCheckBox):
    """Reference-style switch with the same checked API as QCheckBox."""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(24)
        # The label is drawn manually, so its width must be based on the
        # actual text rather than an arbitrary small constant.  This prevents
        # the final words from disappearing in the fixed bottom toolbar.
        label_width = QFontMetrics(QFont("Segoe UI", 11)).horizontalAdvance(text)
        self.setMinimumWidth(label_width + 46)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        toggle = QRectF(1, (self.height() - 16) / 2, 30, 16)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#7657ff") if self.isChecked() else QColor("#36435b"))
        painter.drawRoundedRect(toggle, 8, 8)
        painter.setBrush(QColor("#f2f3ff"))
        x = toggle.right() - 14 if self.isChecked() else toggle.left() + 2
        painter.drawEllipse(QRectF(x, toggle.top() + 2, 12, 12))
        painter.setPen(QColor("#e9ecf7"))
        painter.setFont(QFont("Segoe UI", 11))
        painter.drawText(QRectF(39, 0, self.width() - 39, self.height()), Qt.AlignVCenter, self.text())


class VoiceBrandHeader(QWidget):
    """Compact, self-painted Voice identity used in the existing module header."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(390, 54)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        card = QRectF(1, 1, self.width() - 2, self.height() - 2)

        # Navy surface, violet-to-blue edge and a restrained inner glow.
        surface = QLinearGradient(card.left(), card.top(), card.right(), card.bottom())
        surface.setColorAt(0.0, QColor("#15123b"))
        surface.setColorAt(0.52, QColor("#0c1427"))
        surface.setColorAt(1.0, QColor("#111d35"))
        painter.setPen(Qt.NoPen)
        painter.setBrush(surface)
        painter.drawRoundedRect(card, 13, 13)
        edge = QLinearGradient(card.left(), card.top(), card.right(), card.bottom())
        edge.setColorAt(0.0, QColor("#8765ff"))
        edge.setColorAt(0.48, QColor("#5966ff"))
        edge.setColorAt(1.0, QColor("#28568d"))
        painter.setPen(QPen(QBrush(edge), 1.4))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(card.adjusted(0.7, 0.7, -0.7, -0.7), 12, 12)

        icon = QRectF(10, 8, 38, 38)
        glow = QLinearGradient(icon.left(), icon.top(), icon.right(), icon.bottom())
        glow.setColorAt(0, QColor("#5b34e8"))
        glow.setColorAt(1, QColor("#1673d6"))
        painter.setPen(QPen(QColor("#9d7dff"), 1))
        painter.setBrush(glow)
        painter.drawRoundedRect(icon, 10, 10)
        painter.setPen(Qt.NoPen)
        # The five soft waveform bars mirror the reference icon.
        for x, height, color in ((17, 13, "#d9c5ff"), (23, 23, "#a983ff"), (29, 30, "#f0d7ff"), (35, 23, "#8e75ff"), (41, 13, "#75b4ff")):
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(QRectF(x, 27 - height / 2, 4, height), 2, 2)

        text_x = 62
        title_font = QFont("Segoe UI", 17, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor("#f8f8ff"))
        painter.drawText(text_x, 27, "Deep Voice ")
        live_x = text_x + painter.fontMetrics().horizontalAdvance("Deep Voice ")
        live_gradient = QLinearGradient(live_x, 0, live_x + 38, 0)
        live_gradient.setColorAt(0, QColor("#d0a4ff"))
        live_gradient.setColorAt(1, QColor("#7653ff"))
        painter.setPen(QPen(QBrush(live_gradient), 1))
        painter.drawText(live_x, 27, "Live")
        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor("#9cabc8"))
        painter.drawText(text_x, 42, "Real-time AI Voice Changer")


class VoiceHeroCard(QGroupBox):
    """The existing voice overview card, with its artwork painted as a backdrop."""

    def __init__(self, artwork: Path, parent=None):
        super().__init__("", parent)
        self.setObjectName("hero_card")
        self._artwork = QPixmap(str(artwork))

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        card = QRectF(1, 1, self.width() - 2, self.height() - 2)
        clip = QPainterPath()
        clip.addRoundedRect(card, 12, 12)
        painter.save()
        painter.setClipPath(clip)
        painter.fillRect(self.rect(), QColor("#101728"))
        if not self._artwork.isNull():
            # Cover the panel without distorting the portrait.  It is aligned
            # to the right, so the information column remains legible.
            scaled = self._artwork.scaled(
                self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            painter.drawPixmap(self.width() - scaled.width(), (self.height() - scaled.height()) // 2, scaled)
        shade = QLinearGradient(0, 0, self.width(), 0)
        shade.setColorAt(0.0, QColor(10, 16, 32, 244))
        shade.setColorAt(0.42, QColor(10, 16, 32, 220))
        shade.setColorAt(0.72, QColor(10, 16, 32, 110))
        shade.setColorAt(1.0, QColor(10, 16, 32, 38))
        painter.fillRect(self.rect(), shade)
        painter.restore()
        border = QLinearGradient(card.left(), card.top(), card.right(), card.bottom())
        border.setColorAt(0.0, QColor("#8c66ff"))
        border.setColorAt(0.55, QColor("#5b59df"))
        border.setColorAt(1.0, QColor("#31558c"))
        painter.setPen(QPen(QBrush(border), 1.2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(card.adjusted(0.6, 0.6, -0.6, -0.6), 12, 12)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Voice Live")
        self.resize(1500, 930)
        self.setMinimumSize(1120, 700)
        self.setStyleSheet(APP_QSS)
        self._conversion_active = False
        self._audio_source = None
        self._last_cpu_time = None
        self._last_cpu_wall = None
        self._build_ui()
        self._connect_external_signals()
        self._update_metrics()

    def _connect_external_signals(self):
        # Public widgets and signals are intentionally exposed for the RVC/audio engine.
        self.start_button.clicked.connect(self.start_conversion)
        self.stop_button.clicked.connect(self.stop_conversion)
        self.reset_button.clicked.connect(self.reset_settings)
        self.mic_test_button.clicked.connect(self.test_microphone)
        self.drop_zone.filesDropped.connect(self._handle_dropped_files)
        self.input_device.currentIndexChanged.connect(self._device_changed)
        self.output_device.currentIndexChanged.connect(self._device_changed)

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(8)
        outer.addWidget(self._build_topbar())

        # Deep Live Studio already supplies the application-level navigation.
        # A second nine-item sidebar inside the Voice page made the actual RVC
        # workspace needlessly narrow.  Voice is now one full-width Main page.
        outer.addWidget(self._build_content(), 1)

        bottom = self._build_statusbar()
        outer.addWidget(bottom)

    def _build_topbar(self):
        bar = QFrame()
        bar.setObjectName("topbar")
        bar.setFixedHeight(66)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(VoiceBrandHeader())
        layout.addStretch()

        self.ready_label = QLabel("●  Готов к работе")
        self.ready_label.setStyleSheet("color:#80eab5;font-size:11px;")
        layout.addWidget(self.ready_label)
        layout.addSpacing(16)

        self.cpu_label = QLabel("CPU 0%")
        self.ram_label = QLabel("RAM 0 MB")
        for label in (self.cpu_label, self.ram_label):
            label.setStyleSheet("color:#aeb8cd;font-size:11px;")
            layout.addWidget(label)
        layout.addSpacing(8)

        for text, slot in (("—", self.showMinimized), ("□", self._toggle_maximized), ("×", self.close)):
            button = QToolButton()
            button.setText(text)
            button.setFixedSize(34, 30)
            button.clicked.connect(slot)
            layout.addWidget(button)
        return bar

    def _build_content(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(2, 2, 2, 2)
        content_layout.setSpacing(10)

        page_title = QLabel("⌂  Главная")
        page_title.setObjectName("title")
        content_layout.addWidget(page_title)

        top = QGridLayout()
        top.setHorizontalSpacing(10)
        top.setVerticalSpacing(10)
        top.setColumnStretch(0, 3)
        top.setColumnStretch(1, 2)
        top.addWidget(self._build_model_card(), 0, 0)
        top.addWidget(self._build_brand_card(), 0, 1)
        content_layout.addLayout(top)

        middle = QGridLayout()
        middle.setHorizontalSpacing(10)
        middle.setVerticalSpacing(10)
        middle.setColumnStretch(0, 3)
        middle.setColumnStretch(1, 1)
        middle.setColumnStretch(2, 3)
        middle.addWidget(self._build_audio_card(), 0, 0)
        middle.addWidget(self._build_mic_test(), 0, 1)
        middle.addWidget(self._build_signal_card(), 0, 2)
        content_layout.addLayout(middle)

        lower = QGridLayout()
        lower.setHorizontalSpacing(10)
        lower.setVerticalSpacing(10)
        lower.setColumnStretch(0, 1)
        lower.setColumnStretch(1, 1)
        lower.addWidget(self._build_main_settings(), 0, 0)
        lower.addWidget(self._build_quick_settings(), 0, 1)
        content_layout.addLayout(lower)
        content_layout.addStretch()

        scroll.setWidget(content)
        return scroll

    def _panel(self, title):
        group = QGroupBox(title)
        group.setObjectName("panel")
        return group

    def _build_model_card(self):
        card = self._panel("Загрузка модели")
        layout = QVBoxLayout(card)
        layout.setSpacing(8)

        self.drop_zone = DropZone()
        layout.addWidget(self.drop_zone)

        self.pth_path = QLineEdit()
        self.pth_path.setObjectName("pth_path")
        self.pth_path.setPlaceholderText("Путь к файлу модели .pth")
        self.pth_path.setClearButtonEnabled(True)
        pth_button = QPushButton("Выбрать файл .pth")
        pth_button.clicked.connect(lambda: self._choose_model_file("pth"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Файл модели (.pth)"), 0)
        row.addWidget(self.pth_path, 1)
        row.addWidget(pth_button, 0)
        layout.addLayout(row)

        self.index_path = QLineEdit()
        self.index_path.setObjectName("index_path")
        self.index_path.setPlaceholderText("Путь к файлу индекса .index")
        self.index_path.setClearButtonEnabled(True)
        index_button = QPushButton("Выбрать файл .index")
        index_button.clicked.connect(lambda: self._choose_model_file("index"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Файл индекса (.index)"), 0)
        row.addWidget(self.index_path, 1)
        row.addWidget(index_button, 0)
        layout.addLayout(row)

        return card

    def _build_brand_card(self):
        art_path = Path(__file__).parent / "assets" / "ui" / "hero-syrex.png"
        card = VoiceHeroCard(art_path)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 16)
        top = QHBoxLayout()
        copy = QVBoxLayout()
        copy.setSpacing(2)
        title = QLabel("Deep Voice Live")
        title.setObjectName("brandTitle")
        by = QLabel("by <span>Syrex</span>")
        by.setObjectName("brandBy")
        by.setTextFormat(Qt.RichText)
        description = QLabel("Профессиональный RVC\nголосовой конвертер\nв реальном времени")
        description.setObjectName("muted")
        description.setStyleSheet("font-size:12px;color:#a2abc0;")
        quote = QLabel("“Ваш голос. Больше возможностей.”")
        quote.setObjectName("brandQuote")
        copy.addWidget(title)
        copy.addWidget(by)
        copy.addSpacing(8)
        copy.addWidget(description)
        copy.addWidget(quote)
        copy.addStretch(1)
        top.addLayout(copy, 1)
        top.addStretch(1)
        layout.addLayout(top, 1)

        capabilities = QHBoxLayout()
        capabilities.setSpacing(5)
        for text in ("◉\nRVC / AI", "◷\nReal-time", "◈\nВысокое\nкачество", "◌\nНизкая\nзадержка", "⚙\nПростая\nнастройка"):
            label = QLabel(text)
            label.setAlignment(Qt.AlignCenter)
            label.setWordWrap(True)
            label.setStyleSheet(
                "background:transparent;border:0;"
                "padding:2px;color:#d4c8ff;font-size:9px;"
            )
            capabilities.addWidget(label, 1)
        layout.addLayout(capabilities)

        return card

    def _build_audio_card(self):
        card = self._panel("Аудиоустройство")
        layout = QGridLayout(card)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(7)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        layout.addWidget(QLabel("Тип устройства"), 0, 0)
        self.host_api = QComboBox()
        self.host_api.setObjectName("host_api")
        self.host_api.addItems(["MME", "WASAPI", "DirectSound", "ASIO"])
        layout.addWidget(self.host_api, 0, 1, 1, 2)

        layout.addWidget(QLabel("Входное устройство"), 1, 0)
        self.input_device = QComboBox()
        self.input_device.setObjectName("input_device")
        self.input_device.addItems(["Микрофон (Logitech G735 Gaming)", "Default Microphone", "USB Microphone"])
        layout.addWidget(self.input_device, 1, 1, 1, 2)

        layout.addWidget(QLabel("Выходное устройство"), 2, 0)
        self.output_device = QComboBox()
        self.output_device.setObjectName("output_device")
        self.output_device.addItems(["Динамики (Logitech G735 Gaming)", "Default Speakers", "Headphones"])
        layout.addWidget(self.output_device, 2, 1, 1, 2)

        self.wasapi_exclusive = QCheckBox("Эксклюзивный WASAPI")
        layout.addWidget(self.wasapi_exclusive, 3, 0)
        refresh = QPushButton("⟳")
        refresh.setFixedWidth(36)
        refresh.clicked.connect(self.refresh_devices)
        layout.addWidget(refresh, 3, 1)
        self.model_rate = QRadioButton("Частота модели")
        self.device_rate = QRadioButton("Частота устройства")
        self.device_rate.setChecked(True)
        layout.addWidget(self.model_rate, 3, 2)
        layout.addWidget(self.device_rate, 4, 2)

        return card

    def _build_signal_card(self):
        card = self._panel("Живой входной сигнал")
        layout = QVBoxLayout(card)
        head = QHBoxLayout()
        led = QLabel("●")
        led.setStyleSheet("color:#55e5a2;font-size:16px;")
        head.addWidget(led)
        head.addWidget(QLabel("Входной сигнал"))
        head.addStretch()
        self.signal_db = QLabel("— dB")
        self.signal_db.setObjectName("muted")
        head.addWidget(self.signal_db)
        layout.addLayout(head)

        self.input_waveform = WaveformWidget()
        layout.addWidget(self.input_waveform, 1)
        return card

    def _build_mic_test(self):
        card = self._panel("Тест микрофона")
        layout = QVBoxLayout(card)
        layout.setSpacing(8)
        self.mic_meter = QProgressLike()
        layout.addWidget(self.mic_meter)
        self.mic_test_button = QPushButton("▶  Проверить звук")
        self.mic_test_button.setObjectName("mic_test_button")
        layout.addWidget(self.mic_test_button)
        return card

    def _build_main_settings(self):
        card = self._panel("Основные настройки")
        layout = QVBoxLayout(card)
        layout.setSpacing(8)
        for item in (
            ("Порог ответа", -100, 0, -60, 0),
            ("Настройка высоты звука", -12, 12, -3, 0),
            ("Гендерный коэффициент / толщина голоса", -1, 1, 0, 2),
            ("Темп индекса", 0, 2, 0.70, 2),
            ("Коэффициент громкости", 0, 2, 0.20, 2),
        ):
            layout.addWidget(SliderRow(*item))
        row = QHBoxLayout()
        row.addWidget(QLabel("Алгоритм высоты тона"))
        self.pitch_group = QButtonGroup(self)
        for text, checked in (("PM", True), ("RMVPE", False), ("FCPE", False)):
            radio = QRadioButton(text)
            radio.setChecked(checked)
            self.pitch_group.addButton(radio)
            row.addWidget(radio)
        row.addStretch()
        layout.addLayout(row)
        return card

    def _build_quick_settings(self):
        card = self._panel("Настройки быстроты")
        layout = QVBoxLayout(card)
        layout.setSpacing(8)
        for item in (
            ("Длина сэмпла", 0.01, 1, 0.13, 2),
            ("Длина затухания", 0.01, 1, 0.08, 2),
            ("Доп. время обработки", 0, 10, 2.00, 2),
        ):
            layout.addWidget(SliderRow(*item))
        row = QHBoxLayout()
        self.input_noise = QCheckBox("Уменьшение входного шума")
        self.output_noise = QCheckBox("Уменьшение выходного шума")
        row.addWidget(self.input_noise)
        row.addWidget(self.output_noise)
        row.addStretch()
        layout.addLayout(row)
        note = QLabel("ⓘ  Шумоподавление включайте только при заметном шуме микрофона.")
        note.setWordWrap(True)
        note.setStyleSheet(
            "background:#17233a;border:1px solid #293c5e;border-radius:7px;"
            "padding:9px;color:#9ba9c3;font-size:10px;"
        )
        layout.addWidget(note)
        return card

    def _build_statusbar(self):
        bar = QFrame()
        bar.setObjectName("panel")
        # A one-line toolbar works only when the hosted module has the whole
        # desktop width.  Two calm rows keep every label visible in Studio's
        # normal window as well as in a maximized window.
        bar.setFixedHeight(82)
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(4)
        controls = QHBoxLayout()
        controls.setSpacing(8)
        performance = QHBoxLayout()
        performance.setSpacing(8)

        self.start_button = QPushButton("▶  Начать конвертацию аудио")
        self.start_button.setObjectName("start_button")
        self.start_button.setProperty("primary", True)
        self.start_button.setStyleSheet(
            "QPushButton#start_button{background:#704cff;border:1px solid #a18aff;"
            "border-radius:7px;padding:8px 13px;font-weight:700;}"
            "QPushButton#start_button:hover{background:#805eff;}"
        )
        self.stop_button = QPushButton("■  Остановить")
        self.stop_button.setObjectName("stop_button")
        self.stop_button.setEnabled(False)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)

        self.monitor_toggle = ToggleCheckBox("Мониторинг входа")
        self.monitor_toggle.setObjectName("monitor_toggle")
        self.monitor_toggle.setChecked(True)
        self.convert_toggle = ToggleCheckBox("Преобразование выхода")
        self.convert_toggle.setObjectName("convert_toggle")
        self.convert_toggle.setChecked(True)
        controls.addWidget(self.monitor_toggle)
        controls.addWidget(self.convert_toggle)
        controls.addStretch(1)
        layout.addLayout(controls)

        performance.addStretch(1)
        performance.addWidget(QLabel("Задержка алгоритма (мс)"))
        self.latency_spin = QSpinBox()
        self.latency_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.latency_spin.setRange(0, 5000)
        self.latency_spin.setValue(480)
        self.latency_spin.setFixedWidth(78)
        performance.addWidget(self.latency_spin)

        performance.addWidget(QLabel("Время обработки (мс)"))
        self.processing_spin = QSpinBox()
        self.processing_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.processing_spin.setRange(0, 5000)
        self.processing_spin.setValue(24)
        self.processing_spin.setFixedWidth(78)
        performance.addWidget(self.processing_spin)

        self.reset_button = QPushButton("⟳  Сбросить настройки")
        self.reset_button.setObjectName("reset_button")
        performance.addWidget(self.reset_button)
        layout.addLayout(performance)
        return bar

    def _choose_model_file(self, extension):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл", "", f"Файлы (*.{extension})"
        )
        if path:
            self._set_model_path(extension, path)

    def _set_model_path(self, extension, path):
        if extension == "pth":
            self.pth_path.setText(path)
        elif extension == "index":
            self.index_path.setText(path)
        self.model_path_changed.emit(extension, path)

    def _handle_dropped_files(self, files):
        for file_path in files:
            extension = Path(file_path).suffix.lower().lstrip(".")
            if extension in {"pth", "index"}:
                self._set_model_path(extension, file_path)

    def refresh_devices(self):
        current_in = self.input_device.currentText()
        current_out = self.output_device.currentText()
        self.input_device.clear()
        self.output_device.clear()
        self.input_device.addItems([current_in, "Default Microphone", "USB Microphone"])
        self.output_device.addItems([current_out, "Default Speakers", "Headphones"])
        self.devices_refreshed.emit()

    def _device_changed(self):
        self.device_changed.emit(
            self.input_device.currentText(),
            self.output_device.currentText(),
            self.host_api.currentText()
        )

    def start_conversion(self):
        self._conversion_active = True
        self.input_waveform.set_running(True)
        self.ready_label.setText("●  Конвертация активна")
        self.ready_label.setStyleSheet("color:#c0a8ff;font-size:11px;")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.conversion_started.emit()

    def stop_conversion(self):
        self._conversion_active = False
        self.input_waveform.set_running(False)
        self.ready_label.setText("●  Готов к работе")
        self.ready_label.setStyleSheet("color:#80eab5;font-size:11px;")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.conversion_stopped.emit()

    def test_microphone(self):
        self.mic_test_button.setText("■  Остановить тест")
        self.mic_test_button.setProperty("testing", True)
        self.mic_test_button.style().unpolish(self.mic_test_button)
        self.mic_test_button.style().polish(self.mic_test_button)
        self.input_waveform.set_running(True)
        self.microphone_test_requested.emit()

    def reset_settings(self):
        self.latency_spin.setValue(480)
        self.processing_spin.setValue(24)
        self.input_noise.setChecked(False)
        self.output_noise.setChecked(False)
        self.monitor_toggle.setChecked(True)
        self.convert_toggle.setChecked(True)
        self.settings_reset.emit()

    def _toggle_maximized(self):
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def _update_metrics(self):
        cpu_percent, ram_mb = self._process_metrics()
        self.cpu_label.setText(f"CPU {cpu_percent:.0f}%")
        self.ram_label.setText(f"RAM {ram_mb} MB")
        QTimer.singleShot(1500, self._update_metrics)

    def _process_metrics(self):
        """Return actual CPU load and working memory of this Windows process."""
        if sys.platform != "win32":
            return 0.0, 0
        try:
            class FILETIME(ctypes.Structure):
                _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            creation, exit_time, kernel, user = FILETIME(), FILETIME(), FILETIME(), FILETIME()
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel), ctypes.byref(user))
            cpu_ticks = (kernel.dwHighDateTime << 32) + kernel.dwLowDateTime + (user.dwHighDateTime << 32) + user.dwLowDateTime
            wall = time.perf_counter()
            cpu_percent = 0.0
            if self._last_cpu_time is not None and wall > self._last_cpu_wall:
                cpu_percent = max(0.0, min(100.0, (cpu_ticks - self._last_cpu_time) / 10_000_000 / (wall - self._last_cpu_wall) / max(1, os.cpu_count() or 1) * 100))
            self._last_cpu_time, self._last_cpu_wall = cpu_ticks, wall

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            get_memory_info = ctypes.windll.kernel32.K32GetProcessMemoryInfo
            get_memory_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
            get_memory_info.restype = ctypes.c_int
            get_memory_info(ctypes.c_void_p(-1), ctypes.byref(counters), counters.cb)
            return cpu_percent, int(counters.WorkingSetSize / 1024 / 1024)
        except Exception:
            return 0.0, 0

    model_path_changed = Signal(str, str)
    device_changed = Signal(str, str, str)
    devices_refreshed = Signal()
    conversion_started = Signal()
    conversion_stopped = Signal()
    microphone_test_requested = Signal()
    settings_reset = Signal()
    external_link_requested = Signal(str)


class QProgressLike(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(25)
        self.value = 0.55
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    def showEvent(self, event):
        if not self.timer.isActive():
            self.timer.start(250)
        super().showEvent(event)

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def _tick(self):
        self.value = 0.35 + 0.25 * (0.5 + 0.5 * math.sin(time.monotonic() * 2.3))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(2, 4, -2, -4)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#0c1422"))
        p.drawRoundedRect(r, 6, 6)
        count = 28
        gap = 3
        width = max(2, (r.width() - gap * (count - 1)) / count)
        active = int(count * self.value)
        for i in range(count):
            x = r.x() + i * (width + gap)
            color = QColor("#805dff" if i < active else "#26334b")
            p.setBrush(color)
            p.drawRoundedRect(QRectF(x, r.y() + 4, width, r.height() - 8), 2, 2)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Deep Voice Live")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
