"""
Widget visual del Afinador Musical (Módulo 2A).
"""

from collections import deque
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.analysis.tuner import INSTRUMENT_PRESETS, analyze_pitch


class CentGaugeWidget(QWidget):
    """Medidor horizontal de alta precisión de -50 a +50 cents."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(82)
        self._cents: float | None = None

    def set_cents(self, cents: float | None) -> None:
        self._cents = cents
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        margin = 36.0
        track_w = max(10.0, w - 2 * margin)
        center_y = h * 0.52

        def x_for_cents(c: float) -> float:
            clamped = max(-50.0, min(50.0, c))
            return margin + ((clamped + 50.0) / 100.0) * track_w

        # Barra base
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.drawRoundedRect(QRectF(margin, center_y - 7, track_w, 14), 7, 7)

        # Zona verde objetivo (±5 cents)
        x_left = x_for_cents(-5.0)
        x_right = x_for_cents(5.0)
        painter.setBrush(QBrush(QColor(52, 211, 153, 85)))
        painter.drawRoundedRect(QRectF(x_left, center_y - 7, x_right - x_left, 14), 4, 4)

        # Marcas de graduación
        font = QFont("Noto Sans", 9)
        painter.setFont(font)
        for tick in (-50, -25, -10, 0, 10, 25, 50):
            tx = x_for_cents(tick)
            tick_h = 14 if tick == 0 else 8
            color = QColor("#34d399") if tick == 0 else QColor("#475569")
            painter.setPen(QPen(color, 2 if tick == 0 else 1))
            painter.drawLine(int(tx), int(center_y - tick_h), int(tx), int(center_y + tick_h))
            painter.setPen(QPen(QColor("#94a3b8")))
            label = "0" if tick == 0 else f"{tick:+d}"
            painter.drawText(QRectF(tx - 20, center_y + 16, 40, 18), Qt.AlignmentFlag.AlignCenter, label)

        # Aguja / indicador actual
        if self._cents is not None:
            nx = x_for_cents(self._cents)
            abs_c = abs(self._cents)
            if abs_c <= 5.0:
                needle_color = QColor("#34d399")
            elif abs_c <= 15.0:
                needle_color = QColor("#fb923c")
            else:
                needle_color = QColor("#f43f5e")

            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.setBrush(QBrush(needle_color))
            painter.drawEllipse(QRectF(nx - 9, center_y - 9, 18, 18))


class TunerWidget(QWidget):
    """Panel completo del afinador con lectura de nota, medidor e historial."""

    def __init__(self) -> None:
        super().__init__()
        self._smoothed_freq: float = 0.0
        self._history = deque([np.nan] * 120, maxlen=120)
        self._silent_frames = 0

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        # Barra superior de configuración del afinador
        top_bar = QHBoxLayout()
        top_bar.setSpacing(14)

        preset_lbl = QLabel("Instrumento:")
        preset_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.preset_combo = QComboBox()
        for name in INSTRUMENT_PRESETS:
            self.preset_combo.addItem(name)

        ref_lbl = QLabel("Calibración A4 (Hz):")
        ref_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.a4_spin = QSpinBox()
        self.a4_spin.setRange(430, 450)
        self.a4_spin.setValue(440)
        self.a4_spin.setStyleSheet(
            "background-color: #0f131b; border: 1px solid #2e3748; border-radius: 6px; padding: 4px 8px;"
        )

        top_bar.addWidget(preset_lbl)
        top_bar.addWidget(self.preset_combo)
        top_bar.addStretch()
        top_bar.addWidget(ref_lbl)
        top_bar.addWidget(self.a4_spin)
        layout.addLayout(top_bar)

        # Nota grande detectada
        self.note_label = QLabel("--")
        self.note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note_label.setStyleSheet("font-size: 64px; font-weight: 800; color: #64748b;")
        layout.addWidget(self.note_label)

        # Frecuencia y cuerda sugerida
        self.detail_label = QLabel("Toca una nota o silba para comenzar a afinar")
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_label.setStyleSheet("font-size: 14px; color: #94a3b8;")
        layout.addWidget(self.detail_label)

        # Indicador de dirección (Subir / Afinado / Bajar)
        self.direction_label = QLabel("En espera de señal")
        self.direction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.direction_label.setStyleSheet("font-size: 15px; font-weight: 700; color: #64748b;")
        layout.addWidget(self.direction_label)

        # Medidor de cents
        self.gauge = CentGaugeWidget()
        layout.addWidget(self.gauge)

        # Gráfica histórica de estabilidad en cents
        self.history_plot = pg.PlotWidget()
        self.history_plot.setBackground("#090b10")
        self.history_plot.setMinimumHeight(130)
        self.history_plot.setYRange(-50, 50, padding=0)
        self.history_plot.setXRange(0, 120, padding=0)
        self.history_plot.setMouseEnabled(x=False, y=False)
        self.history_plot.hideButtons()
        self.history_plot.showGrid(x=False, y=True, alpha=0.18)
        self.history_plot.showAxis("bottom", False)
        self.history_plot.setLabel("left", "Desviación (cents)", color="#94a3b8")

        target_band = pg.LinearRegionItem(
            values=(-5, 5),
            orientation="horizontal",
            movable=False,
            brush=pg.mkBrush(52, 211, 153, 35),
            pen=pg.mkPen(None),
        )
        self.history_plot.addItem(target_band)
        self.history_curve = self.history_plot.plot(pen=pg.mkPen("#22d3ee", width=2.2))
        layout.addWidget(self.history_plot, stretch=1)

    def update_reading(self, freq: float, confidence: float) -> float:
        """Actualiza el afinador con la frecuencia fundamental y devuelve la frecuencia suavizada."""
        if freq <= 0.0:
            self._silent_frames += 1
            if self._silent_frames > 12:
                self._smoothed_freq = 0.0
                self.note_label.setText("--")
                self.note_label.setStyleSheet("font-size: 64px; font-weight: 800; color: #64748b;")
                self.detail_label.setText("Sin señal suficiente · toca más cerca del micrófono")
                self.direction_label.setText("En espera de señal")
                self.direction_label.setStyleSheet("font-size: 15px; font-weight: 700; color: #64748b;")
                self.gauge.set_cents(None)
                self._history.append(np.nan)
                self.history_curve.setData(np.arange(len(self._history)), np.array(self._history))
            return 0.0

        self._silent_frames = 0

        # Suavizado exponencial inteligente: si cambia mucho la nota, salta directo; si no, estabiliza
        if self._smoothed_freq <= 0.0 or abs(freq - self._smoothed_freq) / self._smoothed_freq > 0.03:
            self._smoothed_freq = freq
        else:
            self._smoothed_freq = 0.65 * self._smoothed_freq + 0.35 * freq

        reading = analyze_pitch(
            freq=self._smoothed_freq,
            confidence=confidence,
            a4_ref=float(self.a4_spin.value()),
            preset=self.preset_combo.currentText(),
        )
        if reading is None:
            return 0.0

        self.note_label.setText(f"{reading.note_en}  ·  {reading.note_es}")
        string_info = f"  |  {reading.string_label}" if reading.string_label else ""
        self.detail_label.setText(
            f"Detectado: {reading.detected_freq:.1f} Hz   |   Objetivo: {reading.target_freq:.1f} Hz{string_info}"
        )

        if reading.in_tune:
            color = "#34d399"
            dir_text = f"● ¡Afinado! ({reading.cents:+.1f} cents)"
        elif reading.cents < 0:
            color = "#fb923c" if abs(reading.cents) <= 15 else "#f43f5e"
            dir_text = f"▲ Sube / Tensa ({reading.cents:+.1f} cents)"
        else:
            color = "#fb923c" if abs(reading.cents) <= 15 else "#f43f5e"
            dir_text = f"▼ Baja / Afloja ({reading.cents:+.1f} cents)"

        self.note_label.setStyleSheet(f"font-size: 56px; font-weight: 800; color: {color};")
        self.direction_label.setText(dir_text)
        self.direction_label.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {color};")

        self.gauge.set_cents(reading.cents)
        self._history.append(float(np.clip(reading.cents, -50.0, 50.0)))
        self.history_curve.setData(np.arange(len(self._history)), np.array(self._history))

        return self._smoothed_freq