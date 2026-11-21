"""
Interfaz gráfica del Módulo 1: Visualizador de Audio y Espectro FFT en tiempo real.
"""

import time
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.audio.capture import AudioStreamCapture, list_input_devices
from src.audio.spectral import analyze_frame, compute_log_bands
from src.config import MAX_FREQ, MIN_FREQ, SAMPLE_RATE


THEMES = [
    {"name": "Neón", "primary": "#22d3ee", "secondary": "#f43f5e", "highlight": "#ffffff"},
    {"name": "Ámbar", "primary": "#fb923c", "secondary": "#facc15", "highlight": "#fff7ed"},
    {"name": "Azul", "primary": "#38bdf8", "secondary": "#818cf8", "highlight": "#e0f2fe"},
    {"name": "Esmeralda", "primary": "#34d399", "secondary": "#2dd4bf", "highlight": "#ecfdf5"},
    {"name": "Violeta", "primary": "#c084fc", "secondary": "#f472b6", "highlight": "#faf5ff"},
]


class VisualizerWindow(QWidget):
    """Ventana principal del visualizador de audio y análisis de Fourier."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Proyecto Musical Fourier · Módulo 1: Adquisición y FFT")
        self.resize(1120, 720)

        pg.setConfigOptions(antialias=True)

        self.capture = AudioStreamCapture()
        self.is_running = False
        self.prev_magnitudes: np.ndarray | None = None
        self.prev_wave_trail: np.ndarray | None = None
        self.current_theme_idx = 0
        self.num_bands = 64

        # Variables para medir FPS reales
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()
        self._fps_display = 0

        self._setup_ui()
        self._apply_styles()
        self._load_devices()
        self._apply_theme(0)

        # Temporizador a ~60 FPS (cada 16 ms)
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._update_frame)

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)

        # --- Encabezado ---
        header_row = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)

        self.title_label = QLabel("Visualizador de Audio · Módulo 1")
        self.title_label.setObjectName("TitleLabel")
        self.subtitle_label = QLabel(
            "Captura en tiempo real, función ventana (Hann/Hamming) y Transformada Rápida de Fourier (FFT)."
        )
        self.subtitle_label.setObjectName("SubtitleLabel")

        title_box.addWidget(self.title_label)
        title_box.addWidget(self.subtitle_label)
        header_row.addLayout(title_box)
        header_row.addStretch()
        root.addLayout(header_row)

        # --- Contenedor principal del lienzo ---
        self.canvas_card = QFrame()
        self.canvas_card.setObjectName("Card")
        canvas_layout = QVBoxLayout(self.canvas_card)
        canvas_layout.setContentsMargins(14, 14, 14, 14)
        canvas_layout.setSpacing(10)

        # Barra superior interna (Badge de micrófono + HUD de métricas)
        hud_row = QHBoxLayout()

        self.mic_badge = QLabel("● Micrófono · En espera")
        self.mic_badge.setObjectName("MicBadge")
        hud_row.addWidget(self.mic_badge, alignment=Qt.AlignmentFlag.AlignTop)

        hud_row.addStretch()

        self.stats_card = QFrame()
        self.stats_card.setObjectName("StatsCard")
        stats_layout = QVBoxLayout(self.stats_card)
        stats_layout.setContentsMargins(12, 8, 12, 8)
        stats_layout.setSpacing(3)

        self.vol_label = QLabel("Volumen:        0 %")
        self.freq_label = QLabel("Frecuencia pico: 0.0 Hz")
        self.fps_label = QLabel("FPS:            0 fps")
        for lbl in (self.vol_label, self.freq_label, self.fps_label):
            lbl.setObjectName("StatValue")
            stats_layout.addWidget(lbl)

        hud_row.addWidget(self.stats_card)
        canvas_layout.addLayout(hud_row)

        # Área de gráficos con 4 modos
        self.stack = QStackedWidget()

        # Modo 0: Barras (64 bandas logarítmicas)
        self.bars_plot = self._create_plot_widget(show_axes=False)
        self.bars_plot.setXRange(-1, self.num_bands, padding=0)
        self.bars_plot.setYRange(0, 1.0, padding=0)
        self.bar_item = pg.BarGraphItem(
            x=np.arange(self.num_bands),
            height=np.zeros(self.num_bands),
            width=0.72,
            brush=pg.mkBrush("#22d3ee"),
            pen=None,
        )
        self.bars_plot.addItem(self.bar_item)
        self.stack.addWidget(self.bars_plot)

        # Modo 1: Onda (Dominio del tiempo)
        self.wave_plot = self._create_plot_widget(show_axes=False)
        self.wave_plot.setXRange(0, 1024, padding=0)
        self.wave_plot.setYRange(-1.05, 1.05, padding=0)
        self.wave_trail_curve = self.wave_plot.plot(pen=pg.mkPen("#f43f5e", width=1.6))
        self.wave_glow_curve = self.wave_plot.plot(pen=pg.mkPen("#22d3ee", width=3.2))
        self.wave_main_curve = self.wave_plot.plot(pen=pg.mkPen("#ffffff", width=1.8))
        self.stack.addWidget(self.wave_plot)

        # Modo 2: Circular
        self.circular_plot = self._create_plot_widget(show_axes=False)
        self.circular_plot.setAspectLocked(True)
        self.circular_plot.setXRange(-1.25, 1.25, padding=0)
        self.circular_plot.setYRange(-1.25, 1.25, padding=0)
        base_theta = np.linspace(0, 2 * np.pi, 180)
        self.circular_plot.plot(
            0.42 * np.cos(base_theta),
            0.42 * np.sin(base_theta),
            pen=pg.mkPen("#1f2937", width=1.2),
        )
        self.circular_outer = self.circular_plot.plot(pen=pg.mkPen("#22d3ee", width=2.4))
        self.circular_inner = self.circular_plot.plot(pen=pg.mkPen("#f43f5e", width=1.5))
        self.stack.addWidget(self.circular_plot)

        # Modo 3: Dual (Onda arriba + Espectro FFT en Hz abajo)
        dual_widget = QWidget()
        dual_layout = QVBoxLayout(dual_widget)
        dual_layout.setContentsMargins(0, 0, 0, 0)
        dual_layout.setSpacing(6)

        self.dual_wave_plot = self._create_plot_widget(show_axes=True, bottom_label="Muestras (tiempo)")
        self.dual_wave_plot.setXRange(0, 1024, padding=0)
        self.dual_wave_plot.setYRange(-1.0, 1.0, padding=0)
        self.dual_wave_curve = self.dual_wave_plot.plot(pen=pg.mkPen("#22d3ee", width=1.8))

        self.dual_fft_plot = self._create_plot_widget(show_axes=True, bottom_label="Frecuencia (Hz)")
        self.dual_fft_plot.setXRange(MIN_FREQ, MAX_FREQ, padding=0)
        self.dual_fft_plot.setYRange(0, 0.35, padding=0)
        self.dual_fft_curve = self.dual_fft_plot.plot(
            pen=pg.mkPen("#f43f5e", width=1.8),
            fillLevel=0,
            brush=pg.mkBrush(244, 63, 94, 45),
        )
        self.dual_peak_point = pg.ScatterPlotItem(size=10, brush=pg.mkBrush("#ffffff"), pen=pg.mkPen("#22d3ee", width=2))
        self.dual_fft_plot.addItem(self.dual_peak_point)

        dual_layout.addWidget(self.dual_wave_plot)
        dual_layout.addWidget(self.dual_fft_plot)
        self.stack.addWidget(dual_widget)

        # Empezamos en el modo Onda (índice 1)
        self.stack.setCurrentIndex(1)
        canvas_layout.addWidget(self.stack, stretch=1)
        root.addWidget(self.canvas_card, stretch=1)

        # --- Panel inferior de controles ---
        self.controls_card = QFrame()
        self.controls_card.setObjectName("Card")
        controls_layout = QVBoxLayout(self.controls_card)
        controls_layout.setContentsMargins(16, 12, 16, 12)
        controls_layout.setSpacing(10)

        # Fila 1 de controles
        row1 = QHBoxLayout()
        row1.setSpacing(12)

        # Selector de fuente (micrófono)
        mic_col = QVBoxLayout()
        mic_col.setSpacing(4)
        mic_title = QLabel("Fuente")
        mic_title.setObjectName("ControlTitle")
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(210)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        mic_col.addWidget(mic_title)
        mic_col.addWidget(self.device_combo)
        row1.addLayout(mic_col)

        # Botón Iniciar / Detener
        self.toggle_btn = QPushButton("🎙 Iniciar")
        self.toggle_btn.setObjectName("ToggleBtn")
        self.toggle_btn.setProperty("running", False)
        self.toggle_btn.setFixedHeight(38)
        self.toggle_btn.setMinimumWidth(120)
        self.toggle_btn.clicked.connect(self._toggle_capture)
        row1.addWidget(self.toggle_btn, alignment=Qt.AlignmentFlag.AlignBottom)

        # Selector de modo
        mode_frame = QFrame()
        mode_frame.setObjectName("ModeGroup")
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(4, 4, 4, 4)
        mode_layout.setSpacing(4)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for idx, name in enumerate(("Barras", "Onda", "Circular", "Dual")):
            btn = QPushButton(name)
            btn.setObjectName("ModeBtn")
            btn.setCheckable(True)
            if idx == 1:
                btn.setChecked(True)
            self.mode_group.addButton(btn, idx)
            mode_layout.addWidget(btn)

        self.mode_group.idClicked.connect(self.stack.setCurrentIndex)
        row1.addWidget(mode_frame, alignment=Qt.AlignmentFlag.AlignBottom)

        # Slider Sensibilidad
        sens_col = QVBoxLayout()
        sens_col.setSpacing(4)
        self.sens_label = QLabel("Sensibilidad: 1.0")
        self.sens_label.setObjectName("ControlTitle")
        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(2, 40)  # 0.2 a 4.0
        self.sens_slider.setValue(10)
        self.sens_slider.valueChanged.connect(
            lambda v: self.sens_label.setText(f"Sensibilidad: {v / 10:.1f}")
        )
        sens_col.addWidget(self.sens_label)
        sens_col.addWidget(self.sens_slider)
        row1.addLayout(sens_col, stretch=1)

        controls_layout.addLayout(row1)

        # Fila 2 de controles (Suavizado, Ventana FFT y Tema)
        row2 = QHBoxLayout()
        row2.setSpacing(16)

        # Slider Suavizado
        smooth_col = QHBoxLayout()
        self.smooth_label = QLabel("Suavizado: 0.70")
        self.smooth_label.setObjectName("ControlTitle")
        self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.smooth_slider.setRange(0, 90)
        self.smooth_slider.setValue(70)
        self.smooth_slider.setFixedWidth(180)
        self.smooth_slider.valueChanged.connect(
            lambda v: self.smooth_label.setText(f"Suavizado: {v / 100:.2f}")
        )
        smooth_col.addWidget(self.smooth_label)
        smooth_col.addWidget(self.smooth_slider)
        row2.addLayout(smooth_col)

        # Selector de Función Ventana (para analizar fuga espectral)
        win_col = QHBoxLayout()
        win_lbl = QLabel("Ventana FFT:")
        win_lbl.setObjectName("ControlTitle")
        self.window_combo = QComboBox()
        self.window_combo.addItem("Hann (recomendada)", "hann")
        self.window_combo.addItem("Hamming", "hamming")
        self.window_combo.addItem("Rectangular (sin ventana)", "boxcar")
        win_col.addWidget(win_lbl)
        win_col.addWidget(self.window_combo)
        row2.addLayout(win_col)

        row2.addStretch()

        # Selector de Tema de color
        theme_lbl = QLabel("Tema:")
        theme_lbl.setObjectName("ControlTitle")
        row2.addWidget(theme_lbl)

        for idx, theme in enumerate(THEMES):
            t_btn = QPushButton()
            t_btn.setFixedSize(22, 22)
            t_btn.setToolTip(theme["name"])
            t_btn.setStyleSheet(
                f"background-color: {theme['primary']}; border-radius: 6px; border: 2px solid #334155;"
            )
            t_btn.clicked.connect(lambda _=False, i=idx: self._apply_theme(i))
            row2.addWidget(t_btn)

        controls_layout.addLayout(row2)
        root.addWidget(self.controls_card)

    def _create_plot_widget(self, show_axes: bool, bottom_label: str = "") -> pg.PlotWidget:
        pw = pg.PlotWidget()
        pw.setBackground("#090b10")
        pw.setMouseEnabled(x=False, y=False)
        pw.hideButtons()
        pw.setMenuEnabled(False)
        if show_axes:
            pw.showGrid(x=True, y=True, alpha=0.18)
            pw.getAxis("left").setPen("#475569")
            pw.getAxis("bottom").setPen("#475569")
            if bottom_label:
                pw.setLabel("bottom", bottom_label, color="#94a3b8")
        else:
            pw.showAxis("left", False)
            pw.showAxis("bottom", False)
        return pw

    def _load_devices(self) -> None:
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItem("Predeterminado del sistema", None)
        for idx, name in list_input_devices():
            self.device_combo.addItem(f"{idx}: {name}", idx)
        self.device_combo.blockSignals(False)

    def _on_device_changed(self) -> None:
        dev_idx = self.device_combo.currentData()
        try:
            self.capture.set_device(dev_idx)
            if self.is_running:
                self._update_mic_badge()
        except Exception as exc:
            QMessageBox.warning(self, "Error de micrófono", f"No se pudo abrir el dispositivo:\n{exc}")

    def _toggle_capture(self) -> None:
        if self.is_running:
            self.timer.stop()
            self.capture.stop()
            self.is_running = False
            self.toggle_btn.setText("🎙 Iniciar")
            self.toggle_btn.setProperty("running", False)
            self.mic_badge.setText("● Micrófono · Detenido")
        else:
            try:
                self.capture.start()
                self.timer.start()
                self.is_running = True
                self.toggle_btn.setText("■ Detener")
                self.toggle_btn.setProperty("running", True)
                self._update_mic_badge()
            except Exception as exc:
                QMessageBox.warning(self, "Error al iniciar audio", str(exc))

        self.toggle_btn.style().unpolish(self.toggle_btn)
        self.toggle_btn.style().polish(self.toggle_btn)

    def _update_mic_badge(self) -> None:
        name = self.device_combo.currentText()
        short_name = name if len(name) <= 28 else name[:25] + "..."
        self.mic_badge.setText(f"● Micrófono · {short_name}")

    def _apply_theme(self, idx: int) -> None:
        self.current_theme_idx = idx
        t = THEMES[idx]
        self.bar_item.setOpts(brush=pg.mkBrush(t["primary"]))
        self.wave_main_curve.setPen(pg.mkPen(t["highlight"], width=1.8))
        self.wave_glow_curve.setPen(pg.mkPen(t["primary"], width=3.2))
        self.wave_trail_curve.setPen(pg.mkPen(t["secondary"], width=1.5))
        self.circular_outer.setPen(pg.mkPen(t["primary"], width=2.4))
        self.circular_inner.setPen(pg.mkPen(t["secondary"], width=1.5))
        self.dual_wave_curve.setPen(pg.mkPen(t["primary"], width=1.8))
        self.dual_fft_curve.setPen(pg.mkPen(t["secondary"], width=1.8))

    def _update_frame(self) -> None:
        samples = self.capture.get_latest_samples()
        sensitivity = self.sens_slider.value() / 10.0
        smoothing = self.smooth_slider.value() / 100.0
        window_type = self.window_combo.currentData()

        frame = analyze_frame(
            samples=samples,
            sample_rate=SAMPLE_RATE,
            window_type=window_type,
            sensitivity=sensitivity,
            prev_magnitudes=self.prev_magnitudes,
            smoothing=smoothing,
        )
        self.prev_magnitudes = frame.magnitudes

        # Actualizar HUD de métricas
        vol_pct = min(int(frame.rms * 350), 100)
        self.vol_label.setText(f"Volumen:        {vol_pct:3d} %")
        self.freq_label.setText(f"Frecuencia pico: {frame.peak_freq:6.1f} Hz")

        self._frame_count += 1
        now = time.perf_counter()
        if now - self._last_fps_time >= 0.5:
            self._fps_display = int(self._frame_count / (now - self._last_fps_time))
            self._frame_count = 0
            self._last_fps_time = now
        self.fps_label.setText(f"FPS:            {self._fps_display:3d} fps")

        # Dibujar solo el modo visible en pantalla
        mode = self.stack.currentIndex()

        if mode in (0, 2):
            bands = compute_log_bands(
                frame.freqs, frame.magnitudes, num_bands=self.num_bands
            )
            scaled_bands = np.clip(bands * 6.0, 0.0, 1.0)

            if mode == 0:
                self.bar_item.setOpts(height=scaled_bands)
            else:
                theta = np.linspace(0, 2 * np.pi, self.num_bands, endpoint=False)
                r_outer = 0.45 + scaled_bands * 0.65
                r_inner = 0.43 - scaled_bands * 0.18
                theta_closed = np.append(theta, theta[0])
                r_outer_closed = np.append(r_outer, r_outer[0])
                r_inner_closed = np.append(r_inner, r_inner[0])
                self.circular_outer.setData(
                    r_outer_closed * np.cos(theta_closed),
                    r_outer_closed * np.sin(theta_closed),
                )
                self.circular_inner.setData(
                    r_inner_closed * np.cos(theta_closed),
                    r_inner_closed * np.sin(theta_closed),
                )

        elif mode in (1, 3):
            wave = frame.waveform
            half = len(wave) // 2
            crossings = np.where((wave[: half - 1] <= 0) & (wave[1:half] > 0))[0]
            start = int(crossings[0]) if len(crossings) > 0 else 0
            slice_wave = np.clip(wave[start : start + 1024], -1.0, 1.0)

            if mode == 1:
                if self.prev_wave_trail is None or len(self.prev_wave_trail) != len(slice_wave):
                    self.prev_wave_trail = slice_wave.copy()
                else:
                    self.prev_wave_trail = self.prev_wave_trail * 0.82 + slice_wave * 0.18

                x_axis = np.arange(len(slice_wave))
                self.wave_trail_curve.setData(x_axis, self.prev_wave_trail * 1.15)
                self.wave_glow_curve.setData(x_axis, slice_wave)
                self.wave_main_curve.setData(x_axis, slice_wave)
            else:
                self.dual_wave_curve.setData(np.arange(len(slice_wave)), slice_wave)
                mask = (frame.freqs >= MIN_FREQ) & (frame.freqs <= MAX_FREQ)
                f_axis = frame.freqs[mask]
                m_axis = frame.magnitudes[mask]
                self.dual_fft_curve.setData(f_axis, m_axis)

                if frame.peak_freq > 0:
                    peak_mag = float(np.interp(frame.peak_freq, f_axis, m_axis))
                    self.dual_peak_point.setData([frame.peak_freq], [peak_mag])
                else:
                    self.dual_peak_point.setData([], [])

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #11151d;
                color: #e2e8f0;
                font-family: 'Segoe UI', 'Noto Sans', sans-serif;
            }
            #TitleLabel {
                font-size: 22px;
                font-weight: 700;
                color: #f8fafc;
            }
            #SubtitleLabel {
                font-size: 13px;
                color: #94a3b8;
            }
            #Card {
                background-color: #181d27;
                border: 1px solid #262e3d;
                border-radius: 12px;
            }
            #StatsCard {
                background-color: #0e121a;
                border: 1px solid #262e3d;
                border-radius: 8px;
            }
            #StatValue {
                font-family: monospace;
                font-size: 12px;
                color: #38bdf8;
            }
            #MicBadge {
                background-color: #0e121a;
                border: 1px solid #22d3ee;
                border-radius: 12px;
                padding: 5px 12px;
                font-size: 12px;
                color: #cbd5e1;
            }
            #ControlTitle {
                font-size: 12px;
                color: #94a3b8;
            }
            QComboBox {
                background-color: #0f131b;
                border: 1px solid #2e3748;
                border-radius: 8px;
                padding: 6px 10px;
                color: #f1f5f9;
            }
            #ToggleBtn {
                background-color: #0ea5e9;
                color: #ffffff;
                font-weight: 700;
                border-radius: 18px;
                padding: 0 18px;
                border: none;
            }
            #ToggleBtn[running="true"] {
                background-color: #ff5a5f;
            }
            #ModeGroup {
                background-color: #0f131b;
                border: 1px solid #262e3d;
                border-radius: 8px;
            }
            #ModeBtn {
                background-color: transparent;
                color: #94a3b8;
                border: none;
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 12px;
            }
            #ModeBtn:checked {
                background-color: #283142;
                color: #ffffff;
                font-weight: 600;
            }
            """
        )

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.capture.stop()
        super().closeEvent(event)