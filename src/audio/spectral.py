"""
Módulo de preprocesamiento y análisis espectral (Ventanas y FFT).
"""

from dataclasses import dataclass
import numpy as np
from scipy import fft, signal

from src.config import MAX_FREQ, MIN_FREQ, SAMPLE_RATE, SILENCE_THRESHOLD, WINDOW_TYPE


@dataclass
class SpectralFrame:
    """Resultado del análisis de un instante de audio."""

    waveform: np.ndarray        # Señal en el dominio del tiempo (con sensibilidad)
    freqs: np.ndarray           # Eje de frecuencias (Hz)
    magnitudes: np.ndarray      # Magnitud del espectro |X(f)|
    rms: float                  # Volumen RMS (0.0 a 1.0 aprox.)
    peak_freq: float            # Frecuencia dominante en Hz (0.0 si hay silencio)


def compute_rms(samples: np.ndarray) -> float:
    """Calcula la amplitud cuadrática media (Root Mean Square) de la señal."""
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def apply_window(samples: np.ndarray, window_type: str = WINDOW_TYPE) -> tuple[np.ndarray, np.ndarray]:
    """
    Multiplica la señal por una función ventana (Hann o Hamming) para llevar los
    extremos suavemente a cero y reducir la fuga espectral (spectral leakage).
    """
    window = signal.windows.get_window(window_type, len(samples), fftbins=True)
    return samples * window, window


def compute_fft(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    window_type: str = WINDOW_TYPE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la Transformada Rápida de Fourier Real (rfft).
    Devuelve (frecuencias_hz, magnitudes_normalizadas).
    """
    # 1. Eliminar desplazamiento continuo (DC offset) para no distorsionar 0 Hz
    centered = samples - np.mean(samples)

    # 2. Aplicar función ventana
    windowed, window = apply_window(centered, window_type=window_type)

    # 3. Calcular FFT solo para frecuencias positivas (la señal de audio es real)
    spectrum = fft.rfft(windowed)
    freqs = fft.rfftfreq(len(windowed), d=1.0 / sample_rate)

    # 4. Obtener magnitud y normalizar respecto a la suma de la ventana
    window_sum = np.sum(window)
    magnitudes = (2.0 / window_sum) * np.abs(spectrum)
    magnitudes[0] /= 2.0  # El componente 0 Hz no tiene contraparte negativa

    return freqs, magnitudes


def find_peak_frequency(
    freqs: np.ndarray,
    magnitudes: np.ndarray,
    min_freq: float = MIN_FREQ,
    max_freq: float = MAX_FREQ,
) -> float:
    """Encuentra la frecuencia con mayor amplitud dentro del rango musical."""
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(mask):
        return 0.0

    masked_freqs = freqs[mask]
    masked_mags = magnitudes[mask]
    peak_idx = int(np.argmax(masked_mags))
    return float(masked_freqs[peak_idx])


def smooth_spectrum(
    current: np.ndarray,
    previous: np.ndarray | None,
    smoothing: float,
) -> np.ndarray:
    """
    Suavizado exponencial temporal (EMA) entre el espectro anterior y el actual.
    smoothing=0.0 -> sin suavizado; smoothing=0.9 -> transición muy suave.
    """
    if previous is None or len(previous) != len(current):
        return current
    smoothing = float(np.clip(smoothing, 0.0, 0.95))
    return previous * smoothing + current * (1.0 - smoothing)


def analyze_frame(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    window_type: str = WINDOW_TYPE,
    sensitivity: float = 1.0,
    prev_magnitudes: np.ndarray | None = None,
    smoothing: float = 0.0,
) -> SpectralFrame:
    """Ejecuta todo el pipeline de preprocesamiento y análisis sobre una ventana."""
    scaled = samples * sensitivity
    rms = compute_rms(scaled)

    freqs, raw_magnitudes = compute_fft(
        scaled, sample_rate=sample_rate, window_type=window_type
    )
    magnitudes = smooth_spectrum(raw_magnitudes, prev_magnitudes, smoothing)

    if rms < SILENCE_THRESHOLD:
        peak_freq = 0.0
    else:
        peak_freq = find_peak_frequency(freqs, magnitudes)

    return SpectralFrame(
        waveform=scaled,
        freqs=freqs,
        magnitudes=magnitudes,
        rms=rms,
        peak_freq=peak_freq,
    )