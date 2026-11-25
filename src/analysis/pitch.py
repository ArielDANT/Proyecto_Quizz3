### Módulo 2:
"""
Detección de frecuencia fundamental (f0) mediante el algoritmo YIN
acelerado con FFT (correlación cruzada) e interpolación parabólica.
"""

import numpy as np
from scipy import signal

from src.audio.spectral import compute_rms
from src.config import SAMPLE_RATE, SILENCE_THRESHOLD


def detect_pitch_yin(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    min_freq: float = 55.0,
    max_freq: float = 3000.0,
    threshold: float = 0.15,
    silence_threshold: float = SILENCE_THRESHOLD,
) -> tuple[float, float]:
    """
    Estima la frecuencia fundamental f0 (Hz) y la confianza [0.0, 1.0].
    Devuelve (0.0, 0.0) si hay silencio o no hay tono periódico claro.
    """
    if len(samples) < 64 or compute_rms(samples) < silence_threshold:
        return 0.0, 0.0

    # Trabajar en float64 evita pérdida de precisión en sumas acumuladas
    x = samples.astype(np.float64) - np.mean(samples)
    w = len(x) // 2

    tau_min = max(2, int(sample_rate / max_freq))
    tau_max = min(w - 2, int(sample_rate / min_freq))
    if tau_min >= tau_max:
        return 0.0, 0.0

    # 1. Función de diferencia d(tau) = sum((x[j] - x[j+tau])^2) usando FFT
    x_win = x[:w]
    s1 = float(np.dot(x_win, x_win))

    cs = np.concatenate(([0.0], np.cumsum(x * x)))
    s2 = cs[w : w + w + 1] - cs[0 : w + 1]

    # Correlación rápida en O(N log N) mediante Transformada de Fourier
    cross = signal.correlate(x, x_win, mode="valid", method="fft")
    d = np.maximum(s1 + s2 - 2.0 * cross, 0.0)

    # 2. Normalización acumulada media (CMNDF) para evitar elegir múltiplos de periodo
    cmndf = np.ones_like(d)
    running_sum = np.cumsum(d[1:])
    lags = np.arange(1, len(d), dtype=np.float64)
    nonzero = running_sum > 1e-12
    cmndf[1:][nonzero] = d[1:][nonzero] * lags[nonzero] / running_sum[nonzero]

    # 3. Buscar el primer mínimo local por debajo del umbral
    best_tau = -1
    for tau in range(tau_min, tau_max + 1):
        if cmndf[tau] < threshold:
            while tau + 1 <= tau_max and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
            best_tau = tau
            break

    # Respaldo: si no cruzó el umbral, tomar el mínimo global si es suficientemente periódico
    if best_tau == -1:
        best_tau = tau_min + int(np.argmin(cmndf[tau_min : tau_max + 1]))
        if cmndf[best_tau] > 0.45:
            return 0.0, 0.0

    # 4. Interpolación parabólica alrededor de best_tau para precisión sub-muestra
    shift = 0.0
    if 0 < best_tau < len(cmndf) - 1:
        y0, y1, y2 = cmndf[best_tau - 1], cmndf[best_tau], cmndf[best_tau + 1]
        denom = 2.0 * (y0 - 2.0 * y1 + y2)
        if denom > 1e-12:
            shift = float(np.clip((y0 - y2) / denom, -0.5, 0.5))

    refined_tau = best_tau + shift
    if refined_tau <= 0:
        return 0.0, 0.0

    freq = float(sample_rate / refined_tau)
    confidence = float(np.clip(1.0 - cmndf[best_tau], 0.0, 1.0))
    return freq, confidence