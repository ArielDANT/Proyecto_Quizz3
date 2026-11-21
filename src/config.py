"""
Configuración global de adquisición de audio y análisis espectral.
"""

# Frecuencia de muestreo (Hz). Por el Teorema de Nyquist-Shannon,
# podremos analizar frecuencias de hasta SAMPLE_RATE / 2 = 22,050 Hz.
SAMPLE_RATE = 44100

# Cantidad de muestras nuevas que entrega el micrófono en cada lectura.
# 1024 / 44100 ≈ 23.2 milisegundos de latencia por lectura.
BLOCK_SIZE = 1024

# Tamaño de la ventana de análisis para la FFT (debe ser potencia de 2).
# La resolución en frecuencia será: Δf = SAMPLE_RATE / FFT_SIZE ≈ 10.77 Hz.
FFT_SIZE = 4096

# Canal mono (1 canal es suficiente para análisis musical).
CHANNELS = 1

# Tipo de función ventana para evitar fuga espectral ("hann" o "hamming").
WINDOW_TYPE = "hann"

# Rango de frecuencias de interés musical (Hz).
# La nota más grave de una guitarra estándar (Mi2) es ~82.4 Hz.
MIN_FREQ = 50.0
MAX_FREQ = 4500.0

# Umbral mínimo de volumen RMS para no detectar ruido eléctrico como una nota.
SILENCE_THRESHOLD = 0.005