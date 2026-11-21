"""
Módulo de adquisición de audio en tiempo real mediante sounddevice.
"""

import threading
from typing import Optional
import numpy as np
import sounddevice as sd

from src.config import BLOCK_SIZE, CHANNELS, FFT_SIZE, SAMPLE_RATE


def list_input_devices() -> list[tuple[int, str]]:
    """Devuelve una lista de (indice, nombre) para micrófonos disponibles."""
    devices = sd.query_devices()
    input_devices = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_devices.append((idx, dev["name"]))
    return input_devices


class AudioStreamCapture:
    """Captura audio del micrófono en segundo plano usando un buffer deslizante."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        buffer_size: int = FFT_SIZE,
        device_index: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.device_index = device_index

        self._buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Se ejecuta automáticamente cada vez que llegan muestras nuevas del micrófono."""
        mono = indata[:, 0]
        n = len(mono)

        with self._lock:
            if n >= self.buffer_size:
                self._buffer[:] = mono[-self.buffer_size :]
            else:
                # Desplaza las muestras antiguas a la izquierda e inserta las nuevas al final
                self._buffer[:-n] = self._buffer[n:]
                self._buffer[-n:] = mono

    def start(self) -> None:
        """Inicia la captura de audio en tiempo real."""
        if self._stream is not None:
            return

        self._stream = sd.InputStream(
            device=self.device_index,
            channels=CHANNELS,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Detiene y cierra el flujo del micrófono."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_device(self, device_index: Optional[int]) -> None:
        """Cambia el micrófono activo reiniciando el flujo si estaba encendido."""
        was_running = self._stream is not None
        if was_running:
            self.stop()
        self.device_index = device_index
        with self._lock:
            self._buffer.fill(0.0)
        if was_running:
            self.start()

    def get_latest_samples(self) -> np.ndarray:
        """Devuelve una copia segura del bloque actual de muestras."""
        with self._lock:
            return self._buffer.copy()