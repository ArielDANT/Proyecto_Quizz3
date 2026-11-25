"""
Lógica musical del afinador cromático (referencia estándar A4 = 440 Hz).
"""

import math
from dataclasses import dataclass
from typing import Optional

A4_REF = 440.0
NOTE_NAMES_EN = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_ES = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]


@dataclass
class TunerReading:
    """Resultado completo de afinación cromática para una frecuencia detectada."""

    detected_freq: float
    target_freq: float
    note_en: str
    note_es: str
    cents: float
    in_tune: bool
    confidence: float


def analyze_pitch(freq: float, confidence: float = 1.0) -> Optional[TunerReading]:
    """Convierte una frecuencia en Hz a la nota cromática más cercana y su error en cents."""
    if freq <= 0.0:
        return None

    midi_float = 69.0 + 12.0 * math.log2(freq / A4_REF)
    target_midi = int(round(midi_float))

    target_freq = A4_REF * (2.0 ** ((target_midi - 69) / 12.0))
    cents = 1200.0 * math.log2(freq / target_freq)

    note_idx = target_midi % 12
    octave = (target_midi // 12) - 1

    return TunerReading(
        detected_freq=freq,
        target_freq=target_freq,
        note_en=f"{NOTE_NAMES_EN[note_idx]}{octave}",
        note_es=f"{NOTE_NAMES_ES[note_idx]}{octave}",
        cents=cents,
        in_tune=abs(cents) <= 5.0,
        confidence=confidence,
    )